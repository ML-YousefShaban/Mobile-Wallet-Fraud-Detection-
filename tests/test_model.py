import json
import pickle
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from paysim_features import (CalibratedPipeline, apply_logit_platt, fit_logit_platt,
                             model_feature_names)
from paysim_scoring import (BundleError, load_bundle, prepare_batch, runtime_version_warnings,
                            score_batch)

FRAUD_LIKE = dict(step=100, type="TRANSFER", amount=250000.0, oldbalanceOrg=250000.0,
                  newbalanceOrig=0.0, oldbalanceDest=0.0, newbalanceDest=0.0)
LEGIT_LIKE = dict(step=300, type="CASH_OUT", amount=5000.0, oldbalanceOrg=50000.0,
                  newbalanceOrig=45000.0, oldbalanceDest=100000.0, newbalanceDest=105000.0)

FRESH_PROCESS_CODE = """
import json, sys
sys.path.insert(0, {root!r})
import pandas as pd
from paysim_scoring import load_bundle
bundle = load_bundle({bundle!r})
X = pd.read_pickle({x!r})
print(json.dumps(bundle["pipeline"].predict_proba(X)[:, 1].tolist()))
"""


def predict_in_fresh_process(project_root, bundle_path, X, tmp_path):
    """Load the bundle in a NEW interpreter (no notebook state, no __main__ injection)."""
    x_path = tmp_path / "x.pkl"
    X.to_pickle(x_path)
    code = FRESH_PROCESS_CODE.format(root=str(project_root), bundle=str(bundle_path), x=str(x_path))
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode == 0, run.stderr[-800:]
    return np.array(json.loads(run.stdout.strip().splitlines()[-1]))


# ----------------------------- synthetic (always runs) ----------------------------- #
def test_bundle_roundtrip_in_fresh_process(project_root, tiny_bundle_path, tiny_pipeline, scope_df, tmp_path):
    X = scope_df.drop(columns=["isFraud"]).iloc[:50]
    fresh = predict_in_fresh_process(project_root, tiny_bundle_path, X, tmp_path)
    np.testing.assert_allclose(fresh, tiny_pipeline.predict_proba(X)[:, 1], rtol=0, atol=1e-12)
    assert ((fresh >= 0) & (fresh <= 1)).all()


def test_pickle_references_module_not_main(tiny_bundle_path):
    raw = tiny_bundle_path.read_bytes()
    assert b"paysim_features" in raw and b"__main__" not in raw


def test_probabilities_are_valid_and_shaped(tiny_pipeline, scope_df):
    X = scope_df.drop(columns=["isFraud"]).iloc[:200]
    proba = tiny_pipeline.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert len(model_feature_names(tiny_pipeline.named_steps["prep"])) == 17


def test_fraud_like_scores_above_legit_like(tiny_pipeline):
    p = tiny_pipeline.predict_proba(pd.DataFrame([FRAUD_LIKE, LEGIT_LIKE]))[:, 1]
    assert p[0] > 0.5 > p[1]


def test_load_bundle_error_messages(tmp_path):
    with pytest.raises(BundleError, match="not found"):
        load_bundle(tmp_path / "missing.pkl")
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"this is not a pickle")
    with pytest.raises(BundleError, match="could not be loaded"):
        load_bundle(bad)
    wrong = tmp_path / "wrong.pkl"
    joblib.dump({"pipeline": object(), "threshold": 2.0, "fp_cost": -1, "metrics": []}, wrong)
    with pytest.raises(BundleError) as err:
        load_bundle(wrong)
    msg = str(err.value)
    assert "predict_proba" in msg and "threshold" in msg and "fp_cost" in msg and "metrics" in msg
    incomplete = tmp_path / "incomplete.pkl"
    joblib.dump({"pipeline": 1}, incomplete)
    with pytest.raises(BundleError, match="missing key"):
        load_bundle(incomplete)


def test_unsupported_types_are_not_scored(tiny_pipeline):
    raw = pd.DataFrame([
        dict(FRAUD_LIKE), dict(LEGIT_LIKE),
        dict(LEGIT_LIKE, type="PAYMENT"), dict(LEGIT_LIKE, type="CASH_IN"),
        dict(LEGIT_LIKE, type="DEBIT"), dict(LEGIT_LIKE, type="WIRE"),
    ])
    prepared, info = prepare_batch(raw)
    scored = score_batch(tiny_pipeline, prepared, 0.3, 0.5)
    assert list(scored["Status"]) == ["Eligible", "Eligible", "Not Fraud-Eligible", "Not Fraud-Eligible",
                                      "Not Fraud-Eligible", "Unsupported type"]
    assert scored["Fraud_Probability"].iloc[:2].notna().all()
    assert scored["Fraud_Probability"].iloc[2:].isna().all()          # never a fake "low risk" score
    assert list(scored["Decision"].iloc[2:]) == ["Not scored"] * 4
    assert scored["Expected_Loss_If_Approved"].iloc[2:].isna().all()


def test_batch_scores_equal_direct_pipeline_scores(tiny_pipeline, scope_df):
    """The app path (score_batch) and the notebook path (pipeline.predict_proba) must agree."""
    raw = scope_df.drop(columns=["isFraud"]).iloc[:300]
    prepared, _ = prepare_batch(raw)
    scored = score_batch(tiny_pipeline, prepared, 0.3, 0.5)
    np.testing.assert_allclose(scored["Fraud_Probability"], tiny_pipeline.predict_proba(raw)[:, 1],
                               rtol=0, atol=1e-12)


def test_runtime_version_warning_logic(tiny_bundle_path):
    bundle = load_bundle(tiny_bundle_path)
    assert runtime_version_warnings(bundle) == []                      # no 'versions' key -> nothing to compare
    bundle["versions"] = {"scikit-learn": "0.0.1", "lightgbm": "0.0.1"}
    messages = runtime_version_warnings(bundle)
    assert len(messages) == 2 and "scikit-learn" in messages[0]


# ------------------------------ calibration helpers ------------------------------ #
def test_calibrated_pipeline_identity_and_ranking(tiny_pipeline, scope_df):
    X = scope_df.drop(columns=["isFraud"]).iloc[:300]
    raw = tiny_pipeline.predict_proba(X)[:, 1]
    same = CalibratedPipeline(tiny_pipeline, slope=1.0, intercept=0.0).predict_proba(X)[:, 1]
    np.testing.assert_allclose(same, np.clip(raw, 1e-15, 1 - 1e-15), rtol=1e-6, atol=1e-12)
    soft = CalibratedPipeline(tiny_pipeline, slope=0.5, intercept=-1.0).predict_proba(X)[:, 1]
    order = np.argsort(raw, kind="stable")
    assert (np.diff(soft[order]) >= -1e-15).all()                      # monotonic: ranking is preserved
    assert ((soft >= 0) & (soft <= 1)).all()
    clone = pickle.loads(pickle.dumps(CalibratedPipeline(tiny_pipeline, 0.5, -1.0)))
    np.testing.assert_allclose(clone.predict_proba(X), CalibratedPipeline(tiny_pipeline, 0.5, -1.0).predict_proba(X))


def test_platt_fix_reduces_log_loss_of_overconfident_scores():
    rng = np.random.default_rng(0)
    true_p = rng.beta(0.5, 5, 20_000)
    y = (rng.random(20_000) < true_p).astype(int)
    logit = np.log(true_p / (1 - true_p))
    overconfident = 1 / (1 + np.exp(-3.0 * logit))                     # too extreme on both sides
    slope, intercept = fit_logit_platt(overconfident, y)
    assert 0.2 < slope < 0.5                                           # ~1/3 undoes the 3x stretch
    fixed = apply_logit_platt(overconfident, slope, intercept)
    assert log_loss(y, fixed) < 0.9 * log_loss(y, np.clip(overconfident, 1e-15, 1 - 1e-15))


# ------------------- the real bundle shipped with the project -------------------- #
def test_real_bundle_loads_in_fresh_process_without_main_hack(project_root, real_bundle_path, tmp_path):
    X = pd.DataFrame([FRAUD_LIKE, LEGIT_LIKE])
    p = predict_in_fresh_process(project_root, real_bundle_path, X, tmp_path)
    bundle = load_bundle(real_bundle_path)
    assert p[0] >= bundle["threshold"] > p[1]
    assert ((p >= 0) & (p <= 1)).all()


def test_real_bundle_contents(real_bundle_path):
    bundle = load_bundle(real_bundle_path)
    assert 0 < bundle["threshold"] < 1 and bundle["fp_cost"] == 50.0
    for key in ("val_pr_auc", "test_pr_auc", "test_roc_auc"):
        assert 0.5 < bundle["metrics"][key] <= 1.0
    assert len(model_feature_names(bundle["pipeline"].named_steps["prep"])) == 17
