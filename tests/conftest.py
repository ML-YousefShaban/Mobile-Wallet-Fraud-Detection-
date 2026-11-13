"""Shared fixtures.  All data here is SYNTHETIC (see synthetic_paysim.py)."""
from pathlib import Path

import joblib
import pytest
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from paysim_features import ELIGIBLE_TYPES, build_preprocessing_pipeline
from synthetic_paysim import make_synthetic_paysim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_BUNDLE = PROJECT_ROOT / "paysim_fraud_bundle.pkl"


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic_paysim(12_000, scope_fraud_rate=0.04, seed=1)


@pytest.fixture(scope="session")
def scope_df(synthetic_df):
    return synthetic_df[synthetic_df["type"].isin(ELIGIBLE_TYPES)].reset_index(drop=True)


@pytest.fixture(scope="session")
def tiny_pipeline(scope_df):
    """A small but real pipeline (feature engineering + LightGBM) trained on synthetic data."""
    X, y = scope_df.drop(columns=["isFraud"]), scope_df["isFraud"]
    pipe = Pipeline([
        ("prep", build_preprocessing_pipeline()),
        ("classifier", LGBMClassifier(n_estimators=40, num_leaves=15, random_state=0,
                                      n_jobs=1, verbose=-1)),
    ])
    return pipe.fit(X, y)


@pytest.fixture(scope="session")
def tiny_bundle_path(tiny_pipeline, tmp_path_factory):
    path = tmp_path_factory.mktemp("bundle") / "paysim_fraud_bundle.pkl"
    joblib.dump({"pipeline": tiny_pipeline, "threshold": 0.30, "fp_cost": 50.0,
                 "metrics": {"val_pr_auc": 0.9, "test_pr_auc": 0.9}}, path)
    return path


@pytest.fixture(scope="session")
def real_bundle_path():
    if not REAL_BUNDLE.exists():
        pytest.skip("paysim_fraud_bundle.pkl is not present")
    return REAL_BUNDLE
