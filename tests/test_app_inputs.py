import io

import numpy as np
import pandas as pd
import pytest

from paysim_scoring import (InputValidationError, evaluate_ground_truth, prepare_batch,
                            read_csv_safely, score_batch, threshold_grid,
                            validate_single_transaction)

VALID = dict(step=26, type="TRANSFER", amount=1000.0, oldbalanceOrg=1000.0, newbalanceOrig=0.0,
             oldbalanceDest=0.0, newbalanceDest=1000.0)


def csv_bytes(df):
    return io.BytesIO(df.to_csv(index=False).encode())


class Upload(io.BytesIO):
    """Mimics Streamlit's UploadedFile, which exposes a .size attribute."""

    def __init__(self, data, size=None):
        super().__init__(data)
        self._size = len(data) if size is None else size

    @property
    def size(self):
        return self._size


# ------------------------------ single transaction ------------------------------ #
def args(**kw):
    base = dict(tx_type="TRANSFER", step=26, amount=1000.0, old_org=1000.0, new_org=0.0,
                old_dest=0.0, new_dest=1000.0)
    base.update(kw)
    return base


def test_single_valid_transaction_has_no_errors():
    errors, notices = validate_single_transaction(**args())
    assert errors == [] and notices == []


@pytest.mark.parametrize("bad,label", [
    (dict(amount=-1.0), "Amount"), (dict(old_org=-5.0), "Sender old balance"),
    (dict(new_org=-0.01), "Sender new balance"), (dict(old_dest=-1.0), "Recipient old balance"),
    (dict(new_dest=float("nan")), "Recipient new balance"), (dict(amount=float("inf")), "Amount"),
    (dict(amount="abc"), "Amount"), (dict(amount=None), "Amount"),
])
def test_single_invalid_numbers_are_rejected(bad, label):
    errors, _ = validate_single_transaction(**args(**bad))
    assert any(label in e for e in errors)


@pytest.mark.parametrize("bad", [dict(step=0), dict(step=745), dict(step=3.5), dict(tx_type="WIRE")])
def test_single_invalid_step_or_type(bad):
    errors, _ = validate_single_transaction(**args(**bad))
    assert errors


def test_single_unusual_but_valid_pattern_only_produces_notices():
    errors, notices = validate_single_transaction(**args(amount=5000.0, old_org=100.0, new_org=0.0,
                                                         old_dest=0.0, new_dest=0.0))
    assert errors == [] and len(notices) >= 2


def test_single_other_types_skip_balance_notices():
    errors, notices = validate_single_transaction(**args(tx_type="PAYMENT", new_org=7.0))
    assert errors == [] and notices == []


# ----------------------------------- CSV reading ----------------------------------- #
def test_read_valid_csv_and_strip_header_whitespace():
    df = read_csv_safely(io.BytesIO(b" step , type ,amount\n1,TRANSFER,5\n"))
    assert list(df.columns) == ["step", "type", "amount"] and len(df) == 1


def test_read_empty_and_header_only():
    with pytest.raises(InputValidationError, match="empty"):
        read_csv_safely(io.BytesIO(b""))
    with pytest.raises(InputValidationError, match="no data rows"):
        read_csv_safely(io.BytesIO(b"step,type,amount\n"))


def test_read_oversized_by_rows_and_bytes():
    data = csv_bytes(pd.DataFrame({"a": range(6)}))
    with pytest.raises(InputValidationError, match="more than 5 rows"):
        read_csv_safely(data, max_rows=5)
    with pytest.raises(InputValidationError, match="limit is"):
        read_csv_safely(Upload(b"a\n1\n", size=30_000_000), max_bytes=25_000_000)


def test_read_binary_and_malformed_files():
    with pytest.raises(InputValidationError):
        read_csv_safely(io.BytesIO(b"\xff\xfe\x00\x00\x80\x81"))
    for ragged in (b"a,b\n1,2,3,4,5\n6,7\n",          # first data row longer than the header
                   b"a,b\n1,2\n3,4,5,6\n"):              # a later row longer than the header
        with pytest.raises(InputValidationError, match="malformed"):
            read_csv_safely(io.BytesIO(ragged))


# ---------------------------------- batch preparation ---------------------------------- #
def base_frame():
    return pd.DataFrame([
        dict(VALID),
        dict(VALID, type="CASH_OUT", amount=50.0, oldbalanceOrg=500.0, newbalanceOrig=450.0),
        dict(VALID, type="PAYMENT"),
    ])


def test_prepare_valid_batch_and_status_counts():
    prepared, info = prepare_batch(base_frame())
    assert list(prepared["Status"]) == ["Eligible", "Eligible", "Not Fraud-Eligible"]
    assert info["status_counts"] == {"Eligible": 2, "Not Fraud-Eligible": 1}
    assert list(prepared["Input_Row"]) == [2, 3, 4]
    assert not info["ground_truth"]["present"]


def test_prepare_missing_columns_lists_them():
    with pytest.raises(InputValidationError) as err:
        prepare_batch(base_frame().drop(columns=["amount", "newbalanceDest"]))
    assert "amount" in str(err.value) and "newbalanceDest" in str(err.value)


def test_prepare_normalises_type_text():
    df = base_frame()
    df.loc[0, "type"] = "  transfer "
    prepared, _ = prepare_batch(df)
    assert prepared.loc[0, "Type_Clean"] == "TRANSFER" and prepared.loc[0, "Status"] == "Eligible"


@pytest.mark.parametrize("column,value,fragment", [
    ("amount", "abc", "not a number"), ("amount", None, "missing"),
    ("oldbalanceOrg", -5, "negative"), ("newbalanceDest", "inf", "not finite"),
    ("step", "x", "not a number"),
])
def test_prepare_flags_invalid_numbers_per_row(column, value, fragment):
    df = base_frame().astype({column: object})
    df.loc[0, column] = value
    prepared, info = prepare_batch(df)
    assert prepared.loc[0, "Status"] == "Invalid input" and fragment in prepared.loc[0, "Validation_Note"]
    assert prepared.loc[1, "Status"] == "Eligible"                     # other rows are unaffected
    assert info["status_counts"]["Invalid input"] == 1


def test_prepare_missing_and_unknown_type():
    df = base_frame()
    df.loc[0, "type"] = None
    df.loc[1, "type"] = "WIRE"
    prepared, _ = prepare_batch(df)
    assert list(prepared["Status"][:2]) == ["Invalid input", "Unsupported type"]


def test_garbage_numbers_in_non_eligible_rows_do_not_matter():
    df = base_frame().astype({"amount": object})
    df.loc[2, "amount"] = "oops"                                       # the PAYMENT row
    prepared, _ = prepare_batch(df)
    assert prepared.loc[2, "Status"] == "Not Fraud-Eligible"


def test_extra_columns_are_preserved():
    df = base_frame().assign(nameOrig="C1", comment="x")
    prepared, _ = prepare_batch(df)
    assert {"nameOrig", "comment"} <= set(prepared.columns)


@pytest.mark.parametrize("labels,valid", [([0, 1, 0], True), ([0.0, 1.0, 0.0], True),
                                          ([0, 2, 0], False), ([0, None, 1], False), (["a", 1, 0], False)])
def test_isfraud_validation(labels, valid):
    prepared, info = prepare_batch(base_frame().assign(isFraud=pd.Series(labels, dtype=object)))
    gt = info["ground_truth"]
    assert gt["present"] and gt["valid"] == valid
    assert bool(gt["message"]) != valid


def test_step_outside_paysim_range_adds_warning():
    df = base_frame()
    df.loc[0, "step"] = 5000
    _, info = prepare_batch(df)
    assert any("step" in w for w in info["warnings"])


# --------------------------------- ground-truth costs ---------------------------------- #
class StubModel:
    """predict_proba derived from the amount so that the expected result is known by hand."""
    LOOKUP = {100.0: 0.9, 200.0: 0.2, 300.0: 0.8, 400.0: 0.1}

    def predict_proba(self, X):
        p = X["amount"].map(self.LOOKUP).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def test_evaluate_ground_truth_matches_manual_calculation():
    rows = [dict(VALID, amount=a) for a in (100.0, 200.0, 300.0, 400.0)] + [dict(VALID, type="PAYMENT", amount=1000.0)]
    raw = pd.DataFrame(rows).assign(isFraud=[1, 1, 0, 0, 1])          # last row = fraud the model cannot score
    prepared, info = prepare_batch(raw)
    scored = score_batch(StubModel(), prepared, review_threshold=0.5, block_threshold=0.85)
    assert list(scored["Decision"][:4]) == ["Block", "Approve", "Manual Review", "Approve"]
    result = evaluate_ground_truth(scored, fp_cost=50.0, review_threshold=0.5)
    m = result["model"]
    assert (m["TP"], m["FP"], m["FN"]) == (1, 1, 1)                    # alerts: rows 1 and 3
    assert m["missed_fraud_amount"] == 200.0 and m["false_alarm_cost"] == 50.0 and m["total_cost"] == 250.0
    assert result["no_model"]["total_cost"] == 300.0 and result["savings"] == 50.0
    assert result["n_scored"] == 4 and result["n_fraud_scored"] == 2
    assert result["out_of_scope_fraud_count"] == 1 and result["out_of_scope_fraud_amount"] == 1000.0


def test_threshold_grid_contains_exact_defaults_without_near_duplicates():
    grid = threshold_grid(0.24000000000000002, 0.5)
    assert 0.24000000000000002 in grid and 0.5 in grid
    assert (np.diff(grid) > 1e-6).all() and np.all(np.diff(grid) > 0)
    assert grid.min() >= 1e-4 and grid.max() <= 1.0


# ======================================================================================= #
# Streamlit app, driven headlessly with streamlit.testing.v1.AppTest
# (a real browser session is NOT exercised here; see the README for what was tested)
# ======================================================================================= #
import shutil
from pathlib import Path

import joblib
from streamlit.testing.v1 import AppTest

from synthetic_paysim import make_synthetic_paysim

PROJECT = Path(__file__).resolve().parents[1]
APP_FILES = ["app.py", "paysim_features.py", "paysim_scoring.py"]


def make_app_dir(tmp_path, bundle_path=None, samples=False):
    """Copy the app into a scratch folder (optionally with a bundle and SYNTHETIC sample files)."""
    for name in APP_FILES:
        shutil.copy(PROJECT / name, tmp_path / name)
    if bundle_path is not None:
        shutil.copy(bundle_path, tmp_path / "paysim_fraud_bundle.pkl")
    if samples:
        df = make_synthetic_paysim(8_000, 0.04, seed=3)
        scope = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
        fraud, legit = scope[scope["isFraud"] == 1], scope[scope["isFraud"] == 0]
        pd.concat([fraud.sample(25, random_state=0), legit.sample(175, random_state=0)]) \
            .to_csv(tmp_path / "sample_transactions_demo.csv", index=False)
        scope.sample(1_000, random_state=0).to_csv(tmp_path / "sample_transactions_realistic.csv", index=False)
    return tmp_path


def start_app(app_dir):
    return AppTest.from_file(str(app_dir / "app.py"), default_timeout=120).run()


def click_score(at):
    next(b for b in at.button if "Score" in b.label).click()
    return at.run()


def texts(elements):
    return " | ".join(e.value for e in elements)


@pytest.fixture()
def app_dir(tiny_bundle_path, tmp_path):
    return make_app_dir(tmp_path, tiny_bundle_path, samples=True)


def test_app_starts_with_bundle_defaults(app_dir):
    at = start_app(app_dir)
    assert not at.exception and not at.error
    assert [s.value for s in at.sidebar.select_slider] == [0.30, 0.50]
    assert at.sidebar.number_input[0].value == 50.0
    assert "Validation PR-AUC" in [m.label for m in at.sidebar.metric]


def test_app_single_eligible_transaction_shows_decision_and_costs(app_dir):
    at = click_score(start_app(app_dir))
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert any(label.startswith("Fraud probability") for label in labels)
    assert any(label.startswith("Expected loss if approved") for label in labels)
    assert "Decision:" in texts(list(at.error) + list(at.warning) + list(at.success))


def test_app_single_not_eligible_type_is_not_scored_or_called_safe(app_dir):
    at = start_app(app_dir)
    at.selectbox[0].set_value("PAYMENT")
    at = click_score(at)
    assert not at.exception
    message = texts(at.info)
    assert "Not Fraud-Eligible" in message and "not a statement that the transaction is safe" in message
    assert not any(m.label.startswith("Fraud probability") for m in at.metric)


def test_app_missing_bundle_shows_clear_error(tmp_path):
    at = start_app(make_app_dir(tmp_path))
    assert not at.exception
    assert "was not found" in texts(at.error)


def test_app_incompatible_bundle_shows_clear_error(tmp_path):
    app_dir = make_app_dir(tmp_path)
    joblib.dump({"pipeline": "nope"}, app_dir / "paysim_fraud_bundle.pkl")
    at = start_app(app_dir)
    assert not at.exception and "incompatible" in texts(at.error)


def test_app_demo_sample_is_labelled_as_not_representative(app_dir):
    at = start_app(app_dir)
    at.radio[0].set_value("Demo sample (enriched - fraud deliberately over-represented)").run()
    assert not at.exception
    assert "enriched demonstration sample" in texts(at.warning)
    assert "12.5%" in texts(at.warning)                                # 25 frauds in 200 rows


def test_app_realistic_sample_has_no_demo_warning(app_dir):
    at = start_app(app_dir)                                            # realistic sample is the default
    assert not at.exception and "enriched demonstration sample" not in texts(at.warning)
    assert any(m.label == "Rows processed" and m.value == "1,000" for m in at.metric)


def test_app_without_sample_files_still_works(tiny_bundle_path, tmp_path):
    at = start_app(make_app_dir(tmp_path, tiny_bundle_path, samples=False))
    assert not at.exception
    assert at.radio[0].options == ["Upload my own CSV"]
    assert "Upload a CSV" in texts(at.info)


def upload(at, content, name="upload.csv"):
    at.radio[0].set_value("Upload my own CSV").run()
    at.file_uploader[0].upload(name, content, "text/csv").run()
    return at


def sample_rows(n=40, with_label=False):
    df = make_synthetic_paysim(6_000, 0.05, seed=9)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].head(n)
    cols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    return df[cols + (["isFraud"] if with_label else [])]


def test_app_upload_valid_csv_without_labels_has_no_cost_panel(app_dir):
    at = upload(start_app(app_dir), sample_rows().to_csv(index=False).encode())
    assert not at.exception and not at.error
    assert len(at.dataframe) == 1
    assert any("scored CSV" in b.label for b in at.get("download_button"))
    assert not any("Cost evaluation" in m.value for m in at.markdown)


def test_app_upload_csv_with_labels_shows_cost_panel(app_dir):
    at = upload(start_app(app_dir), sample_rows(with_label=True).to_csv(index=False).encode())
    assert not at.exception
    assert any("Cost evaluation" in m.value for m in at.markdown)
    assert any(m.label == "TP / FN" for m in at.metric)


@pytest.mark.parametrize("make_bytes,fragment", [
    (lambda: sample_rows().drop(columns=["amount"]).to_csv(index=False).encode(), "Missing required column"),
    (lambda: b"", "empty"),
    (lambda: b"step,type,amount\n", "no data rows"),
    (lambda: b"\xff\xfe\x00\x00\x80\x81", "UTF-8"),
])
def test_app_upload_bad_files_show_friendly_errors(app_dir, make_bytes, fragment):
    at = upload(start_app(app_dir), make_bytes())
    assert not at.exception                                            # never a raw traceback
    assert fragment in texts(at.error)


def test_app_upload_invalid_values_and_types_are_reported_not_crashing(app_dir):
    df = sample_rows(12, with_label=True).astype({"amount": object})
    df.loc[df.index[0], "amount"] = "abc"
    df.loc[df.index[1], "type"] = "PAYMENT"
    df.loc[df.index[2], "type"] = "WIRE"
    df.loc[df.index[3], "oldbalanceOrg"] = -10
    at = upload(start_app(app_dir), df.to_csv(index=False).encode())
    assert not at.exception
    assert "invalid values" in texts(at.warning) and "unrecognised transaction type" in texts(at.warning)
    assert "Not Fraud-Eligible" in texts(at.info)
    assert any(m.label == "Scored (eligible)" and m.value == "8" for m in at.metric)


def test_app_upload_bad_isfraud_disables_cost_evaluation_but_still_scores(app_dir):
    df = sample_rows(10, with_label=True).astype({"isFraud": object})
    df.loc[df.index[0], "isFraud"] = 7
    at = upload(start_app(app_dir), df.to_csv(index=False).encode())
    assert not at.exception
    assert "must contain only 0 or 1" in texts(at.error)
    assert len(at.dataframe) == 1 and not any(m.label == "TP / FN" for m in at.metric)


def test_app_real_bundle_defaults(project_root, real_bundle_path, tmp_path):
    at = start_app(make_app_dir(tmp_path, real_bundle_path))
    assert not at.exception and not at.error
    assert at.sidebar.select_slider[0].value == joblib.load(real_bundle_path)["threshold"]
    assert click_score(at).exception == []
