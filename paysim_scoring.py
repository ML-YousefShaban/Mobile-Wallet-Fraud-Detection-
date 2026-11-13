"""Bundle loading, input validation and batch scoring for the Streamlit app.

Everything here is plain Python/pandas (no Streamlit), so it can be unit-tested and
reused from the notebook.  All user-facing failures raise ``BundleError`` or
``InputValidationError`` with a readable message instead of a raw traceback.
"""
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from paysim_features import (
    ELIGIBLE_TYPES, KNOWN_TYPES, NUMERIC_COLUMNS, REQUIRED_COLUMNS,
    cost_of, decision_tier,
)

MAX_ROWS = 100_000
MAX_FILE_BYTES = 25_000_000
STEP_RANGE = (1, 744)          # hourly steps seen in PaySim (about 31 days)

STATUS_ELIGIBLE = "Eligible"
STATUS_NOT_ELIGIBLE = "Not Fraud-Eligible"
STATUS_UNSUPPORTED = "Unsupported type"
STATUS_INVALID = "Invalid input"


class BundleError(Exception):
    """The model bundle is missing, unreadable or has an unexpected structure."""


class InputValidationError(ValueError):
    """User-supplied data cannot be processed (message is safe to show in the UI)."""


# --------------------------------------------------------------------------- #
# Model bundle
# --------------------------------------------------------------------------- #
def _bundle_problems(bundle):
    if not isinstance(bundle, dict):
        return ["the file does not contain a dictionary"]
    problems = [f"missing key '{k}'" for k in ("pipeline", "threshold", "fp_cost", "metrics")
                if k not in bundle]
    if problems:
        return problems
    if not hasattr(bundle["pipeline"], "predict_proba"):
        problems.append("'pipeline' has no predict_proba method")
    for key, low, high in (("threshold", 0.0, 1.0), ("block_threshold", 0.0, 1.0)):
        if key in bundle:
            try:
                v = float(bundle[key])
                if not (low < v <= high):
                    problems.append(f"'{key}' must be in ({low}, {high}]")
            except (TypeError, ValueError):
                problems.append(f"'{key}' is not a number")
    try:
        if not (math.isfinite(float(bundle["fp_cost"])) and float(bundle["fp_cost"]) > 0):
            problems.append("'fp_cost' must be a positive number")
    except (TypeError, ValueError):
        problems.append("'fp_cost' is not a number")
    if not isinstance(bundle["metrics"], dict):
        problems.append("'metrics' must be a dictionary")
    return problems


def load_bundle(path):
    """Load and validate ``paysim_fraud_bundle.pkl``."""
    path = Path(path)
    if not path.exists():
        raise BundleError(
            f"Model file '{path.name}' was not found next to app.py. Run the final export "
            "cell of next_gen.ipynb to create it, or copy the file into the project folder."
        )
    try:
        bundle = joblib.load(path)
    except ModuleNotFoundError as exc:
        raise BundleError(
            f"The model needs the Python module '{exc.name}'. Install requirements.txt and "
            "keep paysim_features.py in the same folder as app.py."
        ) from exc
    except Exception as exc:   # noqa: BLE001 - any unpickling failure must be reported nicely
        raise BundleError(
            f"The model file could not be loaded ({type(exc).__name__}). This usually means the "
            "installed scikit-learn / lightgbm / numpy versions differ from the training "
            "environment - install the versions pinned in requirements.txt."
        ) from exc
    problems = _bundle_problems(bundle)
    if problems:
        raise BundleError("The model bundle is incompatible: " + "; ".join(problems) + ".")
    return bundle


def runtime_version_warnings(bundle):
    """Messages for packages whose major.minor version differs from the training environment."""
    import lightgbm
    import sklearn
    runtime = {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
               "pandas": pd.__version__, "numpy": np.__version__}
    trained = bundle.get("versions") or {}
    messages = []
    for package, installed in runtime.items():
        original = trained.get(package)
        if original and original.split(".")[:2] != installed.split(".")[:2]:
            messages.append(f"{package} {installed} is installed, but the model was trained "
                            f"with {original}.")
    return messages


def threshold_grid(*must_include):
    """Log-spaced + linear threshold grid that contains the given values EXACTLY."""
    grid = np.round(np.concatenate([np.logspace(-4, -1, 31), np.linspace(0.1, 0.99, 90)]), 6)
    keep = np.ones(len(grid), dtype=bool)
    for value in must_include:
        keep &= ~np.isclose(grid, value, rtol=0.0, atol=1e-6)
    return np.unique(np.concatenate([grid[keep], np.asarray(must_include, dtype=float)]))


# --------------------------------------------------------------------------- #
# Single transaction
# --------------------------------------------------------------------------- #
def validate_single_transaction(tx_type, step, amount, old_org, new_org, old_dest, new_dest):
    """Return ``(errors, notices)``.

    Errors make scoring impossible (negative / non-finite values, unknown type, step out of
    range).  Notices are informational only: unusual balance patterns are common in PaySim
    (the model even uses them as signals), so they never block scoring.
    """
    errors, notices = [], []
    if tx_type not in KNOWN_TYPES:
        errors.append(f"Transaction type must be one of {list(KNOWN_TYPES)}.")
    low, high = STEP_RANGE
    if not isinstance(step, (int, np.integer)) or isinstance(step, bool) or not low <= step <= high:
        errors.append(f"Time step must be a whole number between {low} and {high}.")
    values = {"Amount": amount, "Sender old balance": old_org, "Sender new balance": new_org,
              "Recipient old balance": old_dest, "Recipient new balance": new_dest}
    numeric = {}
    for label, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue
        if not math.isfinite(number):
            errors.append(f"{label} must be a finite number.")
        elif number < 0:
            errors.append(f"{label} cannot be negative.")
        else:
            numeric[label] = number
    if errors or tx_type not in ELIGIBLE_TYPES:
        return errors, notices

    amount, old_org, new_org = numeric["Amount"], numeric["Sender old balance"], numeric["Sender new balance"]
    old_dest, new_dest = numeric["Recipient old balance"], numeric["Recipient new balance"]
    if amount == 0:
        notices.append("The amount is 0, which is unusual.")
    if amount > old_org:
        notices.append("The amount is larger than the sender's old balance.")
    if abs(old_org - amount - new_org) > 0.01:
        notices.append("Sender balances do not satisfy old - amount = new. This is common in "
                       "PaySim and is used by the model as a signal, so the transaction is still scored.")
    if abs(old_dest + amount - new_dest) > 0.01:
        notices.append("Recipient balances do not satisfy old + amount = new. This is common in "
                       "PaySim and is used by the model as a signal, so the transaction is still scored.")
    return errors, notices


# --------------------------------------------------------------------------- #
# Batch CSV
# --------------------------------------------------------------------------- #
def read_csv_safely(source, max_bytes=MAX_FILE_BYTES, max_rows=MAX_ROWS):
    """Read an uploaded file or a path into a DataFrame, enforcing size/row limits."""
    size = getattr(source, "size", None)
    if size is not None and size > max_bytes:
        raise InputValidationError(
            f"The file is {size / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB. "
            "Please upload a smaller sample.")
    if hasattr(source, "seek"):
        source.seek(0)
    try:
        # index_col=False stops pandas from silently using extra leading fields as an index;
        # a row whose field count differs from the header is reported instead of mis-aligned.
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            df = pd.read_csv(source, nrows=max_rows + 1, index_col=False)
    except pd.errors.EmptyDataError as exc:
        raise InputValidationError("The file is empty.") from exc
    except UnicodeDecodeError as exc:
        raise InputValidationError("The file is not UTF-8 text. Please export it as a CSV.") from exc
    except (pd.errors.ParserError, pd.errors.ParserWarning) as exc:
        raise InputValidationError("The file could not be parsed as CSV: some rows have a different "
                                   "number of fields than the header (malformed rows).") from exc
    except Exception as exc:   # noqa: BLE001
        raise InputValidationError(f"The file could not be read as CSV ({type(exc).__name__}).") from exc
    if len(df) > max_rows:
        raise InputValidationError(f"The CSV has more than {max_rows:,} rows. Please upload a smaller sample.")
    if len(df) == 0:
        raise InputValidationError("The CSV has a header but no data rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _append_note(notes, mask, message):
    if mask.any():
        existing = notes.loc[mask]
        notes.loc[mask] = np.where(existing == "", message, existing + "; " + message)


def prepare_batch(df):
    """Validate a raw batch and assign a ``Status`` to every row.

    Returns ``(prepared, info)``.  Rows are never silently dropped:
      * ``Eligible``            TRANSFER / CASH_OUT with valid numbers -> will be scored
      * ``Not Fraud-Eligible``  CASH_IN / DEBIT / PAYMENT -> outside the model's trained scope
      * ``Unsupported type``    type not recognised
      * ``Invalid input``       missing / non-numeric / negative / infinite values
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise InputValidationError(
            f"Missing required column(s): {missing}. Required columns: {list(REQUIRED_COLUMNS)}.")

    out = df.copy().reset_index(drop=True)
    out.insert(0, "Input_Row", np.arange(2, len(out) + 2))     # line number in the CSV file
    raw_type = out["type"]
    type_clean = raw_type.map(lambda v: str(v).strip().upper() if pd.notna(v) else None)
    out["Type_Clean"] = type_clean

    status = pd.Series(STATUS_ELIGIBLE, index=out.index, dtype=object)
    notes = pd.Series("", index=out.index, dtype=object)
    no_type = type_clean.isna()
    unknown = ~no_type & ~type_clean.isin(KNOWN_TYPES)
    other_known = ~no_type & type_clean.isin(KNOWN_TYPES) & ~type_clean.isin(ELIGIBLE_TYPES)
    status[no_type] = STATUS_INVALID
    _append_note(notes, no_type, "transaction type is missing")
    status[unknown] = STATUS_UNSUPPORTED
    _append_note(notes, unknown, "transaction type is not recognised")
    status[other_known] = STATUS_NOT_ELIGIBLE
    _append_note(notes, other_known, "type is outside the fraud model's scope (TRANSFER / CASH_OUT)")

    eligible = status == STATUS_ELIGIBLE
    for col in NUMERIC_COLUMNS:
        raw = out[col]
        number = pd.to_numeric(raw, errors="coerce").astype("float64")
        _append_note(notes, eligible & raw.isna(), f"{col} is missing")
        _append_note(notes, eligible & raw.notna() & number.isna(), f"{col} is not a number")
        _append_note(notes, eligible & (number < 0), f"{col} is negative")
        _append_note(notes, eligible & np.isinf(number), f"{col} is not finite")
        out[col] = number
    status[eligible & (notes != "")] = STATUS_INVALID

    out["Status"] = status
    out["Validation_Note"] = notes

    warnings = []
    scored_steps = out.loc[status == STATUS_ELIGIBLE, "step"]
    n_odd_step = int(((scored_steps < STEP_RANGE[0]) | (scored_steps > STEP_RANGE[1])).sum())
    if n_odd_step:
        warnings.append(f"{n_odd_step} eligible row(s) have a step outside {STEP_RANGE[0]}-{STEP_RANGE[1]} "
                        "(the range seen in PaySim); hour/day features may be unreliable for them.")

    ground_truth = {"present": "isFraud" in out.columns, "valid": False, "message": ""}
    if ground_truth["present"]:
        label = pd.to_numeric(out["isFraud"], errors="coerce")
        n_bad = int((label.isna() | ~label.isin([0, 1])).sum())
        if n_bad:
            ground_truth["message"] = (f"Column 'isFraud' must contain only 0 or 1; {n_bad} row(s) are "
                                       "missing or invalid, so the cost evaluation is disabled.")
        else:
            ground_truth["valid"] = True
            out["isFraud"] = label.astype(int)

    info = {"n_rows": len(out), "status_counts": status.value_counts().to_dict(),
            "ground_truth": ground_truth, "warnings": warnings}
    return out, info


def score_batch(model, prepared, review_threshold, block_threshold):
    """Score ``Eligible`` rows only; every other row keeps NaN probability and 'Not scored'."""
    out = prepared.copy()
    mask = (out["Status"] == STATUS_ELIGIBLE).to_numpy()
    proba = np.full(len(out), np.nan)
    if mask.any():
        model_input = out.loc[mask, list(REQUIRED_COLUMNS)].copy()
        model_input["type"] = out.loc[mask, "Type_Clean"].astype(str)
        try:
            proba[mask] = np.asarray(model.predict_proba(model_input))[:, 1]
        except Exception as exc:   # noqa: BLE001
            raise InputValidationError(
                f"The model could not score this file ({type(exc).__name__}). "
                "Please check the column values.") from exc
    out["Fraud_Probability"] = proba
    out["Decision"] = decision_tier(proba, review_threshold, block_threshold)
    out["Expected_Loss_If_Approved"] = proba * out["amount"].to_numpy(dtype=float)
    return out


def evaluate_ground_truth(scored, fp_cost, review_threshold):
    """Cost evaluation on the scored rows (requires a valid ``isFraud`` column).

    Alerts = probability >= review_threshold (Manual Review and Block both count as alerts).
    Fraud in rows the model cannot score is reported separately, never hidden.
    """
    in_scope = scored["Status"] == STATUS_ELIGIBLE
    sub = scored.loc[in_scope]
    y = sub["isFraud"].to_numpy(dtype=int)
    amount = sub["amount"].to_numpy(dtype=float)
    flags = sub["Fraud_Probability"].to_numpy(dtype=float) >= review_threshold
    model_stats = cost_of(y, flags, amount, fp_cost)
    baseline = cost_of(y, np.zeros(len(sub), dtype=bool), amount, fp_cost)
    outside = scored.loc[~in_scope & (scored["isFraud"] == 1)]
    return {
        "model": model_stats,
        "no_model": baseline,
        "savings": baseline["total_cost"] - model_stats["total_cost"],
        "n_scored": int(len(sub)),
        "n_fraud_scored": int(y.sum()),
        "fraud_rate_scored": float(y.mean()) if len(sub) else 0.0,
        "out_of_scope_fraud_count": int(len(outside)),
        "out_of_scope_fraud_amount": float(np.nan_to_num(outside["amount"].to_numpy(dtype=float)).sum()),
    }
