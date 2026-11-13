"""Feature engineering, preprocessing, cost utilities and decision policy for the
PaySim mobile-wallet fraud project.

This module is the single source of truth shared by the notebook, the Streamlit
app and the tests.

IMPORTANT
---------
The fitted model in ``paysim_fraud_bundle.pkl`` stores ``PaySimFeatureEngineer``
by import path (``paysim_features.PaySimFeatureEngineer``).  Do not rename this
module or the class, and do not change a feature definition without retraining
the model (otherwise training and inference would silently disagree).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# --------------------------------------------------------------------------- #
# Scope and schema constants
# --------------------------------------------------------------------------- #
ELIGIBLE_TYPES = ("TRANSFER", "CASH_OUT")          # the only types the model is trained on
KNOWN_TYPES = ("TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT")
REQUIRED_COLUMNS = (
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
)
NUMERIC_COLUMNS = tuple(c for c in REQUIRED_COLUMNS if c != "type")
OPTIONAL_COLUMNS = ("nameOrig", "nameDest", "isFlaggedFraud", "isFraud")
DEFAULT_FP_COST = 50.0   # modelling ASSUMPTION, not a measured operational cost

# Decision labels (three-tier policy)
APPROVE = "Approve"
REVIEW = "Manual Review"
BLOCK = "Block"
NOT_SCORED = "Not scored"


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
class PaySimFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create the engineered features from raw PaySim columns.

    Parameters
    ----------
    include_post_tx : bool
        Include features that use ``newbalanceOrig`` / ``newbalanceDest``.
        These balances only exist AFTER the transaction has been executed, so a
        model that uses them cannot be used for pre-authorisation decisions.
    include_hour : bool
        Include ``hour_of_day`` and ``is_night_window`` (derived from ``step``).
        The hourly fraud pattern in PaySim is a simulator artefact.
    """

    def __init__(self, include_post_tx=True, include_hour=True):
        self.include_post_tx = include_post_tx
        self.include_hour = include_hour

    def fit(self, X, y=None):
        return self

    def _required_columns(self):
        cols = ["step", "amount", "oldbalanceOrg", "oldbalanceDest"]
        if self.include_post_tx:
            cols += ["newbalanceOrig", "newbalanceDest"]
        return cols

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PaySimFeatureEngineer expects a pandas DataFrame with named columns.")
        missing = [c for c in self._required_columns() if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")

        X_out = X.copy()

        # All arithmetic is done in float64 so that float32 inputs (as produced by
        # memory-saving read_csv dtypes) cannot add rounding noise to the balance
        # differences.  The original columns are left untouched.
        amount = X_out["amount"].astype("float64")
        old_org = X_out["oldbalanceOrg"].astype("float64")
        old_dest = X_out["oldbalanceDest"].astype("float64")
        step = X_out["step"]

        # 1. Features that are known BEFORE the transaction is executed
        X_out["log_amount"] = np.log1p(np.maximum(0.0, amount)).astype(np.float32)
        X_out["is_account_emptied"] = ((amount == old_org) & (old_org > 0)).astype(np.float32)
        X_out["amount_to_oldbalance_orig"] = (
            (amount / (old_org + 1.0)).clip(0.0, 100.0).astype(np.float32)
        )
        X_out["is_orig_zero_start"] = (old_org == 0).astype(np.float32)
        X_out["is_dest_zero_start"] = (old_dest == 0).astype(np.float32)
        X_out["day_of_month"] = ((step // 24) + 1).astype(np.float32)

        # 2. Hour features (ablation C).  Convention: hour_of_day = step % 24.
        if self.include_hour:
            hour = step % 24
            X_out["hour_of_day"] = hour.astype(np.float32)
            X_out["is_night_window"] = ((hour >= 2) & (hour <= 6)).astype(np.float32)

        # 3. POST-transaction features (ablation B): they use the new balances
        if self.include_post_tx:
            new_org = X_out["newbalanceOrig"].astype("float64")
            new_dest = X_out["newbalanceDest"].astype("float64")
            X_out["balance_change_orig"] = (new_org - old_org).astype(np.float32)
            X_out["orig_balance_diff"] = (old_org - amount - new_org).astype(np.float32)
            balance_change_dest = (new_dest - old_dest).astype(np.float32)
            X_out["balance_change_dest"] = balance_change_dest
            X_out["dest_balance_diff"] = (old_dest + amount - new_dest).astype(np.float32)
            # The ratio uses the float32-rounded balance change on purpose: this is
            # what the model in the bundle was trained/served with (bit-identical).
            X_out["dest_change_ratio"] = (
                (balance_change_dest.astype("float64") / (amount + 1.0))
                .clip(-10.0, 10.0).astype(np.float32)
            )
            X_out["is_dest_zero_both"] = (
                (old_dest == 0) & (new_dest == 0) & (amount > 0)
            ).astype(np.float32)

        cols_to_drop = [
            "step", "nameOrig", "nameDest", "isFlaggedFraud",
            "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        ]
        X_out.drop(columns=[c for c in cols_to_drop if c in X_out.columns], inplace=True)
        return X_out


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Winsorize heavy-tailed numeric columns at boxplot (Tukey) fences learned on TRAINING data.

    * Each tail gets its own fence: the upper fence comes from the positive values and the lower
      fence from the negative values. Fences are Q3 + w*IQR on the log1p(|x|) scale, so they are
      sensible for the very right-skewed money columns.
    * Zeros are never changed (zero is a legitimate value, e.g. an empty account); binary and
      constant columns are never changed either.
    * Rows are never removed: in fraud data the extreme values are often the fraud signal.
    """

    def __init__(self, whisker=1.5):
        self.whisker = whisker

    def _tail_cap(self, magnitudes):
        """Largest magnitude that is still inside the boxplot whisker (None if not applicable)."""
        if magnitudes.size < 20 or np.unique(magnitudes).size <= 2:      # binary / constant / too few
            return None
        log_values = np.log1p(magnitudes)
        q1, q3 = np.percentile(log_values, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return None
        return float(np.expm1(q3 + self.whisker * iqr))

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.full(X.shape[1], -np.inf)
        self.upper_ = np.full(X.shape[1], np.inf)
        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[np.isfinite(col)]
            upper = self._tail_cap(col[col > 0])
            lower = self._tail_cap(-col[col < 0])
            if upper is not None:
                self.upper_[j] = upper
            if lower is not None:
                self.lower_[j] = -lower
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build_preprocessing_pipeline(include_post_tx=True, include_hour=True, cap_outliers=False):
    """Feature engineering + imputation (+ optional outlier capping) + scaling + one-hot encoding.

    Everything is fitted on training data only. ``cap_outliers=False`` (default) reproduces the
    pipeline of the model stored in the bundle.
    """
    numeric_features = [
        "amount", "log_amount", "is_account_emptied", "amount_to_oldbalance_orig",
        "is_orig_zero_start", "is_dest_zero_start", "day_of_month",
    ]
    if include_hour:
        numeric_features += ["hour_of_day", "is_night_window"]
    if include_post_tx:
        numeric_features += [
            "balance_change_orig", "orig_balance_diff", "balance_change_dest",
            "dest_balance_diff", "dest_change_ratio", "is_dest_zero_both",
        ]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if cap_outliers:                       # optional winsorizing, fitted on training data only
        numeric_steps.append(("capper", OutlierCapper()))
    numeric_steps.append(("scaler", RobustScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, ["type"]),
    ])
    return Pipeline(steps=[
        ("feature_engineer", PaySimFeatureEngineer(include_post_tx=include_post_tx,
                                                   include_hour=include_hour)),
        ("preprocessor", preprocessor),
    ])


def model_feature_names(fitted_preprocessing):
    """Names of the columns the classifier receives (numeric first, then one-hot type)."""
    prep = fitted_preprocessing.named_steps["preprocessor"]
    numeric = list(prep.transformers_[0][2])
    onehot = prep.named_transformers_["cat"].named_steps["onehot"]
    return numeric + list(onehot.get_feature_names_out(["type"]))


# --------------------------------------------------------------------------- #
# Cost utilities
#   missed fraud cost  = full transaction amount (a PROXY for the real loss)
#   false alarm cost   = fp_cost per legitimate transaction that is flagged (ASSUMPTION)
# --------------------------------------------------------------------------- #
def cost_of(y_true, flag, amount, fp_cost):
    """Confusion counts and financial cost for one set of decisions."""
    y_true = np.asarray(y_true)
    flag = np.asarray(flag, dtype=bool)
    amount = np.asarray(amount, dtype=float)
    if not (len(y_true) == len(flag) == len(amount)):
        raise ValueError("y_true, flag and amount must have the same length.")
    tp = int((flag & (y_true == 1)).sum())
    fp = int((flag & (y_true == 0)).sum())
    fn = int((~flag & (y_true == 1)).sum())
    missed = float(amount[~flag & (y_true == 1)].sum())
    return dict(
        alerts=tp + fp,
        TP=tp,
        FP=fp,
        FN=fn,
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        missed_fraud_amount=missed,
        false_alarm_cost=fp * fp_cost,
        total_cost=missed + fp * fp_cost,
    )


def cost_curve(y_true, proba, amount, fp_cost, thresholds):
    """``cost_of`` evaluated for ``flag = proba >= t`` over a grid of thresholds (ascending)."""
    proba = np.asarray(proba, dtype=float)
    rows = [{"threshold": float(t), **cost_of(y_true, proba >= t, amount, fp_cost)}
            for t in thresholds]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def recost(curve, fp_cost):
    """Re-price an existing curve for another fp_cost (the counts do not depend on it)."""
    out = curve.copy()
    out["false_alarm_cost"] = out["FP"] * fp_cost
    out["total_cost"] = out["missed_fraud_amount"] + out["false_alarm_cost"]
    return out


def select_threshold(curve):
    """Lowest threshold that reaches the minimum total cost.

    Ties are broken toward the LOWER threshold (higher recall) because a missed
    fraud (its full amount) is far more expensive than one false alarm.
    """
    curve = curve.sort_values("threshold")
    best = curve["total_cost"].min()
    return float(curve.loc[curve["total_cost"] <= best, "threshold"].min())


def near_optimal_range(curve, fp_cost, tolerance_fp=1.0):
    """(low, high) thresholds whose cost is within ``tolerance_fp`` false alarms of the minimum."""
    best = curve["total_cost"].min()
    near = curve.loc[curve["total_cost"] <= best + tolerance_fp * fp_cost, "threshold"]
    return float(near.min()), float(near.max())


# --------------------------------------------------------------------------- #
# Three-tier decision policy
#   p <  review_threshold            -> Approve
#   review_threshold <= p < block    -> Manual Review   (review_threshold is the
#                                        validation cost-optimal threshold)
#   p >= block_threshold             -> Block           (policy parameter, NOT optimised)
# --------------------------------------------------------------------------- #
def decision_tier(proba, review_threshold, block_threshold):
    block_threshold = max(float(block_threshold), float(review_threshold))
    p = np.asarray(proba, dtype=float)
    tier = np.where(p >= block_threshold, BLOCK,
                    np.where(p >= review_threshold, REVIEW, APPROVE)).astype(object)
    tier = np.where(np.isnan(p), NOT_SCORED, tier)
    return str(tier) if p.ndim == 0 else tier


# --------------------------------------------------------------------------- #
# Probability calibration (sigmoid / Platt scaling on the logit of the raw score)
# --------------------------------------------------------------------------- #
_EPS = 1e-15


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def fit_logit_platt(raw_proba, y):
    """Fit p_cal = sigmoid(slope * logit(p_raw) + intercept); returns (slope, intercept)."""
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(_logit(raw_proba).reshape(-1, 1), np.asarray(y).astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_logit_platt(raw_proba, slope, intercept):
    z = slope * _logit(raw_proba) + intercept
    return 1.0 / (1.0 + np.exp(-z))


class CalibratedPipeline:
    """A fitted pipeline plus a monotonic sigmoid calibrator; exposes ``predict_proba``."""

    def __init__(self, base_pipeline, slope, intercept):
        self.base_pipeline = base_pipeline
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw = self.base_pipeline.predict_proba(X)[:, 1]
        p = apply_logit_platt(raw, self.slope, self.intercept)
        return np.column_stack([1.0 - p, p])
