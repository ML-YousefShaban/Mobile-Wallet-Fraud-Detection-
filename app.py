"""Mobile-wallet fraud screening on PaySim - Streamlit app.

Run with:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from paysim_features import (
    APPROVE, BLOCK, ELIGIBLE_TYPES, REQUIRED_COLUMNS, REVIEW,
    cost_curve, decision_tier, select_threshold,
)
from paysim_scoring import (
    MAX_FILE_BYTES, MAX_ROWS, STATUS_ELIGIBLE, STATUS_INVALID, STATUS_NOT_ELIGIBLE,
    STATUS_UNSUPPORTED, BundleError, InputValidationError, evaluate_ground_truth, load_bundle,
    prepare_batch, read_csv_safely, runtime_version_warnings, score_batch, threshold_grid,
    validate_single_transaction,
)

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "paysim_fraud_bundle.pkl"
DEMO_SAMPLE = "Demo sample (enriched - fraud deliberately over-represented)"
REALISTIC_SAMPLE = "Realistic sample (natural fraud rate)"
SAMPLE_FILES = {
    REALISTIC_SAMPLE: BASE_DIR / "sample_transactions_realistic.csv",
    DEMO_SAMPLE: BASE_DIR / "sample_transactions_demo.csv",
}
UPLOAD = "Upload my own CSV"
UI_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
# Policy default for the auto-block tier (documented in the notebook). It is NOT optimised
# from data; the bundle can override it with a 'block_threshold' entry.
DEFAULT_BLOCK_THRESHOLD = 0.50
UNIT = "currency units"

st.set_page_config(page_title="Mobile-Wallet Fraud Detection (PaySim)", page_icon="🛡️", layout="wide")


def bundle_signature():
    """Path + modification time: the caches below reload automatically when the model file is
    replaced (for example after re-running the notebook) instead of serving a stale model."""
    exists = BUNDLE_PATH.exists()
    return str(BUNDLE_PATH), (BUNDLE_PATH.stat().st_mtime_ns if exists else None)


@st.cache_resource(show_spinner="Loading model...")
def get_bundle(path, modified_ns):
    return load_bundle(path)


@st.cache_data(show_spinner="Scoring transactions...", max_entries=3)
def prepare_and_score(raw: pd.DataFrame, path, modified_ns):
    """Validation + model scoring (cached; thresholds are applied afterwards, so moving a
    slider never re-runs the model)."""
    prepared, info = prepare_batch(raw)
    model = get_bundle(path, modified_ns)["pipeline"]
    return score_batch(model, prepared, review_threshold=0.5, block_threshold=0.5), info


def fmt_probability(p):
    return f"{p * 100:.3f}%" if p >= 1e-4 else f"{p:.2e}"


def fmt_money(x):
    return f"{x:,.2f}"


# --------------------------------------------------------------------------- #
# Load the model bundle (clear error, never fake fallback values)
# --------------------------------------------------------------------------- #
try:
    bundle = get_bundle(*bundle_signature())
except BundleError as exc:
    st.error(f"❌ {exc}")
    st.stop()

model = bundle["pipeline"]
metrics = bundle["metrics"]
default_review = float(bundle["threshold"])
default_block = max(float(bundle.get("block_threshold", DEFAULT_BLOCK_THRESHOLD)), default_review)
default_fp_cost = float(bundle["fp_cost"])
natural_rate = metrics.get("natural_fraud_rate")

st.title("🛡️ Mobile-Wallet Fraud Detection & Cost Optimization")
st.markdown(
    "Screening of **completed** mobile-money transactions (**TRANSFER** and **CASH_OUT**) with a "
    "LightGBM model trained on the synthetic **PaySim** dataset, plus a cost-based decision threshold."
)
for message in runtime_version_warnings(bundle):
    st.warning(f"⚠️ {message} Install the versions pinned in requirements.txt.")

with st.expander("About this model and its limitations"):
    st.markdown(
        "- **Synthetic data.** PaySim is a simulation, not real bank data; high scores on it do not "
        "guarantee equivalent performance in production.\n"
        "- **Scope.** Only TRANSFER and CASH_OUT are scored (the only types that contain fraud in PaySim). "
        "Other types are marked *Not Fraud-Eligible*: they are outside the model's trained scope, which "
        "does **not** mean they are safe.\n"
        "- **Post-transaction information.** The model uses the sender's and recipient's balances *after* the "
        "transaction. It therefore screens executed transactions and is **not** ready for pre-authorisation "
        "(real-time blocking) use.\n"
        "- **Random split.** The model was validated on a random split, which does not show that it "
        "generalises to future periods.\n"
        f"- **Costs are assumptions.** A missed fraud is priced at its full amount (a proxy for the real loss) "
        f"and a false alarm at a fixed, configurable cost ({UNIT}); neither is a measured operational value.\n"
        "- **Probabilities.** Expected-loss figures multiply the model score by the amount; they are only "
        "meaningful to the extent that the score is a calibrated probability (see the notebook's calibration section)."
    )

# --------------------------------------------------------------------------- #
# Sidebar: decision policy, cost assumption, stored evaluation metrics
# --------------------------------------------------------------------------- #
st.sidebar.header("⚙️ Decision policy & cost assumption")
grid = [float(v) for v in threshold_grid(default_review, default_block)]
review_threshold = st.sidebar.select_slider(
    "Manual-review threshold", options=grid, value=default_review, format_func=lambda x: f"{x:.4f}",
    help="Probability at or above which a transaction is flagged. Default = the threshold that "
         "minimised total cost on the validation set (log-spaced slider).")
block_threshold = st.sidebar.select_slider(
    "Auto-block threshold (policy)", options=grid, value=default_block, format_func=lambda x: f"{x:.4f}",
    help="Probability at or above which a flagged transaction is blocked automatically instead of "
         "reviewed. This is a business-policy parameter, not an optimised value.")
effective_block = max(block_threshold, review_threshold)
if block_threshold < review_threshold:
    st.sidebar.caption("The block threshold is below the review threshold, so the review band is empty.")
fp_cost = st.sidebar.number_input(
    f"False-alarm cost ({UNIT}) - assumption", min_value=1.0, value=default_fp_cost, step=25.0, format="%.2f",
    help="Assumed cost of wrongly flagging one legitimate transaction. This is a modelling assumption, "
         "not a measured operational cost.")
st.sidebar.caption(
    f"Policy: score < {review_threshold:.4f} → **{APPROVE}**; "
    f"{review_threshold:.4f} ≤ score < {effective_block:.4f} → **{REVIEW}**; "
    f"score ≥ {effective_block:.4f} → **{BLOCK}**.")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Stored evaluation (from the model file)")
stored_metrics = [
    ("val_pr_auc", "Validation PR-AUC", "Natural fraud rate; this split chose the threshold."),
    ("test_pr_auc", "Test PR-AUC", "Held-out random test split, evaluated once."),
    ("test_roc_auc", "Test ROC-AUC", "Held-out random test split, evaluated once."),
]
shown = False
for key, label, hint in stored_metrics:
    if key in metrics:
        st.sidebar.metric(label, f"{metrics[key]:.4f}", help=hint)
        shown = True
if not shown:
    st.sidebar.info("No evaluation metrics are stored in this model file.")
if "cv_pr_auc" in metrics:
    st.sidebar.caption(f"CV PR-AUC on the undersampled tuning folds: {metrics['cv_pr_auc']:.4f} "
                       "(about 3% fraud - not comparable with the PR-AUC values above).")
st.sidebar.caption(f"Validation-optimal threshold at an assumed false-alarm cost of {default_fp_cost:,.0f}: "
                   f"{default_review:.4f}. Random split, synthetic data - see 'About this model'.")

tab_single, tab_batch = st.tabs(["📝 Single transaction", "📁 Batch CSV & cost analysis"])


# --------------------------------------------------------------------------- #
# Tab 1: single transaction
# --------------------------------------------------------------------------- #
def render_single_tab():
    st.subheader("Single transaction")
    st.caption("The balances **after** the transaction are model inputs, so this scores a transaction "
               "that has already been executed.")
    col1, col2, col3 = st.columns(3)
    with col1:
        tx_type = st.selectbox("Transaction type", UI_TYPES)
        amount = st.number_input(f"Amount ({UNIT})", min_value=0.0, value=250_000.0, step=1000.0, format="%.2f")
        step = st.slider("Time step (hour 1-744)", min_value=1, max_value=744, value=5)
        st.caption(f"Hour of day = step mod 24 = {step % 24}; day = {step // 24 + 1}.")
    with col2:
        old_org = st.number_input("Sender balance before", min_value=0.0, value=250_000.0, step=1000.0, format="%.2f")
        new_org = st.number_input("Sender balance after", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
    with col3:
        old_dest = st.number_input("Recipient balance before", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
        new_dest = st.number_input("Recipient balance after", min_value=0.0, value=0.0, step=1000.0, format="%.2f")

    if not st.button("🔍 Score transaction", type="primary"):
        return
    st.markdown("---")
    errors, notices = validate_single_transaction(tx_type, step, amount, old_org, new_org, old_dest, new_dest)
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
        return
    if tx_type not in ELIGIBLE_TYPES:
        st.info(
            f"ℹ️ **Not Fraud-Eligible** - `{tx_type}` is outside the model's trained scope "
            "(TRANSFER and CASH_OUT). No fraud probability was computed. This is not a statement that the "
            "transaction is safe; apply your normal controls.")
        return

    transaction = pd.DataFrame([{
        "step": step, "type": tx_type, "amount": amount, "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org, "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
    }])
    probability = float(model.predict_proba(transaction)[0, 1])
    tier = decision_tier(probability, review_threshold, block_threshold)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fraud probability (p)", fmt_probability(probability))
    m2.metric("Review / block thresholds", f"{review_threshold:.4f} / {effective_block:.4f}")
    m3.metric("Expected loss if approved (p × amount)", fmt_money(probability * amount))
    m4.metric("Expected cost if blocked ((1-p) × false-alarm cost)", fmt_money((1.0 - probability) * fp_cost))

    if tier == BLOCK:
        st.error(f"🚫 **Decision: {BLOCK}** - the score ({probability:.4f}) is at or above the block threshold "
                 f"({effective_block:.4f}).")
    elif tier == REVIEW:
        st.warning(f"🕵️ **Decision: {REVIEW}** - the score ({probability:.4f}) is at or above the review "
                   f"threshold ({review_threshold:.4f}) but below the block threshold ({effective_block:.4f}).")
    else:
        st.success(f"✅ **Decision: {APPROVE}** - the score ({probability:.4f}) is below the review "
                   f"threshold ({review_threshold:.4f}).")
    for notice in notices:
        st.caption(f"ℹ️ {notice}")
    st.caption("The decision comes from the validated thresholds. The two expected-cost figures are "
               "informational and rely on the score behaving like a calibrated probability.")


# --------------------------------------------------------------------------- #
# Tab 2: batch CSV
# --------------------------------------------------------------------------- #
def plot_cost_curve(y_true, proba, amount):
    """Cost of the decisions as a share of the no-model cost, over a threshold grid (this file only)."""
    curve = cost_curve(y_true, proba, amount, fp_cost, threshold_grid(review_threshold))
    no_model_cost = float(amount[y_true == 1].sum())
    if no_model_cost <= 0:
        return None
    batch_optimal = select_threshold(curve)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(curve["threshold"], curve["total_cost"] / no_model_cost * 100, color="teal", linewidth=2,
            label="Total cost (% of no-model cost)")
    ax.axhline(100, color="gray", linewidth=1, label="No model = 100%")
    ax.axvline(review_threshold, color="crimson", linestyle="--", linewidth=1.8,
               label=f"Selected review threshold ({review_threshold:.4f})")
    ax.axvline(batch_optimal, color="darkorange", linestyle=":", linewidth=1.8,
               label=f"Optimal for THIS file only ({batch_optimal:.4f})")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Decision threshold (log scale)")
    ax.set_ylabel("% of no-model cost")
    ax.set_title("Cost vs. threshold on the scored rows of this file")
    ax.legend(fontsize=8)
    return fig


def load_batch_source():
    """Return (raw DataFrame, source label) or (None, None) after showing a clear message."""
    available = {label: path for label, path in SAMPLE_FILES.items() if path.exists()}
    options = [UPLOAD] + list(available)
    source = st.radio("Data source", options, horizontal=True, index=1 if available else 0)
    missing = [path.name for path in SAMPLE_FILES.values() if not path.exists()]
    if missing:
        st.caption(f"Sample file(s) not found: {', '.join(missing)}. They are created by the last cells of "
                   "`next_gen.ipynb`.")
    try:
        if source == UPLOAD:
            st.download_button("⬇️ CSV template (column names only)",
                               ",".join(REQUIRED_COLUMNS) + ",isFraud\n", "paysim_template.csv", "text/csv")
            uploaded = st.file_uploader(
                "Upload a PaySim-format CSV", type=["csv"],
                help=f"Limits: {MAX_ROWS:,} rows and {MAX_FILE_BYTES / 1e6:.0f} MB. "
                     f"Required columns: {', '.join(REQUIRED_COLUMNS)}. Optional: isFraud (0/1).")
            if uploaded is None:
                st.info(f"Upload a CSV with the columns `{', '.join(REQUIRED_COLUMNS)}` (optionally `isFraud`).")
                return None, None
            return read_csv_safely(uploaded), source
        return read_csv_safely(available[source]), source
    except InputValidationError as exc:
        st.error(f"❌ {exc}")
        return None, None


def render_batch_tab():
    st.subheader("Batch scoring & cost analysis")
    raw, source = load_batch_source()
    if raw is None:
        return
    try:
        scored, info = prepare_and_score(raw, *bundle_signature())
    except InputValidationError as exc:
        st.error(f"❌ {exc}")
        return

    scored = scored.copy()
    scored["Decision"] = decision_tier(scored["Fraud_Probability"].to_numpy(), review_threshold, block_threshold)
    is_scored = scored["Status"] == STATUS_ELIGIBLE
    n_alerts = int((scored["Decision"].isin([REVIEW, BLOCK])).sum())
    counts = info["status_counts"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows processed", f"{info['n_rows']:,}")
    c2.metric("Scored (eligible)", f"{int(is_scored.sum()):,}")
    c3.metric("Alerts (review + block)", f"{n_alerts:,}")
    c4.metric("Auto-block", f"{int((scored['Decision'] == BLOCK).sum()):,}")
    c5.metric("Not scored", f"{info['n_rows'] - int(is_scored.sum()):,}")

    if counts.get(STATUS_INVALID):
        st.warning(f"⚠️ {counts[STATUS_INVALID]:,} row(s) contain invalid values and were not scored "
                   "(see `Validation_Note`).")
    if counts.get(STATUS_NOT_ELIGIBLE):
        st.info(f"ℹ️ {counts[STATUS_NOT_ELIGIBLE]:,} row(s) are **Not Fraud-Eligible** (outside TRANSFER / CASH_OUT) "
                "and were not scored. This does not mean they are safe.")
    if counts.get(STATUS_UNSUPPORTED):
        st.warning(f"⚠️ {counts[STATUS_UNSUPPORTED]:,} row(s) have an unrecognised transaction type and were not scored.")
    for warning in info["warnings"]:
        st.warning(f"⚠️ {warning}")

    truth = info["ground_truth"]
    if truth["present"] and not truth["valid"]:
        st.error(f"❌ {truth['message']}")
    if truth["valid"]:
        render_ground_truth(scored, source)

    st.markdown("### 🔝 Transactions ranked by fraud probability")
    columns = ["Input_Row", "type", "amount", "Status", "Fraud_Probability", "Decision", "Expected_Loss_If_Approved"]
    if truth["valid"]:
        columns.append("isFraud")
    ranked = scored.sort_values("Fraud_Probability", ascending=False, na_position="last")
    st.dataframe(
        ranked[columns].head(50), width="stretch", hide_index=True,
        column_config={
            "Fraud_Probability": st.column_config.NumberColumn(format="%.6f"),
            "Expected_Loss_If_Approved": st.column_config.NumberColumn(format="%.2f"),
        })
    st.caption("Top 50 rows shown; rows that were not scored appear last. The download contains every row.")
    st.download_button("📥 Download scored CSV", ranked.to_csv(index=False).encode("utf-8"),
                       "paysim_scored_transactions.csv", "text/csv")


def render_ground_truth(scored, source):
    st.markdown("### 💰 Cost evaluation against `isFraud`")
    result = evaluate_ground_truth(scored, fp_cost, review_threshold)
    model_stats, base = result["model"], result["no_model"]
    share = result["fraud_rate_scored"]
    if source == DEMO_SAMPLE:
        st.warning(f"⚠️ This is the **enriched demonstration sample**: fraud is {share:.1%} of the scored rows. "
                   "Precision, costs and the batch-optimal threshold below are **not representative** of the "
                   "natural fraud rate and must not be quoted as model performance.")
    elif natural_rate and result["n_scored"] and (share > 5 * natural_rate or share < natural_rate / 5):
        st.info(f"ℹ️ Fraud is {share:.2%} of the scored rows here versus about {natural_rate:.2%} in PaySim's "
                "TRANSFER / CASH_OUT data, so these figures are not representative of the natural distribution.")
    if result["out_of_scope_fraud_count"]:
        st.warning(f"⚠️ {result['out_of_scope_fraud_count']:,} fraud row(s) "
                   f"(total amount {fmt_money(result['out_of_scope_fraud_amount'])}) are of a type or validity "
                   "the model does not score; they are excluded from the figures below.")
    if not result["n_scored"]:
        st.info("No scored rows to evaluate.")
        return

    a, b, c, d = st.columns(4)
    a.metric("TP / FN", f"{model_stats['TP']:,} / {model_stats['FN']:,}")
    b.metric("False alarms (FP)", f"{model_stats['FP']:,}")
    c.metric("Precision / recall", f"{model_stats['precision']:.3f} / {model_stats['recall']:.3f}")
    d.metric("Alerts", f"{model_stats['alerts']:,}")
    e, f, g = st.columns(3)
    e.metric(f"Missed fraud amount ({UNIT})", fmt_money(model_stats["missed_fraud_amount"]))
    f.metric(f"False-alarm cost ({UNIT})", fmt_money(model_stats["false_alarm_cost"]))
    saving = result["savings"]
    g.metric(f"Total cost ({UNIT})", fmt_money(model_stats["total_cost"]),
             delta=f"{fmt_money(abs(saving))} {'lower' if saving >= 0 else 'higher'} than no model "
                   f"({fmt_money(base['total_cost'])})",
             delta_color="normal" if saving >= 0 else "inverse")
    st.caption("Alerts are all rows at or above the review threshold (review + block). A missed fraud costs its "
               "full amount (a proxy for the real loss); a false alarm costs the assumed false-alarm cost.")

    in_scope = scored["Status"] == STATUS_ELIGIBLE
    sub = scored.loc[in_scope]
    fig = plot_cost_curve(sub["isFraud"].to_numpy(dtype=int), sub["Fraud_Probability"].to_numpy(dtype=float),
                          sub["amount"].to_numpy(dtype=float))
    if fig is None:
        st.info("The cost curve needs at least one fraud with a positive amount among the scored rows.")
    else:
        st.pyplot(fig)
        plt.close(fig)
        st.caption("The orange line is optimal for this file only; it is not the validated operating threshold.")


with tab_single:
    render_single_tab()
with tab_batch:
    render_batch_tab()
