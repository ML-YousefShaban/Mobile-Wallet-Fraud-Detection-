# Mobile-Wallet Fraud Detection & Cost Optimization (PaySim)

A LightGBM fraud-screening model for mobile-money **TRANSFER** and **CASH_OUT** transactions, a cost-based decision
policy (Approve / Manual Review / Block) and a Streamlit app for single and batch scoring.

> ## Status of this repository - read first
> | Item | Status |
> |---|---|
> | Code (`paysim_features.py`, `paysim_scoring.py`, `app.py`, notebook code) | Audited and corrected. **103 automated tests pass** (feature engineering, cost maths, model/bundle round-trip, CSV validation, headless Streamlit runs). |
> | `next_gen.ipynb` | Executed end-to-end on **synthetic data only** (to prove that every cell runs). It has **not** been executed on the real PaySim file and is shipped **without outputs**. |
> | `paysim_fraud_bundle.pkl` | The **original model from the pre-audit run**, byte-for-byte unchanged. It loads in a fresh Python process and reproduces its predictions exactly with the corrected code. |
> | Results quoted below | Taken from the **original run** (notebook outputs / bundle). They were not re-computed during the audit. Re-running the audited notebook regenerates them and they may move slightly (see [section 9](#9-model-training-and-hyper-parameter-tuning)). |
> | `sample_transactions_*.csv` | Not included: they are drawn from the held-out test split by the last cell of the notebook (needs the dataset). The app works without them. |

## 1. Project overview
* Data: PaySim synthetic mobile-money transactions (Kaggle `ealaxi/paysim1`).
* Model: LightGBM inside a scikit-learn `Pipeline` (feature engineering, imputation, scaling, one-hot encoding).
* Decision: a validation-optimised **review threshold** (minimum total cost) and a policy-based **block threshold**.
* App: Streamlit (`app.py`) for one transaction or a CSV, with cost evaluation when `isFraud` is present.

## 2. Problem statement
Decide, for each TRANSFER / CASH_OUT, whether to approve it, send it for manual review, or block it, so that the total cost
(missed fraud + false alarms) is minimal. Fraud is extremely rare, so the primary ranking metric is **PR-AUC (average
precision)** and the operating point is chosen by **expected cost**, not accuracy.

## 3. Dataset description
PaySim simulates mobile-money logs. Columns: `step` (1 step = 1 hour; 744 steps is about 31 days), `type`, `amount`,
`nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.
Facts printed by the original run: 6,362,620 rows, no missing values, 8,213 frauds (0.1291 %).
The CSV (about 470 MB) is **not** included; see `data/README.md`.

## 4. Project scope
Fraud occurs **only** in `TRANSFER` (4,097 of 532,909 rows, 0.7688 %) and `CASH_OUT` (4,116 of 2,237,500 rows, 0.1840 %);
`CASH_IN`, `DEBIT` and `PAYMENT` (3,592,211 rows) contain none. The scope therefore has 2,770,409 rows with a fraud rate of
0.2965 %. The notebook **verifies this in code** and stops with an error if fraud ever appears in another type.
Other types are labelled **Not Fraud-Eligible** in the app: outside the trained scope, which does **not** mean safe.

## 5. Exploratory data analysis
Key facts from the original run (full data; the audited notebook recomputes feature-motivating statistics on the training split only):
* Fraud rows almost always drain the sender account: `amount == oldbalanceOrg` in 97.63 % of fraud rows vs about 0 % of legitimate rows.
* The recipient balance stays at zero before and after in 49.56 % of fraud rows vs 0.06 % of legitimate rows.
* Median amount: 441,423 (fraud) vs 171,034 (legitimate).
* The fraud *rate* peaks at night (hours 2-6), but the *count* of frauds per hour is roughly flat: this is low legitimate volume at night, a
  simulator artefact, not customer behaviour.
* The legacy rule `isFlaggedFraud` flags only 16 of the 8,213 frauds.

The notebook also contains class/type charts, a Spearman correlation heatmap, and a boxplot-based **outlier analysis** (section 4b):
Tukey fences on a log scale for non-zero values, the fraud rate inside vs outside the outliers, and winsorizing (capping) fitted on the
training split. Extreme values are **not deleted** (they are largely the fraud signal); only invalid rows would be dropped, from the fit split only.

## 6. Feature engineering
All logic is in `paysim_features.py` and is used by the notebook, the saved model and the app.

| Group | Features | Needs post-transaction data? |
|---|---|---|
| Base | `amount`, `log_amount`, `type` (one-hot) | no |
| Pre-transaction balances | `is_account_emptied`, `amount_to_oldbalance_orig`, `is_orig_zero_start`, `is_dest_zero_start` | no |
| Time (from `step`) | `day_of_month`, `hour_of_day`, `is_night_window` (hour = `step % 24`) | no (hour pattern is an artefact) |
| **Post-transaction** | `balance_change_orig`, `orig_balance_diff`, `balance_change_dest`, `dest_balance_diff`, `dest_change_ratio`, `is_dest_zero_both` | **yes** (new balances) |

Arithmetic is done in float64. Account ids are never features. **The full model uses post-transaction balances, so it is a post-hoc
screening model and is not ready for pre-authorisation (real-time) blocking.**

## 7. Machine-learning approach
Baseline, Logistic Regression, Decision Tree, Random Forest and LightGBM are compared inside one pipeline. LightGBM is the project
model (direction fixed at the start of the project). Imputation/scaling/encoding are fitted on training folds only.

## 8. Class-imbalance strategy
Stratified random split **60 / 20 / 20** (fit / validation / test) **before** anything else. Only for model comparison and tuning, a
tuning set is built from the **fit split**: all 4,927 frauds plus 150,000 random legitimate rows (154,927 rows, 3.18 % fraud). No
`class_weight`, no SMOTE. The **final model is refit on the whole natural-distribution fit split** (1,662,245 rows; this was verified from
the saved model: its first tree was trained on exactly that many rows). Validation and test metrics always use the natural distribution.

## 9. Model training and hyper-parameter tuning
`RandomizedSearchCV` (12 candidates, 5-fold stratified CV, average precision) on the **undersampled tuning set only**. Original-run best
parameters: `n_estimators=350, learning_rate=0.1, num_leaves=63, max_depth=-1, min_child_samples=50, subsample=0.9, colsample_bytree=0.8,
reg_alpha=0, reg_lambda=0`.

Differences between the original run and the audited notebook that will change the regenerated numbers slightly:
* `subsample` had **no effect** in the original run (LightGBM ignores it when `subsample_freq=0`); the audited notebook sets `subsample_freq=1`.
* Data is loaded as float64 (the original used float32, which rounds large amounts and balances).
* The threshold grid now starts at 1e-4 (it started at 1e-3), and tie-breaking is explicit.
* Optional outlier capping (`CAP_OUTLIERS`, off by default) and a calibration check were added.

## 10. Evaluation metrics
PR-AUC (average precision), ROC-AUC, precision, recall, F1, the confusion matrix, and total cost. Three kinds of numbers are kept apart:

| Where | Distribution | Original-run values |
|---|---|---|
| **CV** on the tuning set (best config) | undersampled, 3.18 % fraud | PR-AUC 0.9981 |
| **Validation** split | natural, 0.30 % fraud | PR-AUC 0.9977 (final model) |
| **Test** split (evaluated once) | natural | PR-AUC 0.9975, ROC-AUC 0.9993 |

**Do not compare the CV number with the validation/test numbers** (different class balance). Model comparison on the validation split
(original run, PR-AUC): Baseline 0.0030, Logistic Regression 0.9975, Decision Tree 0.9879, Random Forest 0.9977, LightGBM 0.9978. These
differences are within sampling noise. One anomaly is unresolved: **LightGBM (default) showed a CV PR-AUC of 0.8594 +/- 0.2433** while the
same model scores 0.9978 on validation, which implies that at least one CV fold failed (the numbers are consistent with one fold near
PR-AUC 0.3). The audited notebook prints per-fold scores so this can be inspected after a re-run.

## 11. Threshold optimization
Cost = **missed fraud amount + FP_COST x false alarms**. The review threshold is the **lowest** threshold with the minimum cost on the
**validation** split (ties go toward higher recall). The test split plays no role. Original run: threshold **0.24** at `FP_COST = 50`.
The validation cost curve is flat over a wide range (about 0.001 to 0.55), so 0.24 sits on a plateau and depends on a handful of
transactions; the notebook now shades the near-optimal band. Sensitivity (validation): the selected threshold was 0.24 for FP_COST 50, 500
and 5,000 (alerts 1,640, precision 0.99939, recall 0.997565 each); the audited notebook adds 50,000 and 500,000 to show where it starts to move.

**Three-tier policy** (same in notebook and app): score < review threshold -> **Approve**; review <= score < block -> **Manual Review**;
score >= block -> **Block**. The block threshold (default 0.50) is a **policy parameter that was not optimised**; it is configurable.

## 12. Financial cost analysis
**Assumptions:** `FP_COST = 50` is a modelling assumption, not a measured operational cost; counting the **full transaction amount** as the
loss of a missed fraud is only a proxy. Original-run test results at threshold 0.24 (1,643 frauds in the test split):

| Strategy | Alerts | TP | FP | FN | Total cost |
|---|---|---|---|---|---|
| No model | 0 | 0 | 0 | 1,643 | about 2.427 billion |
| Legacy rule (`isFlaggedFraud`) | 3 | 3 | 0 | 1,640 | about 2.416 billion |
| **Model @ 0.24** | 1,639 | 1,637 | 2 | 6 | **about 1.799 million** (missed fraud about 1.799 million + 2 x 50) |
| Flag everything | 554,082 | 1,643 | 552,439 | 0 | 27,621,950 (552,439 x 50) |

(Amounts were computed from float32-rounded values in the original run; the audited notebook uses float64.)
Caution: the cost model has **no alert-capacity limit**. When a score is not near-separable and `FP_COST` is small relative to fraud amounts,
the cost optimum can be to flag most transactions. PaySim's pattern is so clean that this does not happen here.

## 13. Error analysis
Original run, test split: 6 missed frauds (median amount about 259k, none of them draining the account, probabilities between 1e-10 and 9e-6, i.e.
indistinguishable for this model) and 2 false alarms (both CASH_OUT, probabilities 0.33 and 0.50). The notebook lists the most expensive missed
frauds and the highest-scoring false alarms, and breaks errors down by type, hour and account-draining behaviour. These tables are interpretation
only; nothing is tuned from them.

## 14. Feature importance
LightGBM gain: `is_account_emptied` carries almost all of the gain; `dest_balance_diff` is a distant second. The model therefore behaves almost
like a one-feature rule plus corrections. The audited notebook shows the gain *share* (%) for the top 15 features.

## 15. Ablation study
Same hyper-parameters, same data, **validation set only**. Original-run PR-AUC / ROC-AUC:

| Variant | PR-AUC | ROC-AUC |
|---|---|---|
| A: full feature set | 0.9977 | 0.9992 |
| B: without post-transaction features | 0.6216 | 0.9219 |
| C: without hour features | 0.9980 | 0.9997 |

Reading: the hour features add nothing measurable (the model does not depend on that artefact), and the post-transaction balances matter a lot,
so the full model must not be presented as a pre-authorisation model. **Caveat:** variant B still contains `is_account_emptied`, a pre-transaction
feature that separates 97.6 % of frauds from about 0 % of legitimate rows (EDA above). A single rule on that feature should reach a PR-AUC far above
0.62, so B's value is **suspicious and unverified**; the audited notebook adds that rule as a baseline row (and a variant D for outlier capping).

## 16. Streamlit application
`streamlit run app.py`
* **Single transaction:** TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT. Only TRANSFER/CASH_OUT are scored; other types show **Not Fraud-Eligible**
  (no probability is invented). Inputs are validated (no negatives, finite numbers, step 1-744); unusual balance patterns only produce notes because
  they are common in PaySim. Shows fraud probability, thresholds, decision, **expected loss if approved = p x amount** and **expected cost if blocked = (1-p) x FP_COST**.
* **Batch CSV:** required columns `step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest`; optional `isFraud`.
  Limits: 100,000 rows and 25 MB. Rows are never dropped silently: each gets a `Status` (Eligible / Not Fraud-Eligible / Unsupported type / Invalid input) and
  non-scored rows have no probability. Ranked table, decisions, expected loss and CSV download.
* **Ground truth:** if `isFraud` is valid (0/1), TP/FP/FN, precision, recall, missed fraud amount, false-alarm cost, total cost vs no model, and a cost-vs-threshold
  chart (selected threshold vs the optimum **for that file only**). Demo samples are labelled as not representative.
* The model file is loaded once and reloaded automatically if it changes; a missing/incompatible file gives a clear error. Nothing is hard-coded: thresholds, cost and
  metrics come from the bundle.

## 17. Installation instructions
Windows PowerShell (Python 3.12 or 3.13):
```powershell
cd path\to\Mobile-Wallet-Fraud-Detection
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
```
`numpy`, `pandas`, `scikit-learn` and `lightgbm` are pinned to the versions stored in the model bundle; keep them, otherwise the saved model may not load.

## 18. How to run
```powershell
streamlit run app.py                  # the app (use this, not "python app.py")
python -m pytest                      # the test-suite (no dataset needed)
pip install jupyterlab; jupyter lab   # then open next_gen.ipynb (or use VS Code)
```
To **regenerate the model** put the PaySim CSV in `data\` (or set `$env:PAYSIM_CSV`) and run all cells of `next_gen.ipynb` from the project folder. The last cells write
the new bundle (only after a fresh-process check passes; the old one is kept as `paysim_fraud_bundle.previous.pkl`) and the two sample CSVs.

## 19. Repository structure
```
app.py                      Streamlit app
paysim_features.py          features, preprocessing, cost utilities, decision tiers, calibration helper, OutlierCapper
paysim_scoring.py           bundle loading, input/CSV validation, batch scoring (shared by app and tests)
next_gen.ipynb              full workflow (kept at the root so it imports the module without sys.path tricks)
paysim_fraud_bundle.pkl     trained model bundle
requirements.txt  pytest.ini  .gitignore
data/README.md              where to put the PaySim CSV
tests/                      test_features.py  test_cost.py  test_model.py  test_app_inputs.py  conftest.py  synthetic_paysim.py
sample_transactions_demo.csv / sample_transactions_realistic.csv   created by the notebook (enriched demo vs natural fraud rate)
```
`tests/synthetic_paysim.py` makes **synthetic** data for tests only; nothing derived from it is a result.

## 20. Limitations
1. PaySim is **synthetic**; high scores do not guarantee production performance.
2. Scope is TRANSFER / CASH_OUT only.
3. **Post-transaction** balance features are not available before execution: no pre-authorisation claim.
4. The simulation has artefacts (very regular fraud patterns), which makes the task unrealistically easy.
5. The hourly fraud pattern is a simulator artefact. `day_of_month` (from `step`) stays in every variant.
6. The split is **random, not temporal**; account-overlap and time-ordered diagnostics are in the notebook, but they do not prove generalisation.
7. `FP_COST` is an **assumption**.
8. The missed amount is only a **proxy** for real loss.
9. Scores are raw boosted-tree outputs unless the notebook's calibration rule adopts a calibrator; expected-loss figures depend on calibration.
10. One seed, one split, no confidence intervals: last-decimal differences are noise.
11. The shipped model comes from the original run; the audited notebook has not been run on the real data.

## 21. Future improvements
Temporal and account-grouped validation on real data; a genuine **pre-authorisation** model using only information available before execution (plus
velocity/device features); an alert-capacity constraint and a measured false-alarm cost; calibration monitoring; drift monitoring and scheduled retraining;
optional SHAP explanations.

Data: PaySim is distributed on Kaggle (see the licence on the dataset page; credit E. A. Lopez-Rojas et al.).
