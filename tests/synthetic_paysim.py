"""Tiny SYNTHETIC generator that mimics the PaySim *schema* and a few qualitative patterns.

It exists only so the unit tests and the notebook smoke-run can execute without the
real 470 MB Kaggle file.  Nothing produced from it may be reported as a PaySim result.
"""
import numpy as np
import pandas as pd

COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
           "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]


def make_synthetic_paysim(n_rows=20_000, scope_fraud_rate=0.03, seed=0,
                          hard_fraud_share=0.0, lookalike_share=0.0):
    """hard_fraud_share: fraction of frauds that look legitimate; lookalike_share: fraction of
    in-scope legitimate rows that look like fraud (both make the classes overlap)."""
    rng = np.random.default_rng(seed)
    rng_extra = np.random.default_rng(seed + 10_000)      # separate stream: defaults reproduce the old data
    types = np.array(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    p = np.array([0.22, 0.35, 0.0065, 0.34, 0.0835])
    typ = rng.choice(types, size=n_rows, p=p / p.sum())
    in_scope = np.isin(typ, ["TRANSFER", "CASH_OUT"])
    fraud = in_scope & (rng.random(n_rows) < scope_fraud_rate)
    hard = fraud & (rng_extra.random(n_rows) < hard_fraud_share)
    lookalike = in_scope & ~fraud & (rng_extra.random(n_rows) < lookalike_share)
    pattern = (fraud & ~hard) | lookalike                    # rows that show the account-draining pattern

    # --- time: legit activity concentrated in daytime hours, fraud uniform over the day
    day = rng.integers(0, 31, n_rows)
    day_w = np.r_[np.full(8, 0.2), np.full(12, 1.0), np.full(4, 0.4)]
    legit_hour = rng.choice(24, size=n_rows, p=day_w / day_w.sum())
    hour = np.where(fraud & ~hard, rng.integers(0, 24, n_rows), legit_hour)
    step = np.clip(day * 24 + hour, 1, 743)

    # --- amounts and sender balances
    amount = np.round(np.clip(rng.lognormal(11.6, 1.3, n_rows), 1, 9e7), 2)
    old_org = np.zeros(n_rows)
    kind = rng.random(n_rows)
    big = np.round(amount * (1.05 + rng.random(n_rows) * 20), 2)    # balance > amount
    small = np.round(amount * rng.random(n_rows) * 0.95, 2)         # balance < amount (overdraft)
    old_org = np.where(kind < 0.80, 0.0, np.where(kind < 0.90, small, big))
    new_org = np.where(typ == "CASH_IN", old_org + amount, np.maximum(old_org - amount, 0.0))

    # fraud: the sender account is emptied (amount == old balance, new balance 0)
    f_amount = np.round(np.clip(rng.lognormal(13.0, 1.1, n_rows), 100, 9e7), 2)
    amount = np.where(pattern, f_amount, amount)
    old_org = np.where(pattern, f_amount, old_org)
    new_org = np.where(pattern, 0.0, new_org)

    # --- recipient balances
    old_dest = np.where(rng.random(n_rows) < 0.35, 0.0,
                        np.round(rng.lognormal(12.0, 1.8, n_rows), 2))
    new_dest = np.round(old_dest + amount, 2)
    zero_zero = pattern & (rng.random(n_rows) < 0.5)                  # fraud: untouched recipient balance
    old_dest = np.where(zero_zero, 0.0, old_dest)
    new_dest = np.where(zero_zero, 0.0, new_dest)
    merchant = typ == "PAYMENT"
    old_dest = np.where(merchant, 0.0, old_dest)
    new_dest = np.where(merchant, 0.0, new_dest)

    name_orig = np.array([f"C{i:09d}" for i in rng.permutation(n_rows) + 100_000_000])
    pool = rng.integers(10_000_000, 10_005_000, n_rows)
    name_dest = np.where(merchant, [f"M{v}" for v in pool], [f"C{v}" for v in pool])
    flagged = ((typ == "TRANSFER") & (amount > 200_000) & (rng.random(n_rows) < 0.002)).astype(int)

    df = pd.DataFrame({
        "step": step.astype("int16"), "type": typ, "amount": amount,
        "nameOrig": name_orig, "oldbalanceOrg": np.round(old_org, 2),
        "newbalanceOrig": np.round(new_org, 2), "nameDest": name_dest,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        "isFraud": fraud.astype("int8"), "isFlaggedFraud": flagged.astype("int8"),
    })
    return df[COLUMNS]


def write_synthetic_csv(path, **kwargs):
    df = make_synthetic_paysim(**kwargs)
    df.to_csv(path, index=False)
    return df
