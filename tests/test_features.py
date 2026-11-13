import numpy as np
import pandas as pd
import pytest

from paysim_features import (OutlierCapper, PaySimFeatureEngineer, build_preprocessing_pipeline,
                             model_feature_names)

RAW = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
       "nameDest", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"]


def row(**kw):
    base = dict(step=26, type="TRANSFER", amount=1000.0, nameOrig="C1", oldbalanceOrg=1000.0,
                newbalanceOrig=0.0, nameDest="C2", oldbalanceDest=0.0, newbalanceDest=0.0,
                isFlaggedFraud=0)
    base.update(kw)
    return pd.DataFrame([base])[RAW]


def test_hand_computed_feature_values():
    out = PaySimFeatureEngineer().transform(row()).iloc[0]
    assert out["log_amount"] == pytest.approx(np.log1p(1000.0), rel=1e-6)
    assert out["is_account_emptied"] == 1
    assert out["amount_to_oldbalance_orig"] == pytest.approx(1000 / 1001, rel=1e-6)
    assert out["is_orig_zero_start"] == 0 and out["is_dest_zero_start"] == 1
    assert out["day_of_month"] == 2                      # 26 // 24 + 1
    assert out["hour_of_day"] == 2 and out["is_night_window"] == 1
    assert out["balance_change_orig"] == -1000 and out["orig_balance_diff"] == 0
    assert out["balance_change_dest"] == 0 and out["dest_balance_diff"] == 1000
    assert out["dest_change_ratio"] == 0 and out["is_dest_zero_both"] == 1


@pytest.mark.parametrize("step,hour,night", [(1, 1, 0), (2, 2, 1), (6, 6, 1), (7, 7, 0),
                                              (24, 0, 0), (26, 2, 1), (743, 23, 0)])
def test_hour_convention_is_step_mod_24(step, hour, night):
    out = PaySimFeatureEngineer().transform(row(step=step)).iloc[0]
    assert out["hour_of_day"] == hour and out["is_night_window"] == night


def test_account_emptied_requires_positive_balance():
    out = PaySimFeatureEngineer().transform(row(amount=0.0, oldbalanceOrg=0.0)).iloc[0]
    assert out["is_account_emptied"] == 0 and out["is_orig_zero_start"] == 1
    assert out["is_dest_zero_both"] == 0                 # needs amount > 0


def test_ratio_features_are_clipped_and_finite_for_edge_values():
    edge = pd.concat([
        row(amount=9e7, oldbalanceOrg=0.0, newbalanceOrig=0.0),
        row(amount=0.0, oldbalanceOrg=0.0, oldbalanceDest=3.5e8, newbalanceDest=0.0),
        row(amount=1.0, oldbalanceOrg=1e9, newbalanceDest=1e9),
    ], ignore_index=True)
    out = PaySimFeatureEngineer().transform(edge)
    assert np.isfinite(out.drop(columns=["type"]).to_numpy(dtype=float)).all()
    assert out["amount_to_oldbalance_orig"].between(0, 100).all()
    assert out["dest_change_ratio"].between(-10, 10).all()


def test_float32_inputs_do_not_add_rounding_noise():
    # In float32 arithmetic 16777218 - 1.5 - 16777216 evaluates to 0.0; the exact answer is 0.5.
    df = row(amount=1.5, oldbalanceOrg=16777218.0, newbalanceOrig=16777216.0)
    df32 = df.astype({c: "float32" for c in ["amount", "oldbalanceOrg", "newbalanceOrig",
                                              "oldbalanceDest", "newbalanceDest"]})
    assert PaySimFeatureEngineer().transform(df32).iloc[0]["orig_balance_diff"] == 0.5


def test_input_frame_is_not_modified():
    df = row()
    before = df.copy()
    PaySimFeatureEngineer().transform(df)
    pd.testing.assert_frame_equal(df, before)


def test_extra_columns_are_tolerated_and_raw_ids_dropped():
    df = row()
    df["isFraud"] = 1
    out = PaySimFeatureEngineer().transform(df)
    assert not {"step", "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceDest"} & set(out.columns)


@pytest.mark.parametrize("flags,missing", [
    (dict(include_post_tx=True), "newbalanceOrig"),
    (dict(include_post_tx=True), "newbalanceDest"),
])
def test_missing_columns_raise_clear_error(flags, missing):
    df = row().drop(columns=[missing])
    with pytest.raises(ValueError, match=missing):
        PaySimFeatureEngineer(**flags).transform(df)
    PaySimFeatureEngineer(include_post_tx=False).transform(df)   # not needed without post-tx features


def test_non_dataframe_is_rejected():
    with pytest.raises(TypeError):
        PaySimFeatureEngineer().transform(np.zeros((2, 3)))


@pytest.mark.parametrize("post,hour,n_features", [(True, True, 17), (False, True, 11),
                                                  (True, False, 15), (False, False, 9)])
def test_pipeline_output_width_and_names(scope_df, post, hour, n_features):
    X = scope_df.drop(columns=["isFraud"])
    pipe = build_preprocessing_pipeline(include_post_tx=post, include_hour=hour)
    out = pipe.fit_transform(X)
    names = model_feature_names(pipe)
    assert out.shape == (len(X), n_features) == (len(X), len(names))
    assert names[-2:] == ["type_CASH_OUT", "type_TRANSFER"]
    assert ("hour_of_day" in names) == hour and ("orig_balance_diff" in names) == post


def test_feature_order_is_deterministic(scope_df):
    X = scope_df.drop(columns=["isFraud"])
    pipe = build_preprocessing_pipeline().fit(X)
    assert model_feature_names(pipe)[:9] == ["amount", "log_amount", "is_account_emptied",
                                            "amount_to_oldbalance_orig", "is_orig_zero_start",
                                            "is_dest_zero_start", "day_of_month",
                                            "hour_of_day", "is_night_window"]
    shuffled = X[list(reversed(X.columns))]                      # column order must not matter
    np.testing.assert_array_equal(pipe.transform(X), pipe.transform(shuffled))


def test_single_row_and_multi_row_agree(scope_df):
    X = scope_df.drop(columns=["isFraud"])
    pipe = build_preprocessing_pipeline().fit(X)
    batch = pipe.transform(X.iloc[:5])
    singles = np.vstack([pipe.transform(X.iloc[[i]]) for i in range(5)])
    np.testing.assert_allclose(batch, singles)


def test_missing_values_are_imputed_not_propagated(scope_df):
    X = scope_df.drop(columns=["isFraud"])
    pipe = build_preprocessing_pipeline().fit(X)
    dirty = X.iloc[:3].copy()
    dirty.loc[dirty.index[0], "amount"] = np.nan
    dirty.loc[dirty.index[1], "oldbalanceDest"] = np.nan
    dirty.loc[dirty.index[2], "type"] = None
    assert np.isfinite(pipe.transform(dirty)).all()


def test_unseen_transaction_type_is_encoded_as_all_zeros(scope_df):
    X = scope_df.drop(columns=["isFraud"])
    pipe = build_preprocessing_pipeline().fit(X)
    out = pipe.transform(X.iloc[:2].assign(type="PAYMENT"))
    assert (out[:, -2:] == 0).all()


# ------------------------------------ outlier capping ------------------------------------ #
def test_outlier_capper_caps_the_tail_using_training_statistics_only():
    rng = np.random.default_rng(0)
    train = np.column_stack([rng.lognormal(10, 1, 5000), rng.lognormal(10, 1, 5000)])
    capper = OutlierCapper().fit(train)
    out = capper.transform(np.array([[1e12, 5.0], [1.0, 1e12]]))
    assert out[0, 0] == pytest.approx(capper.upper_[0]) and out[0, 0] < 1e12   # extreme value capped
    assert out[1, 0] == 1.0 and out[0, 1] == 5.0                                  # in-range values untouched
    assert OutlierCapper().fit(train * 10).upper_[0] > capper.upper_[0]           # fences come from the data it was fitted on


def test_outlier_capper_leaves_binary_constant_and_zero_values_alone():
    X = np.column_stack([np.r_[np.zeros(95), np.ones(5)], np.full(100, 7.0),
                         np.r_[np.zeros(60), np.linspace(1, 1000, 40)]])
    out = OutlierCapper().fit(X).transform(X)
    np.testing.assert_array_equal(out[:, 0], X[:, 0])
    np.testing.assert_array_equal(out[:, 1], X[:, 1])
    assert (out[X[:, 2] == 0, 2] == 0).all()


def test_outlier_capper_handles_negative_tail_and_nan():
    rng = np.random.default_rng(1)
    signed = rng.choice([-1.0, 1.0], 3000) * rng.lognormal(8, 1, 3000)
    capper = OutlierCapper().fit(signed.reshape(-1, 1))
    assert capper.lower_[0] < 0 < capper.upper_[0]
    out = capper.transform(np.array([[-1e12], [1e12], [np.nan]]))
    assert out[0, 0] == pytest.approx(capper.lower_[0]) and out[1, 0] == pytest.approx(capper.upper_[0])
    assert np.isnan(out[2, 0])


def test_cap_outliers_option_changes_only_the_pipeline_structure(scope_df):
    X = scope_df.drop(columns=["isFraud"])
    default, capped = build_preprocessing_pipeline(), build_preprocessing_pipeline(cap_outliers=True)
    assert list(default.named_steps["preprocessor"].transformers[0][1].named_steps) == ["imputer", "scaler"]
    assert list(capped.named_steps["preprocessor"].transformers[0][1].named_steps) == ["imputer", "capper", "scaler"]
    out_default, out_capped = default.fit_transform(X), capped.fit_transform(X)
    assert out_default.shape == out_capped.shape == (len(X), 17)
    assert np.abs(out_capped).max() <= np.abs(out_default).max()
