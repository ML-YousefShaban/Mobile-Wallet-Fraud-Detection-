import numpy as np
import pytest

from paysim_features import (APPROVE, BLOCK, NOT_SCORED, REVIEW, cost_curve, cost_of,
                             decision_tier, near_optimal_range, recost, select_threshold)

# 3 frauds (amounts 100, 200, 300) and 4 legitimate transactions (40, 50, 60, 70)
Y = np.array([1, 1, 1, 0, 0, 0, 0])
AMOUNT = np.array([100.0, 200.0, 300.0, 40.0, 50.0, 60.0, 70.0])
FP_COST = 50.0


def test_two_tp_one_fp_one_fn_all_outputs():
    flag = np.array([True, True, False, True, False, False, False])   # catches 2 frauds, 1 false alarm
    r = cost_of(Y, flag, AMOUNT, FP_COST)
    assert (r["TP"], r["FP"], r["FN"], r["alerts"]) == (2, 1, 1, 3)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["missed_fraud_amount"] == 300.0              # only the third fraud is missed
    assert r["false_alarm_cost"] == 50.0
    assert r["total_cost"] == 350.0


def test_no_alerts_baseline():
    r = cost_of(Y, np.zeros(7, bool), AMOUNT, FP_COST)
    assert (r["TP"], r["FP"], r["FN"], r["alerts"]) == (0, 0, 3, 0)
    assert r["precision"] == 0.0 and r["recall"] == 0.0   # no alerts: precision defined as 0
    assert r["total_cost"] == r["missed_fraud_amount"] == 600.0


def test_flag_everything():
    r = cost_of(Y, np.ones(7, bool), AMOUNT, FP_COST)
    assert (r["TP"], r["FP"], r["FN"]) == (3, 4, 0)
    assert r["recall"] == 1.0 and r["precision"] == pytest.approx(3 / 7)
    assert r["missed_fraud_amount"] == 0.0 and r["total_cost"] == 200.0


def test_empty_input_and_length_mismatch():
    assert cost_of([], [], [], 50.0)["total_cost"] == 0.0
    with pytest.raises(ValueError):
        cost_of([1, 0], [True], [1.0, 2.0], 50.0)


def test_cost_curve_counts_follow_threshold():
    proba = np.array([0.9, 0.6, 0.3, 0.5, 0.2, 0.1, 0.05])
    curve = cost_curve(Y, proba, AMOUNT, FP_COST, [0.95, 0.55, 0.25, 0.04])
    assert list(curve["threshold"]) == [0.04, 0.25, 0.55, 0.95]          # sorted ascending
    assert list(curve["alerts"]) == [7, 4, 2, 0]
    assert list(curve["TP"]) == [3, 3, 2, 0]
    # t=0.04: 4 FP -> 200 | t=0.25: 3 TP + 1 FP -> 50 | t=0.55: miss the 300 fraud -> 300 | t=0.95: miss all -> 600
    assert list(curve["total_cost"]) == [200.0, 50.0, 300.0, 600.0]


def test_recost_equals_recomputation():
    proba = np.array([0.9, 0.6, 0.3, 0.5, 0.2, 0.1, 0.05])
    grid = [0.04, 0.25, 0.55, 0.95]
    base = cost_curve(Y, proba, AMOUNT, 50.0, grid)
    direct = cost_curve(Y, proba, AMOUNT, 5000.0, grid)
    np.testing.assert_allclose(recost(base, 5000.0)["total_cost"], direct["total_cost"])


def test_select_threshold_breaks_ties_toward_lower_threshold():
    proba = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.1, 0.1])               # perfectly separable
    curve = cost_curve(Y, proba, AMOUNT, FP_COST, [0.05, 0.15, 0.3, 0.5, 0.75, 0.95])
    assert select_threshold(curve) == 0.3                              # 0.3 .. 0.5 tie (cost 0)
    low, high = near_optimal_range(curve, FP_COST, tolerance_fp=0.0)
    assert (low, high) == (0.3, 0.5)
    low, high = near_optimal_range(curve, FP_COST, tolerance_fp=1.0)
    assert low == 0.15 and high == 0.5                                  # 0.15 adds 1 false alarm...


def test_decision_tiers_boundaries_and_nan():
    p = np.array([0.0, 0.2399, 0.24, 0.4999, 0.5, 1.0, np.nan])
    got = list(decision_tier(p, 0.24, 0.5))
    assert got == [APPROVE, APPROVE, REVIEW, REVIEW, BLOCK, BLOCK, NOT_SCORED]
    assert decision_tier(0.3, 0.24, 0.5) == REVIEW                      # scalar input
    # block < review: block is raised to the review threshold, so there is no review band
    assert list(decision_tier([0.3, 0.6], 0.5, 0.2)) == [APPROVE, BLOCK]
