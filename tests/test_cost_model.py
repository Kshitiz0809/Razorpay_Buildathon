import numpy as np
import pytest

from fraud_risk.cost.cost_model import (
    total_cost,
    select_optimal_threshold,
    cost_flag_none,
    cost_flag_all,
)

CFG = {
    "false_negative": {"flat_chargeback_fee": 10.0},
    "false_positive": {
        "avg_friction_cost": 5.0,
        "lost_revenue_pct_of_amount": 0.0,
        "churn_risk_cost": 0.0,
    },
    "true_positive": {"investigation_cost": 2.0},
    "true_negative": {"cost": 0.0},
}

# 4 rows: [legit low score, legit high score, fraud low score, fraud high score]
Y = np.array([0, 0, 1, 1])
AMOUNT = np.array([100.0, 100.0, 200.0, 200.0])
SCORES = np.array([0.1, 0.9, 0.2, 0.8])


def test_total_cost_hand_computed_at_threshold_0_5():
    # threshold 0.5 -> flagged = [F, T, F, T]
    # row0 legit unflagged -> 0
    # row1 legit flagged -> FP cost = 5.0
    # row2 fraud unflagged -> FN cost = amount(200) + fee(10) = 210
    # row3 fraud flagged -> TP cost = 2.0
    # total = 0 + 5 + 210 + 2 = 217
    cost = total_cost(Y, AMOUNT, SCORES, 0.5, CFG)
    assert cost == pytest.approx(217.0)


def test_flag_all_and_flag_none_bounds():
    # flag none: both frauds missed -> (200+10)*2 = 420, legits cost 0
    assert cost_flag_none(Y, AMOUNT, CFG) == pytest.approx(420.0)
    # flag all: both legits FP (5 each) = 10, both frauds TP (2 each) = 4 -> 14
    assert cost_flag_all(Y, AMOUNT, CFG) == pytest.approx(14.0)


def test_optimal_threshold_beats_naive_extremes():
    best_t, best_cost, curve = select_optimal_threshold(Y, AMOUNT, SCORES, CFG)
    assert best_cost <= cost_flag_none(Y, AMOUNT, CFG)
    assert best_cost <= cost_flag_all(Y, AMOUNT, CFG)
    assert set(["threshold", "total_cost"]).issubset(curve.columns)


def test_optimal_threshold_finds_known_minimum():
    # Candidate thresholds are the unique scores [0.1, 0.2, 0.8, 0.9].
    # t=0.1 flags all 4 -> 5+5+2+2 = 14
    # t=0.2 flags rows 1,2,3 (skips the cheap legit row0) -> 0+5+2+2 = 9  <- minimum
    # t=0.8 flags rows 1,3 -> 0+5+210+2 = 217
    # t=0.9 flags row 1 only -> 0+5+210+210 = 425
    best_t, best_cost, _ = select_optimal_threshold(Y, AMOUNT, SCORES, CFG)
    assert best_cost == pytest.approx(9.0)
    assert best_t == pytest.approx(0.2)
