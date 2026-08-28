import numpy as np
import pytest

from fraud_risk.cost.cost_model import (
    total_cost,
    sweep_thresholds,
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


def test_sweep_considers_flag_nothing_as_a_candidate():
    # Regression test: when false-positive/investigation costs dominate
    # fraud losses, "flag nothing" (threshold=+inf) must be a reachable
    # candidate -- a sweep restricted to real score values can never select
    # it, since every real score is < +inf.
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) < 0.05).astype(int)  # 5% fraud
    amount = rng.uniform(10, 50, size=n)  # small fraud losses
    scores = rng.random(n)

    expensive_cfg = {
        "false_negative": {"flat_chargeback_fee": 5.0},
        "false_positive": {
            "avg_friction_cost": 50.0,  # blocking anything is very costly
            "lost_revenue_pct_of_amount": 0.5,
            "churn_risk_cost": 50.0,
        },
        "true_positive": {"investigation_cost": 40.0},
        "true_negative": {"cost": 0.0},
    }

    best_t, best_cost, curve = select_optimal_threshold(y, amount, scores, expensive_cfg)
    none_cost = cost_flag_none(y, amount, expensive_cfg)

    assert best_t == float("inf")
    assert best_cost == pytest.approx(none_cost)
    assert (curve["threshold"] == float("inf")).sum() == 1


def test_sweep_matches_naive_total_cost_at_every_candidate_threshold():
    # Cross-check the vectorized cumulative-sum sweep against the simple,
    # obviously-correct per-threshold total_cost() for every finite
    # candidate it produces.
    rng = np.random.default_rng(1)
    n = 200
    y = (rng.random(n) < 0.1).astype(int)
    amount = rng.uniform(5, 300, size=n)
    scores = rng.random(n)

    curve = sweep_thresholds(y, amount, scores, CFG)
    finite = curve[curve["threshold"] != float("inf")]
    for t, expected_cost in zip(finite["threshold"], finite["total_cost"]):
        assert total_cost(y, amount, scores, t, CFG) == pytest.approx(expected_cost)
