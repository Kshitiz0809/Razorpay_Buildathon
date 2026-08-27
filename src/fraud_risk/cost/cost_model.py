"""Cost-sensitive threshold selection.

All dollar assumptions live in configs/cost_config.yaml, never hardcoded
here -- this module only implements the arithmetic, so the assumptions stay
auditable and swappable without touching code.
"""
import numpy as np
import pandas as pd


def _row_costs(y_true: np.ndarray, amount: np.ndarray, flagged: np.ndarray, cfg: dict) -> np.ndarray:
    fn_cfg = cfg["false_negative"]
    fp_cfg = cfg["false_positive"]
    tp_cost = cfg["true_positive"]["investigation_cost"]
    tn_cost = cfg["true_negative"]["cost"]

    is_fraud = y_true == 1
    fn_cost = amount + fn_cfg["flat_chargeback_fee"]
    fp_cost = (
        fp_cfg["avg_friction_cost"]
        + fp_cfg["lost_revenue_pct_of_amount"] * amount
        + fp_cfg["churn_risk_cost"]
    )

    return np.where(
        is_fraud,
        np.where(flagged, tp_cost, fn_cost),
        np.where(flagged, fp_cost, tn_cost),
    )


def cost_of_decisions(y_true, amount, flagged, cfg: dict) -> float:
    """Total cost of already-made flag/no-flag decisions (e.g. from a live API)."""
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype=float)
    flagged = np.asarray(flagged, dtype=bool)
    return float(_row_costs(y_true, amount, flagged, cfg).sum())


def total_cost(y_true, amount, scores, threshold: float, cfg: dict) -> float:
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype=float)
    scores = np.asarray(scores, dtype=float)
    flagged = scores >= threshold
    return float(_row_costs(y_true, amount, flagged, cfg).sum())


def sweep_thresholds(y_true, amount, scores, cfg: dict, n_candidates: int = 500) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype=float)
    scores = np.asarray(scores, dtype=float)

    candidates = np.unique(scores)
    if len(candidates) > n_candidates:
        candidates = np.unique(np.quantile(scores, np.linspace(0, 1, n_candidates)))

    costs = np.array([total_cost(y_true, amount, scores, t, cfg) for t in candidates])
    return (
        pd.DataFrame({"threshold": candidates, "total_cost": costs})
        .sort_values("threshold")
        .reset_index(drop=True)
    )


def select_optimal_threshold(y_true, amount, scores, cfg: dict, n_candidates: int = 500):
    """Returns (best_threshold, min_cost, full_cost_curve_df)."""
    curve = sweep_thresholds(y_true, amount, scores, cfg, n_candidates)
    best_idx = curve["total_cost"].idxmin()
    return (
        float(curve.loc[best_idx, "threshold"]),
        float(curve.loc[best_idx, "total_cost"]),
        curve,
    )


def cost_flag_none(y_true, amount, cfg: dict) -> float:
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype=float)
    flagged = np.zeros(len(y_true), dtype=bool)
    return float(_row_costs(y_true, amount, flagged, cfg).sum())


def cost_flag_all(y_true, amount, cfg: dict) -> float:
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype=float)
    flagged = np.ones(len(y_true), dtype=bool)
    return float(_row_costs(y_true, amount, flagged, cfg).sum())
