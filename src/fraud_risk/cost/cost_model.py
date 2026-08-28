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


def sweep_thresholds(y_true, amount, scores, cfg: dict) -> pd.DataFrame:
    """Exact cost curve at every distinct flagging policy, "flag nothing"
    through "flag everything," in one O(n log n) pass.

    Ranks rows by score descending and walks the cutover one tie-group at a
    time using cumulative sums, rather than re-scoring the whole dataset per
    candidate threshold. Critically, this always includes threshold=+inf
    ("flag nothing") as a candidate -- a naive sweep over np.unique(scores)
    never can, since every real score is < +inf, which means it can never
    select "flag nothing" even when that is the true cost-minimizing policy.
    """
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)

    fn_cfg = cfg["false_negative"]
    fp_cfg = cfg["false_positive"]
    tp_cost = cfg["true_positive"]["investigation_cost"]
    tn_cost = cfg["true_negative"]["cost"]

    is_fraud = y_true == 1
    flagged_cost = np.where(
        is_fraud,
        tp_cost,
        fp_cfg["avg_friction_cost"] + fp_cfg["lost_revenue_pct_of_amount"] * amount + fp_cfg["churn_risk_cost"],
    )
    unflagged_cost = np.where(is_fraud, amount + fn_cfg["flat_chargeback_fee"], tn_cost)

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_flagged_cost = flagged_cost[order]
    sorted_unflagged_cost = unflagged_cost[order]

    # total_cost_by_k[k] = cost if the top-k highest-scored rows are flagged
    cum_flagged = np.concatenate([[0.0], np.cumsum(sorted_flagged_cost)])
    cum_unflagged_top = np.concatenate([[0.0], np.cumsum(sorted_unflagged_cost)])
    suffix_unflagged = sorted_unflagged_cost.sum() - cum_unflagged_top
    total_cost_by_k = cum_flagged + suffix_unflagged

    # Only keep k at tie-group boundaries (plus k=0 "flag none" and k=n "flag
    # all") -- flagging part of a group of equally-scored rows isn't a real
    # threshold.
    is_boundary = np.ones(n + 1, dtype=bool)
    if n > 1:
        is_boundary[1:n] = sorted_scores[1:] != sorted_scores[:-1]
    k_values = np.nonzero(is_boundary)[0]

    thresholds = np.empty(len(k_values))
    for i, k in enumerate(k_values):
        thresholds[i] = np.inf if k == 0 else sorted_scores[k - 1]

    return (
        pd.DataFrame({"threshold": thresholds, "total_cost": total_cost_by_k[k_values]})
        .sort_values("threshold")
        .reset_index(drop=True)
    )


def select_optimal_threshold(y_true, amount, scores, cfg: dict):
    """Returns (best_threshold, min_cost, full_cost_curve_df).

    best_threshold can be +inf, meaning "flag nothing" was cost-optimal
    under the given assumptions -- a legitimate, meaningful answer, not an
    error case.
    """
    curve = sweep_thresholds(y_true, amount, scores, cfg)
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
