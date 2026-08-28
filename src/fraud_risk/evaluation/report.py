"""Generates the final evaluation report -- the one and only time TEST is touched.

Run:  python -m fraud_risk.evaluation.report
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fraud_risk.config import DATA_PROCESSED, REPORTS_DIR, FEATURE_COLUMNS, LABEL_COLUMN, cost_config
from fraud_risk.artifacts import load_champion, load_baseline, load_threshold, load_metadata
from fraud_risk.evaluation.metrics import classification_metrics, pr_curve_points, calibration_curve_points
from fraud_risk.cost.cost_model import sweep_thresholds, cost_flag_none, cost_flag_all, total_cost

FIG_DIR = REPORTS_DIR / "figures"


def main():
    test_df = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[LABEL_COLUMN]
    amount_test = test_df["Amount"].values

    champion = load_champion()
    baseline = load_baseline()
    threshold = load_threshold()
    cost_cfg = cost_config()

    champion_scores = champion.predict_proba(X_test)[:, 1]
    baseline_scores = baseline.predict_proba(X_test)[:, 1]

    champion_metrics = classification_metrics(y_test, champion_scores, threshold)
    baseline_metrics = classification_metrics(y_test, baseline_scores, 0.5)

    realized_cost = total_cost(y_test.values, amount_test, champion_scores, threshold, cost_cfg)
    naive_none = cost_flag_none(y_test.values, amount_test, cost_cfg)
    naive_all = cost_flag_all(y_test.values, amount_test, cost_cfg)
    naive_fixed_50 = total_cost(y_test.values, amount_test, champion_scores, 0.5, cost_cfg)

    report = {
        "champion": champion_metrics,
        "baseline_logistic_regression": baseline_metrics,
        "cost_usd": {
            "at_selected_threshold": realized_cost,
            "flag_nothing_naive": naive_none,
            "flag_everything_naive": naive_all,
            "fixed_threshold_0.5_naive": naive_fixed_50,
            "savings_vs_flag_nothing": naive_none - realized_cost,
            "savings_vs_fixed_0.5": naive_fixed_50 - realized_cost,
        },
        "model_metadata": load_metadata(),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Cache predictions so the dashboard's Performance/Cost pages are instant
    # and deterministic -- they read this instead of hitting the live API.
    preds = test_df[["Time", "Amount", "Class"]].copy().reset_index(drop=True)
    preds.insert(0, "transaction_id", [f"txn_{i:06d}" for i in range(len(preds))])
    preds["champion_score"] = champion_scores
    preds["baseline_score"] = baseline_scores
    preds["flagged"] = (champion_scores >= threshold).astype(int)
    preds.to_parquet(REPORTS_DIR / "test_predictions.parquet")

    _write_figures(y_test.values, champion_scores, amount_test, threshold, cost_cfg, champion_metrics)

    print(json.dumps(report["champion"], indent=2))
    print(f"Saved report to {REPORTS_DIR / 'evaluation_report.json'}")
    print(f"Saved cached predictions to {REPORTS_DIR / 'test_predictions.parquet'}")


def _write_figures(y_test, scores, amount, threshold, cost_cfg, metrics):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pr = pr_curve_points(y_test, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pr["recall"], pr["precision"])
    ax.scatter([metrics["recall"]], [metrics["precision"]], color="red", zorder=5, label="selected threshold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (TEST)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curve.png", dpi=120)
    plt.close(fig)

    calib = calibration_curve_points(y_test, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(calib["mean_predicted"], calib["mean_actual"], marker="o")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Calibration curve (TEST)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_curve.png", dpi=120)
    plt.close(fig)

    curve = sweep_thresholds(y_test, amount, scores, cost_cfg)
    finite_curve = curve[curve["threshold"] != float("inf")]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(finite_curve["threshold"], finite_curve["total_cost"])
    if threshold != float("inf"):
        ax.axvline(threshold, color="red", linestyle="--", label=f"selected t={threshold:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total expected cost ($)")
    ax.set_title("Cost vs. threshold (TEST)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost_curve.png", dpi=120)
    plt.close(fig)

    cm = [
        [metrics["true_negatives"], metrics["false_positives"]],
        [metrics["false_negatives"], metrics["true_positives"]],
    ]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Legit", "Pred Fraud"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual Legit", "Actual Fraud"])
    ax.set_title("Confusion matrix (TEST)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
