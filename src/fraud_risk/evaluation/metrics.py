import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
)


def classification_metrics(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "n_flagged": int(tp + fp),
        "n_total": int(len(y_true)),
    }


def pr_curve_points(y_true, scores, max_points: int = 200) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(precision) > max_points:
        idx = np.linspace(0, len(precision) - 1, max_points).astype(int)
        precision, recall = precision[idx], recall[idx]
    return {"precision": precision.tolist(), "recall": recall.tolist()}


def calibration_curve_points(y_true, scores, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(scores, bins[1:-1])

    mean_predicted, mean_actual, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        mean_predicted.append(float(scores[mask].mean()))
        mean_actual.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))

    return {"mean_predicted": mean_predicted, "mean_actual": mean_actual, "counts": counts}
