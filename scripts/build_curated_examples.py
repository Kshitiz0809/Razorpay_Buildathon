"""Builds reports/curated_examples.json: a handful of hand-picked TP/FP/TN/
borderline TEST transactions with SHAP explanations, for the dashboard's
Explainability page and for the pitch video.

Run after training + evaluation/report have produced reports/test_predictions.parquet:
  python scripts/build_curated_examples.py
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from fraud_risk.config import DATA_PROCESSED, REPORTS_DIR, FEATURE_COLUMNS, LABEL_COLUMN
from fraud_risk.artifacts import load_champion, load_threshold
from fraud_risk.explain.shap_explainer import ShapExplainer

N_PER_CATEGORY = 2


def main():
    test_df = pd.read_parquet(DATA_PROCESSED / "test.parquet").reset_index(drop=True)
    preds = pd.read_parquet(REPORTS_DIR / "test_predictions.parquet")
    assert len(test_df) == len(preds), "test.parquet and test_predictions.parquet row counts diverged"

    merged = pd.concat(
        [test_df[FEATURE_COLUMNS + [LABEL_COLUMN]], preds[["transaction_id", "champion_score", "flagged"]]],
        axis=1,
    )
    threshold = load_threshold()

    true_positive = merged[(merged.Class == 1) & (merged.flagged == 1)].nlargest(N_PER_CATEGORY, "champion_score")
    false_positive = merged[(merged.Class == 0) & (merged.flagged == 1)].nlargest(N_PER_CATEGORY, "champion_score")
    false_negative = merged[(merged.Class == 1) & (merged.flagged == 0)].nlargest(N_PER_CATEGORY, "champion_score")
    true_negative_pool = merged[(merged.Class == 0) & (merged.flagged == 0)]
    true_negative = true_negative_pool.sample(
        n=min(N_PER_CATEGORY, len(true_negative_pool)), random_state=42
    )
    merged["dist_to_threshold"] = (merged.champion_score - threshold).abs()
    borderline = merged.nsmallest(N_PER_CATEGORY, "dist_to_threshold")

    categories = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "borderline": borderline,
    }

    champion = load_champion()
    explainer = ShapExplainer(champion)

    examples = []
    for category, rows in categories.items():
        for _, row in rows.iterrows():
            raw_row = row[FEATURE_COLUMNS].to_frame().T
            contributions = explainer.explain_one(raw_row)
            examples.append(
                {
                    "transaction_id": row["transaction_id"],
                    "category": category,
                    "true_label": "fraud" if row[LABEL_COLUMN] == 1 else "legitimate",
                    "flagged": bool(row["flagged"]),
                    "champion_score": float(row["champion_score"]),
                    "amount": float(row["Amount"]),
                    "threshold": float(threshold),
                    "top_contributors": [asdict(c) for c in contributions],
                }
            )

    with open(REPORTS_DIR / "curated_examples.json", "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Wrote {len(examples)} curated examples to {REPORTS_DIR / 'curated_examples.json'}")


if __name__ == "__main__":
    main()
