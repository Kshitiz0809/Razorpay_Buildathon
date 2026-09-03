"""Exploratory data analysis over the raw creditcard.csv.

Writes reports/EDA_REPORT.md plus PNG figures under reports/figures/eda/.
Run after the raw CSV has been downloaded:  python scripts/eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fraud_risk.config import REPORTS_DIR, train_config
from fraud_risk.data.load import load_raw_transactions
from fraud_risk.data.split import chronological_split

FIG_DIR = REPORTS_DIR / "figures" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = load_raw_transactions()
    n = len(df)
    n_fraud = int(df["Class"].sum())
    fraud_rate = n_fraud / n

    cfg = train_config()
    split = chronological_split(
        df,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
        min_test_frauds=cfg["split"]["min_test_frauds"],
    )
    block_counts = {
        "train": (len(split.train), int(split.train["Class"].sum())),
        "val": (len(split.val), int(split.val["Class"].sum())),
        "test": (len(split.test), int(split.test["Class"].sum())),
    }

    # Figure 1: class balance
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Legitimate", "Fraud"], [n - n_fraud, n_fraud], color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_title(f"Class balance ({fraud_rate:.3%} fraud)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_balance.png", dpi=120)
    plt.close(fig)

    # Figure 2: Amount distribution by class (log scale)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df.loc[df.Class == 0, "Amount"].clip(upper=500), bins=50, alpha=0.6, label="Legitimate", density=True)
    ax.hist(df.loc[df.Class == 1, "Amount"].clip(upper=500), bins=50, alpha=0.6, label="Fraud", density=True)
    ax.set_xlabel("Amount (clipped at 500)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Transaction amount by class")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "amount_by_class.png", dpi=120)
    plt.close(fig)

    # Figure 3: fraud rate by hour-of-day
    hour = (df["Time"] % 86400) // 3600
    fraud_by_hour = df.groupby(hour)["Class"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(fraud_by_hour.index, fraud_by_hour.values, color="#C44E52")
    ax.set_xlabel("Hour of day (Time mod 24h)")
    ax.set_ylabel("Fraud rate")
    ax.set_title("Fraud rate by hour of day")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fraud_rate_by_hour.png", dpi=120)
    plt.close(fig)

    report = f"""# EDA Report

Generated from `data/raw/creditcard.csv`.

## Overview

- Rows: {n:,}
- Fraud cases: {n_fraud:,} ({fraud_rate:.4%} of all rows)
- Time span: {df['Time'].min():.0f}s to {df['Time'].max():.0f}s ({(df['Time'].max() - df['Time'].min()) / 3600:.1f} hours)
- Features: `Time`, `Amount`, `V1`..`V28` (PCA-anonymized by the dataset authors), label `Class`

## Chronological split fraud counts

| Block | Rows | Frauds | Fraud rate |
|---|---|---|---|
| Train (earliest {cfg['split']['train_frac']:.0%}) | {block_counts['train'][0]:,} | {block_counts['train'][1]} | {block_counts['train'][1] / block_counts['train'][0]:.4%} |
| Val (next {cfg['split']['val_frac']:.0%}) | {block_counts['val'][0]:,} | {block_counts['val'][1]} | {block_counts['val'][1] / block_counts['val'][0]:.4%} |
| Test (latest {1 - cfg['split']['train_frac'] - cfg['split']['val_frac']:.0%}) | {block_counts['test'][0]:,} | {block_counts['test'][1]} | {block_counts['test'][1] / block_counts['test'][0]:.4%} |

Each block's fraud count was checked against `min_test_frauds` in
`configs/train_config.yaml` *before* any model was trained -- the split
boundary is chosen on row/fraud counts only, never on resulting metrics.

## Figures

- `figures/eda/class_balance.png` -- extreme imbalance (~{fraud_rate:.3%} positive), log-scaled y-axis
- `figures/eda/amount_by_class.png` -- fraud amounts skew differently from legitimate transactions
- `figures/eda/fraud_rate_by_hour.png` -- fraud rate is not uniform across hour-of-day, motivating the cyclical hour feature

## Implications for modeling

1. Extreme class imbalance rules out plain accuracy as a metric; PR-AUC and
   a cost-sensitive threshold (see `configs/cost_config.yaml`) are used instead.
2. `V1`-`V28` are already PCA outputs from the original authors and are left
   untouched -- no attempt is made to reconstruct or reverse them.
3. `Amount` is heavily right-skewed; `log1p(Amount)` is used for the
   logistic-regression baseline (tree models are scale-invariant so LightGBM
   uses raw `Amount`).
4. `Time` is converted into an hour-of-day cyclical feature (`sin`/`cos`) to
   capture the daily fraud rhythm without exposing absolute chronology as a
   raw, potentially-leaky feature.
"""
    (REPORTS_DIR / "EDA_REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote {REPORTS_DIR / 'EDA_REPORT.md'} and 3 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
