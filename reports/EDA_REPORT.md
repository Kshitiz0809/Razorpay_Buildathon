# EDA Report

Generated from `data/raw/creditcard.csv`.

## Overview

- Rows: 284,807
- Fraud cases: 492 (0.1727% of all rows)
- Time span: 0s to 172792s (48.0 hours)
- Features: `Time`, `Amount`, `V1`..`V28` (PCA-anonymized by the dataset authors), label `Class`

## Chronological split fraud counts

| Block | Rows | Frauds | Fraud rate |
|---|---|---|---|
| Train (earliest 55%) | 156,643 | 350 | 0.2234% |
| Val (next 20%) | 56,961 | 48 | 0.0843% |
| Test (latest 25%) | 71,203 | 94 | 0.1320% |

Each block's fraud count was checked against `min_test_frauds` in
`configs/train_config.yaml` *before* any model was trained -- the split
boundary is chosen on row/fraud counts only, never on resulting metrics.

## Figures

- `figures/eda/class_balance.png` -- extreme imbalance (~0.173% positive), log-scaled y-axis
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
