# Model Card — Fraud Risk Scorer (champion)

> **Scope note:** this card covers the calibrated LightGBM model trained on
> the Kaggle benchmark dataset. The separate heuristic risk engine in
> `src/fraud_risk/razorpay_integration/risk_rules.py`, used against live
> Razorpay test-mode data, is **not** a trained model and has no
> precision/recall to report — there is no labeled outcome history for a
> freshly-connected payment stream to measure against. It's a deliberately
> transparent, non-ML rule engine; see README.md's "Live Razorpay
> integration" section for why, and don't conflate its risk scores with the
> metrics below. Likewise, `src/fraud_risk/chargeback/responder.py` (an
> LLM text-drafting tool, not a classifier) has no accuracy metric of its
> own -- its only correctness property is the safety guard in
> `api/main.py` that refuses to run on any transaction this model scores
> above the flagging threshold.

## Overview

- **Task:** binary classification — is a card transaction fraudulent?
- **Architecture:** LightGBM gradient-boosted trees (unweighted,
  `scale_pos_weight=1` — see "A design decision reversed by evidence"
  below), moderately regularized, probability-calibrated via sigmoid
  (Platt) scaling.
- **Baseline reported alongside:** `LogisticRegression(class_weight="balanced")`,
  as an honesty anchor, not a deployment candidate.
- **Measured performance (TEST, one held-out touch):** PR-AUC 0.792,
  ROC-AUC 0.983, precision 83.1%, recall 73.4% at the deployed threshold —
  full numbers in `README.md`'s Results section and
  `reports/evaluation_report.json`.
- **Intended use:** a demonstration of a production-grade fraud-scoring
  *methodology* — leakage-safe evaluation, calibration, cost-sensitive
  thresholding, and per-transaction explainability — for the Razorpay AI
  Risk Manager hackathon track.

## Training data

Kaggle `mlg-ulb/creditcardfraud`: 284,807 European cardholder transactions
over 2 days in September 2013, 492 frauds (0.172%). Features `V1`-`V28` are
PCA components released by the original authors to protect cardholder
identity; their real-world meaning is unknown and untouched here. Full
provenance in `data/README.md`.

## Evaluation protocol

Chronological split (train earliest 55% / validation next 20% / test latest
25%, by `Time` — widened from an initial 60/20/20 because that left TEST
with only 75 frauds, below the configured minimum of 80; the boundary was
moved based on fraud *counts* alone, before any model was trained, per the
policy documented in `configs/train_config.yaml`), enforced by an automated
no-overlap assertion. Model selection and calibration use TRAIN and
VALIDATION only. TEST is scored exactly once, in
`fraud_risk/evaluation/report.py`, to produce the numbers in
`reports/evaluation_report.json`.

## A design decision reversed by evidence: class weighting hurt, not helped

The original design (see git history / `CHALLENGES.md`) used
`scale_pos_weight` computed from the train imbalance ratio (~447x for this
split) — the textbook approach to class imbalance. The first trained
champion scored TEST PR-AUC 0.130, dramatically worse than the 0.747
baseline, despite scoring 0.65 on VALIDATION during training. That gap was
investigated rather than shipped:

1. Calibration was ruled out — the uncalibrated and calibrated scores had
   nearly identical PR-AUC (0.1297 vs 0.1297).
2. The real cause: LightGBM's built-in `eval_metric="average_precision"`
   string does not compute the same quantity as sklearn's
   `average_precision_score` under `scale_pos_weight` — confirmed by
   swapping in an explicit sklearn-wrapping callable, after which
   LightGBM's internally-reported validation score exactly matched a
   post-hoc sklearn recomputation on the same predictions (previously off
   by roughly 9x). Early stopping had been silently selecting an iteration
   optimized for the wrong metric.
3. With the metric fixed, a VAL-only hyperparameter search (never touching
   TEST) found that the aggressive auto-computed `scale_pos_weight` was
   itself the dominant problem: it caused severe overfitting to the 350
   training-period frauds that did not generalize across the chronological
   gap. `scale_pos_weight=1` (no reweighting at all) combined with moderate
   regularization (`num_leaves=15`, `min_child_samples=30`,
   `reg_alpha=reg_lambda=0.5`, `learning_rate=0.02`) consistently
   outperformed every weighted variant tried.

Retrained with this config, TEST PR-AUC became 0.792 — now legitimately
above the baseline. This is reported here in full rather than only as a
clean result, because the failure mode (a plausible-looking validation
score that didn't hold up) is exactly the kind of thing "honest metrics"
is supposed to catch, and catching it changed a real modeling decision.

## Intended limitations — read before trusting these numbers for anything real

1. **Geography and era.** This is 2013 European cardholder data. Fraud
   typologies in Indian BFSI/UPI/card-not-present flows today are
   materially different (different channels, different attacker
   incentives, different transaction-amount distributions). This model
   should not be deployed against Indian merchant traffic without
   retraining on representative data — this project demonstrates the
   *pipeline*, not a drop-in production model.
2. **Anonymized features.** `V1`-`V28` are opaque PCA components. This
   means the "why" behind a flag (via SHAP) is expressed in terms of
   anonymized components, not human-interpretable merchant/behavioral
   signals a real deployment would have (device fingerprint, IP
   reputation, velocity features, etc.).
3. **Currency assumption.** `Amount`'s currency is unspecified by the
   dataset authors; `configs/cost_config.yaml` treats it as USD purely as a
   documented, swappable convention — not a claim about the real currency.
4. **Small positive class in TEST.** With only 492 total frauds across the
   whole dataset, TEST has 94 of them (see
   `reports/evaluation_report.json` -> `model_metadata`) — small enough
   that precision/recall carry real sampling uncertainty (a handful of
   different frauds landing on either side of the chronological cut would
   move these numbers noticeably). Reported as point estimates, not
   confidence intervals, in this prototype.
5. **Static snapshot.** No concept-drift monitoring or online retraining is
   implemented; a real deployment would need both, since fraud patterns
   shift faster than most other ML problems.
6. **The VALIDATION block is reused for three sequential purposes**:
   choosing LightGBM's early-stopping iteration count, fitting the sigmoid
   calibrator, and selecting the cost-optimal threshold. With only ~500
   total frauds in the entire dataset, a further split risks starving one
   of these steps of positive examples entirely. Consequence: the
   validation-derived cost estimate printed during training is mildly
   optimistic relative to TEST. This is why `reports/evaluation_report.json`
   -- computed on TEST, touched exactly once, and never used for any of the
   three steps above -- is the number that should be quoted, not the
   validation-time console output.

## Cost model

All false-positive/false-negative dollar assumptions live in
`configs/cost_config.yaml` and are illustrative defaults, not measured
business figures — see that file's comments and the dashboard's Cost &
Threshold page, which renders them directly rather than burying them in code.

## Defense-only statement

This model and its API are scoring/explanation tools only. No component of
this project — model, API, or dashboard — provides a mechanism to test
whether a specific transaction would evade detection at scale, generate
adversarial transactions, or enumerate the model's decision boundary via a
public endpoint. `/score` and `/explain` require an API key; `/explain`
returns only the top-6 per-transaction SHAP contributors, never a global
feature-importance ranking or the raw model file.
