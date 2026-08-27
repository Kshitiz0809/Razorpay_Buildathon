# Model Card — Fraud Risk Scorer (champion)

## Overview

- **Task:** binary classification — is a card transaction fraudulent?
- **Architecture:** LightGBM gradient-boosted trees, class-weighted
  (`scale_pos_weight`), probability-calibrated via sigmoid (Platt) scaling.
- **Baseline reported alongside:** `LogisticRegression(class_weight="balanced")`,
  as an honesty anchor, not a deployment candidate.
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

Chronological split (train earliest 60% / validation next 20% / test latest
20%, by `Time`), enforced by an automated no-overlap assertion. Model
selection and calibration use TRAIN and VALIDATION only. TEST is scored
exactly once, in `fraud_risk/evaluation/report.py`, to produce the numbers
in `reports/evaluation_report.json`.

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
   whole dataset, the TEST block's fraud count (see
   `reports/evaluation_report.json` -> `model_metadata`) is small enough
   that precision/recall carry real sampling uncertainty — reported as
   point estimates, not confidence intervals, in this prototype.
5. **Static snapshot.** No concept-drift monitoring or online retraining is
   implemented; a real deployment would need both, since fraud patterns
   shift faster than most other ML problems.

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
