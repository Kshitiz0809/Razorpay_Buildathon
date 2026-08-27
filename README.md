# Fraud Risk Scorer — Razorpay AI Risk Manager Track

A real-time transaction fraud-risk scoring system, built the way a payment
gateway's own risk team would build it: a calibrated, cost-optimized,
explainable classifier evaluated with a strictly chronological train/
validation/test split, so the reported numbers reflect what the model would
have actually caught in production, not an optimistic random split.

**Track:** AI Risk Manager — "Build a working detector, verifier or
auto-responder for one class of loss, with measured precision and recall on
a held-out test set." **The bar:** honest metrics including false-positive
cost; strictly defense-only.

## What this is

A transaction-level fraud classifier trained on the Kaggle [Credit Card
Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset (284,807 real anonymized card transactions, 492 frauds), served via
a FastAPI scoring/explanation API and a Streamlit dashboard for live
demonstration. See `data/README.md` for dataset provenance and
`MODEL_CARD.md` for scope, limitations, and the generalization caveat.

## Why the numbers here are trustworthy, not just high

- **Chronological split, not random.** Fraud scoring is a forecasting
  problem — production models only ever see the past. Train/val/test are
  cut by `Time`, in order, with an automated assertion that no block's time
  range overlaps another's (`tests/test_split_no_leakage.py`).
- **Baseline + champion, not just a champion.** A `LogisticRegression`
  baseline is reported alongside the LightGBM champion as an honesty
  anchor — so the champion's lift is demonstrably real, not an unverifiable
  black-box claim.
- **Calibrated probabilities.** Class-weighted GBM output isn't a true
  probability; `CalibratedClassifierCV` (sigmoid, fit on VALIDATION only)
  is required before the score can feed a dollar-cost formula at all.
- **Cost-sensitive threshold, not F1.** The operating threshold is chosen
  to minimize an explicit, editable dollar-cost model
  (`configs/cost_config.yaml`), not a generic classification metric —
  directly answering the track's "honest metrics including false-positive
  cost" bar.
- **TEST is touched exactly once**, at the end, in
  `fraud_risk/evaluation/report.py` — never during training, calibration,
  or threshold selection.

## Results

Populated by running the pipeline (`scripts/run_pipeline.ps1`) — see
`reports/evaluation_report.json` and `reports/figures/` for the full report
and PR / calibration / cost curves once trained.

<!-- RESULTS_PLACEHOLDER -->

## Architecture

```
                         ┌─────────────────────┐
  data/raw/creditcard.csv│   Kaggle dataset     │
                         └──────────┬──────────┘
                                    │  chronological split (60/20/20)
                                    ▼
                    ┌───────────────────────────────┐
                    │  fraud_risk.models.train       │
                    │  LogisticRegression (baseline) │
                    │  LightGBM (champion)           │
                    │  + sigmoid calibration on VAL   │
                    │  + cost-optimal threshold on VAL│
                    └───────────────┬────────────────┘
                                    │  models/champion/*.joblib
                                    ▼
        ┌────────────────────────────────────────────────┐
        │  fraud_risk.evaluation.report  (touches TEST once)│
        │  -> reports/evaluation_report.json + figures      │
        │  -> reports/test_predictions.parquet (cached)      │
        └───────────────────────┬────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                                 ▼
     ┌───────────────────────┐         ┌───────────────────────────┐
     │  api/  (FastAPI)        │◄───────│  dashboard/ (Streamlit)     │
     │  /health  /score  /explain│ HTTP  │  Live Stream, Performance,  │
     │  X-API-Key gated          │       │  Cost/Threshold, Explain    │
     └───────────────────────┘         └───────────────────────────┘
```

## Project structure

```
src/fraud_risk/       Core package: data, features, models, cost, evaluation, explain
configs/               train_config.yaml, cost_config.yaml (all $ assumptions live here)
api/                   FastAPI service (main, schemas, auth, dependencies)
dashboard/             Streamlit multipage app
scripts/                eda.py, download_data.ps1, run_pipeline.ps1, benchmark_latency.py, build_curated_examples.py
tests/                  pytest suite (schema, leakage, cost model, feature engineering, API contract)
reports/                Generated: evaluation_report.json, figures/, test_predictions.parquet, curated_examples.json
models/champion/        Generated: serialized model bundle + metadata (gitignored except structure)
```

## Setup

### 1. Python environment

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
```

### 2. Data

Requires a Kaggle API token — see `data/README.md`. Then:

```
scripts\download_data.ps1
```

### 3. Run the pipeline

```
scripts\run_pipeline.ps1
```

This runs EDA, trains baseline + champion, calibrates, selects the
cost-optimal threshold, evaluates once on TEST, builds curated
explainability examples, and runs the test suite.

### 4. Run the API

```
.venv\Scripts\uvicorn api.main:app --reload
```

Interactive docs at `http://localhost:8000/docs`. `/score` and `/explain`
require the `X-API-Key` header — set in `.env` (copy `.env.example` first).

### 5. Run the dashboard

```
.venv\Scripts\streamlit run dashboard/app.py
```

### 6. Docker (optional, for one-command reproducibility)

```
docker-compose up --build
```

API at `localhost:8000`, dashboard at `localhost:8501`. Model artifacts are
mounted read-only from `./models` — train locally first (step 3); retraining
never requires an image rebuild.

## Testing

```
.venv\Scripts\pytest -q
```

Schema validation, split-leakage assertions, cost-model correctness against
hand-computed values, feature-engineering correctness, and FastAPI contract
tests (the API tests auto-skip until a model has been trained).

## Defense-only scope

This system scores and explains individual transactions. It deliberately
does **not** expose: global feature importances, training data, raw model
internals, or an unauthenticated scoring oracle — `/score` and `/explain`
require an API key, and `/explain` returns only the top-6 contributing
features for the specific transaction scored, never the full feature vector
or model weights. See `MODEL_CARD.md` for the full scope statement.
