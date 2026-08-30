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

Two things, deliberately built as a matched pair:

1. **A calibrated fraud classifier**, trained and rigorously evaluated on
   the Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   dataset (284,807 real anonymized card transactions, 492 frauds) — see
   `data/README.md` for provenance and `MODEL_CARD.md` for scope and
   limitations.
2. **A live integration against Razorpay's own Test Mode API** — not a
   simulation of Razorpay, an actual authenticated client of it, using
   Razorpay's real Orders API and implementing Razorpay's real webhook
   HMAC-SHA256 signature scheme for a payment-risk gate. See "Live Razorpay
   integration" below.

Both are served through the same FastAPI backend and the same Streamlit
dashboard.

## Live Razorpay integration

Real Razorpay accounts have no fraud-labeled outcome history on day one —
you can't calibrate a supervised model against chargebacks that haven't
happened yet. So this project demonstrates the two-stage strategy a real
risk team actually ships:

- **Stage 1 — live now, in `src/fraud_risk/razorpay_integration/`.** A
  transparent, config-driven heuristic risk gate (`configs/razorpay_risk_rules.yaml`)
  wired to Razorpay's real API: `client.py` authenticates with real test
  credentials against the real Orders API (verified live — see
  `CHALLENGES.md`); `webhook.py` implements Razorpay's actual
  HMAC-SHA256 `X-Razorpay-Signature` verification; the `/webhook/razorpay/payment`
  endpoint is exactly what a production integration would register with
  Razorpay. Because Razorpay deliberately keeps payment completion
  client-side only (PCI-DSS scope reduction — no backend can finish a
  payment alone), the risk-scoring path is exercised with signed events
  shaped exactly like Razorpay's real `payment.authorized` webhook
  (`scripts/simulate_razorpay_webhook.py`, also live in the dashboard's
  **Razorpay Live** page) — same schema, same signature algorithm, only the
  sender differs from a real production deployment.
- **Stage 2 — the calibrated model above**, ready to retrain the moment
  labeled outcomes exist for a real payment stream, using the exact
  leakage-safe, cost-aware, explainable methodology already proven on the
  benchmark dataset.

Dashboard: [Razorpay Live](dashboard/pages/5_Razorpay_Live.py) shows real
fetched test-mode orders and lets you fire simulated payment events at the
live risk gate, including a velocity-abuse demo (repeat the same payer to
watch the rule trigger statefully in real time).

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
     ┌─────────────────────────────┐     ┌───────────────────────────┐
     │  api/  (FastAPI)              │◄────│  dashboard/ (Streamlit)     │
     │  /health  /score  /explain    │ HTTP │  Live Stream, Performance,  │
     │  /webhook/razorpay/payment    │     │  Cost/Threshold, Explain,   │
     │  /razorpay/recent-orders      │     │  Razorpay Live              │
     │  X-API-Key gated (webhook:    │     └───────────────────────────┘
     │  Razorpay HMAC signature)     │
     └───────────────┬───────────────┘
                      │ HTTPS, Basic Auth (real test credentials)
                      ▼
          ┌─────────────────────────┐
          │  api.razorpay.com (real)  │
          │  Orders API                │
          └─────────────────────────┘
```

## Project structure

```
src/fraud_risk/                Core package: data, features, models, cost, evaluation, explain
src/fraud_risk/razorpay_integration/  Live Razorpay client, webhook signature verification, heuristic risk engine
configs/                        train_config.yaml, cost_config.yaml, razorpay_risk_rules.yaml (all assumptions live here)
api/                            FastAPI service (main, schemas, auth, dependencies, razorpay_router)
dashboard/                      Streamlit multipage app (incl. Razorpay Live page)
scripts/                        eda.py, download_data.ps1, run_pipeline.ps1, benchmark_latency.py,
                                 build_curated_examples.py, simulate_razorpay_webhook.py, seed_razorpay_test_orders.py
tests/                          pytest suite (schema, leakage, cost model, feature engineering, API contract, Razorpay integration)
reports/                        Generated: evaluation_report.json, figures/, test_predictions.parquet, curated_examples.json
models/champion/                Generated: serialized model bundle + metadata (gitignored except structure)
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
The API starts even without a trained model (`/score`/`/explain` return 503
until then); the Razorpay endpoints work independently and don't need one.

### 5. Run the dashboard

```
.venv\Scripts\streamlit run dashboard/app.py
```

### Razorpay integration (optional)

Set `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` (from a **test-mode**
Razorpay account — `key_id` must start with `rzp_test_`; the client refuses
to write against anything else) and `RAZORPAY_WEBHOOK_SECRET` (any string —
mirrors the secret you'd configure in the Razorpay Dashboard under
Settings > Webhooks) in `.env`. Then, with the API running:

```
python scripts/seed_razorpay_test_orders.py              # creates 3 real orders in your test account
python scripts/simulate_razorpay_webhook.py --profile high_risk
python scripts/simulate_razorpay_webhook.py --profile normal --repeat 5   # triggers the velocity rule
```

Or drive it interactively from the dashboard's **Razorpay Live** page.

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
hand-computed values, feature-engineering correctness, the Razorpay
heuristic risk engine and webhook signature verification (pure logic, no
network calls), and FastAPI contract tests (`/score`/`/explain` auto-skip
until a model has been trained; `/health` and the Razorpay webhook run
regardless, since they don't depend on the trained model).

## Defense-only scope

This system scores and explains individual transactions. It deliberately
does **not** expose: global feature importances, training data, raw model
internals, or an unauthenticated scoring oracle — `/score` and `/explain`
require an API key, and `/explain` returns only the top-6 contributing
features for the specific transaction scored, never the full feature vector
or model weights. The Razorpay webhook endpoint's authentication is the
HMAC signature itself (exactly how Razorpay's own servers would call it —
they never send our internal API key), verified against the raw request
body before anything else runs. See `MODEL_CARD.md` for the full scope
statement.
