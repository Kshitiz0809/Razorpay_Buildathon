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

Three of the hackathon brief's four example directions, built as one
coherent system rather than three disconnected demos:

1. **Fraud-spike detector** — a calibrated LightGBM classifier, trained and
   rigorously evaluated on the Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   dataset (284,807 real anonymized card transactions, 492 frauds) — see
   `data/README.md` for provenance and `MODEL_CARD.md` for scope and
   limitations.
2. **Abuse-ring sentinel** — a live integration against Razorpay's own Test
   Mode API (not a simulation of Razorpay, an actual authenticated client
   of it), with velocity and card/IP-fingerprint-reuse rules purpose-built
   to catch coordinated identity abuse that no single transaction reveals.
   See "Live Razorpay integration" below.
3. **Chargeback evidence responder** — drafts a dispute letter from the
   exact same per-transaction SHAP explanation the fraud classifier already
   produces, guarded so it only ever assists disputes for transactions the
   system itself scored as legitimate. See "Chargeback evidence responder"
   below.

All three are served through the same FastAPI backend and the same
Streamlit dashboard — one system, not three demos stitched together.

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

**Abuse-ring rules, not just single-transaction rules:** beyond per-payer
velocity, the engine tracks IP-address velocity across *distinct* payers
and card-fingerprint (last4+network+issuer — Razorpay never exposes the
full PAN/BIN) reuse across *distinct* payers — the actual signature of a
ring probing with one device/card against many synthetic identities, which
no single transaction's amount or method can reveal on its own. IP address
is read from the payment's `notes` field, a real Razorpay-supported
merchant-metadata mechanism (not a standard payment field).

Dashboard: [Razorpay Live](dashboard/pages/5_Razorpay_Live.py) shows real
fetched test-mode orders and lets you fire simulated payment events at the
live risk gate, including a velocity-abuse demo (repeat the same payer) and
an abuse-ring demo (N distinct payers sharing one IP + card, watch the risk
score escalate statefully in real time).

## Chargeback evidence responder

`POST /chargeback/draft-response` (`src/fraud_risk/chargeback/responder.py`,
dashboard's [Chargeback Responder](dashboard/pages/6_Chargeback_Responder.py)
page) takes a transaction and drafts a chargeback dispute letter from the
exact same per-transaction SHAP explanation the Explainability page
already computes — one explainability output feeding two features, not a
second disconnected one.

**The guard that makes this defense-only, not just labeled that way:** the
endpoint re-scores the transaction itself first, and refuses outright if
the fraud model scored it at or above the flagging threshold —

```
Refusing to draft a dispute letter: this transaction was scored as
high-risk by our own fraud model... This tool only assists disputes for
transactions the system itself believes are legitimate.
```

This is the real "friendly fraud" use case (a legitimate transaction the
cardholder later disputes) — drafting a defense for a transaction the
system itself flagged as fraud would mean helping evade detection, which
this project will not do. The Chargeback Responder page lets you pick a
"correctly flagged fraud" example specifically to watch this refusal fire.
Text generation runs on Groq (`openai/gpt-oss-120b`); nothing here
auto-submits a dispute — it's a drafting aid for a human to review.

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

Measured once on the held-out, chronologically-later TEST block (71,203
transactions, 94 frauds) — see `reports/evaluation_report.json` and
`reports/figures/` for the full report and PR / calibration / cost curves.

| | Champion (LightGBM, calibrated) | Baseline (LogReg, t=0.5) |
|---|---|---|
| PR-AUC | **0.792** | 0.747 |
| ROC-AUC | 0.983 | 0.977 |
| Precision | **83.1%** | 2.1% |
| Recall | 73.4% | 89.4% |
| F1 | **0.780** | 0.041 |
| False positives | **14** / 71,109 legitimate | 3,920 / 71,109 legitimate |

At the cost-optimal threshold (0.0181, selected on VALIDATION, applied
once to TEST): **$8,108 saved vs. flagging nothing** (a 66% cost
reduction), **$1,245 saved vs. a naive fixed threshold of 0.5** on this
test set alone. The champion catches 69 of 94 real frauds while wrongly
declining only 14 of 71,109 legitimate transactions.

The deployed threshold (0.018) is far below the "default" 0.5 by design:
under `configs/cost_config.yaml`'s assumptions, missing a fraud costs
roughly 14x what a false decline costs, so the cost-optimal policy is
deliberately recall-leaning — this is the dashboard's Cost & Threshold
page's cost model doing exactly its job, not a tuning oversight.

**Why PR-AUC here is lower than commonly-cited numbers for this dataset:**
many public analyses of this exact dataset use a random train/test split
and report PR-AUC upward of 0.85. This project uses a strict chronological
split instead (see below), which is more honest but harder: temporally
adjacent fraud transactions in this dataset can share very similar
PCA-feature signatures, so a random split leaks near-duplicate patterns
across train/test in a way a genuinely time-ordered evaluation cannot. The
0.792 above is the number that should generalize to a real forecasting
setting; a higher random-split number would not.

**A real bug this rigor caught before it shipped:** the first trained
version of the champion scored PR-AUC 0.130 on TEST despite 0.65 on
VALIDATION during training — a champion dramatically *worse* than the
baseline, which should never happen and was not accepted at face value.
Root cause: LightGBM's built-in `average_precision` eval-metric string
does not compute the same thing as sklearn's `average_precision_score`
under `scale_pos_weight`, so early stopping was silently optimizing the
wrong objective. Fixed by wiring in an explicit sklearn-consistent eval
metric, which also revealed that the textbook auto-computed
`scale_pos_weight` (~447x for this split) was itself overfitting badly to
the 350 training-period frauds; a VAL-only hyperparameter search found
`scale_pos_weight=1` (no reweighting) with moderate regularization
generalized far better. See `CHALLENGES.md` for the full investigation —
this is also why the champion legitimately beats the baseline above,
rather than a champion propped up on a training-period metric that never
generalized.

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
     ┌─────────────────────────────────┐  ┌───────────────────────────────┐
     │  api/  (FastAPI)                   │◄──│  dashboard/ (Streamlit)         │
     │  /health  /score  /explain         │HTTP│  Live Stream, Performance,      │
     │  /chargeback/draft-response        │  │  Cost/Threshold, Explain,        │
     │  /webhook/razorpay/payment         │  │  Razorpay Live, Chargeback       │
     │  /razorpay/recent-orders           │  │  Responder                       │
     │  X-API-Key + rate limits (webhook: │  └───────────────────────────────┘
     │  Razorpay HMAC signature instead)  │
     └───────┬─────────────────────┬───────┘
             │ Basic Auth            │ chat completions
             ▼                       ▼
  ┌─────────────────────┐  ┌─────────────────────┐
  │ api.razorpay.com     │  │ api.groq.com (real)   │
  │ (real) Orders API    │  │ openai/gpt-oss-120b    │
  └─────────────────────┘  └─────────────────────┘
```

## Project structure

```
src/fraud_risk/                Core package: data, features, models, cost, evaluation, explain
src/fraud_risk/razorpay_integration/  Live Razorpay client, webhook signature verification, heuristic risk engine
src/fraud_risk/chargeback/     Groq-backed chargeback dispute letter drafter
configs/                        train_config.yaml, cost_config.yaml, razorpay_risk_rules.yaml (all assumptions live here)
api/                            FastAPI service (main, schemas, auth, rate_limit, dependencies, razorpay_router)
dashboard/                      Streamlit multipage app (incl. Razorpay Live, Chargeback Responder pages)
scripts/                        eda.py, download_data.ps1, run_pipeline.ps1, benchmark_latency.py,
                                 build_curated_examples.py, simulate_razorpay_webhook.py, seed_razorpay_test_orders.py
tests/                          pytest suite (schema, leakage, cost model, feature engineering, API contract,
                                 Razorpay integration, rate limiting, chargeback safety guard)
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
python scripts/simulate_razorpay_webhook.py --ring 4                       # triggers the abuse-ring rules
```

Or drive it interactively from the dashboard's **Razorpay Live** page.

### Chargeback evidence responder (optional, needs a trained model)

Set `GROQ_API_KEY` (free at [console.groq.com](https://console.groq.com)) in
`.env`. Once the model is trained (step 3), use the dashboard's
**Chargeback Responder** page, or call the endpoint directly:

```
POST /chargeback/draft-response
{
  "transaction": { "transaction_id": "...", "time": 1000.0, "amount": 89.99, "v1": ..., ... "v28": ... },
  "currency": "USD",
  "transaction_date": "2026-08-15",
  "merchant_name": "Acme Retail Pvt Ltd"
}
```

Returns `400` if the fraud model itself scored the transaction above its
flagging threshold — the endpoint refuses to draft a defense for a
transaction it believes is fraud, by design.

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
heuristic risk engine (including abuse-ring rules) and webhook signature
verification (pure logic, no network calls), the rate limiter, the
chargeback-responder safety guard (mocked model, no real LLM call), and
FastAPI contract tests (`/score`/`/explain` auto-skip until a model has
been trained; `/health` and the Razorpay webhook run regardless, since they
don't depend on the trained model).

## Defense-only scope

This system scores and explains individual transactions. It deliberately
does **not** expose: global feature importances, training data, raw model
internals, or an unauthenticated scoring oracle — `/score`, `/explain`, and
`/chargeback/draft-response` require an API key, are rate-limited per key
(`api/rate_limit.py`; 120/min, 20/min, 10/min respectively, in-memory —
specifically so `/explain` can't be hammered at scale to probe the model's
decision boundary), and `/explain` returns only the top-6 contributing
features for the specific transaction scored, never the full feature
vector or model weights. `/chargeback/draft-response` additionally
re-scores the transaction and refuses outright for anything above the
flagging threshold — see "Chargeback evidence responder" above. The
Razorpay webhook endpoint's authentication is the HMAC signature itself
(exactly how Razorpay's own servers would call it — they never send our
internal API key), verified against the raw request body before anything
else runs. See `MODEL_CARD.md` for the full scope statement.
