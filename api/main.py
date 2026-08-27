"""Fraud Risk Scoring API.

Defense-only: this service scores and explains individual transactions. It
never exposes global feature importances, training data, or model
internals, and /score + /explain sit behind an API key -- deliberate design
choices so nothing here becomes a tool for reverse-engineering or evading
the model.

Run:  uvicorn api.main:app --reload
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request

from fraud_risk.artifacts import load_champion, load_metadata, load_threshold
from fraud_risk.explain.shap_explainer import ShapExplainer

from api.auth import require_api_key
from api.dependencies import transaction_to_dataframe
from api.schemas import ExplainOut, FeatureContribution, HealthOut, ScoreOut, TransactionIn

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.champion = load_champion()
    app.state.threshold = load_threshold()
    app.state.metadata = load_metadata()
    app.state.explainer = ShapExplainer(app.state.champion)
    yield


app = FastAPI(
    title="Fraud Risk Scoring API",
    description=(
        "Real-time transaction fraud-risk scoring for a payment gateway. "
        "Defense-only: scoring and per-transaction explanation, no evasion capability."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    return HealthOut(status="ok", model_version=request.app.state.metadata["model_version"])


@app.post("/score", response_model=ScoreOut, dependencies=[Depends(require_api_key)])
def score(txn: TransactionIn, request: Request) -> ScoreOut:
    state = request.app.state
    df = transaction_to_dataframe(txn)
    proba = float(state.champion.predict_proba(df)[0, 1])
    return ScoreOut(
        transaction_id=txn.transaction_id,
        fraud_probability=proba,
        is_flagged=proba >= state.threshold,
        threshold_used=state.threshold,
        model_version=state.metadata["model_version"],
        scored_at=datetime.now(timezone.utc),
    )


@app.post("/explain", response_model=ExplainOut, dependencies=[Depends(require_api_key)])
def explain(txn: TransactionIn, request: Request) -> ExplainOut:
    state = request.app.state
    df = transaction_to_dataframe(txn)
    proba = float(state.champion.predict_proba(df)[0, 1])
    contributions = state.explainer.explain_one(df)
    return ExplainOut(
        transaction_id=txn.transaction_id,
        fraud_probability=proba,
        is_flagged=proba >= state.threshold,
        threshold_used=state.threshold,
        model_version=state.metadata["model_version"],
        scored_at=datetime.now(timezone.utc),
        top_contributors=[FeatureContribution(**vars(c)) for c in contributions],
    )
