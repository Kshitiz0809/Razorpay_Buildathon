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
from fastapi import Depends, FastAPI, HTTPException, Request, status

from fraud_risk.artifacts import load_champion, load_metadata, load_threshold
from fraud_risk.chargeback.agent import investigate_and_respond
from fraud_risk.chargeback.responder import draft_dispute_letter
from fraud_risk.config import CONFIGS_DIR, load_yaml
from fraud_risk.explain.shap_explainer import ShapExplainer
from fraud_risk.razorpay_integration.risk_rules import PaymentRiskEngine

from api import razorpay_router
from api.auth import require_api_key
from api.dependencies import transaction_to_dataframe
from api.rate_limit import rate_limit_chargeback, rate_limit_explain, rate_limit_investigate, rate_limit_score
from api.schemas import (
    ChargebackDraftIn,
    ChargebackDraftOut,
    ChargebackInvestigateIn,
    ChargebackInvestigateOut,
    ExplainOut,
    FeatureContribution,
    HealthOut,
    ScoreOut,
    TransactionIn,
)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The fraud-scoring model and the Razorpay integration are independently
    # deployable capabilities -- one being unavailable (e.g. the model
    # hasn't been trained yet) must not take down the other.
    try:
        app.state.champion = load_champion()
        app.state.threshold = load_threshold()
        app.state.metadata = load_metadata()
        app.state.explainer = ShapExplainer(app.state.champion)
        app.state.model_loaded = True
    except FileNotFoundError:
        app.state.champion = None
        app.state.threshold = None
        app.state.metadata = {"model_version": "not-trained-yet"}
        app.state.explainer = None
        app.state.model_loaded = False

    app.state.razorpay_risk_engine = PaymentRiskEngine(load_yaml(CONFIGS_DIR / "razorpay_risk_rules.yaml"))
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
app.include_router(razorpay_router.router)


def _require_model_loaded(request: Request) -> None:
    if not request.app.state.model_loaded:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Fraud model not trained yet -- run the training pipeline first.",
        )


@app.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    return HealthOut(status="ok", model_version=request.app.state.metadata["model_version"])


@app.post(
    "/score",
    response_model=ScoreOut,
    dependencies=[Depends(require_api_key), Depends(rate_limit_score)],
)
def score(txn: TransactionIn, request: Request) -> ScoreOut:
    _require_model_loaded(request)
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


@app.post(
    "/explain",
    response_model=ExplainOut,
    dependencies=[Depends(require_api_key), Depends(rate_limit_explain)],
)
def explain(txn: TransactionIn, request: Request) -> ExplainOut:
    _require_model_loaded(request)
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


@app.post(
    "/chargeback/draft-response",
    response_model=ChargebackDraftOut,
    dependencies=[Depends(require_api_key), Depends(rate_limit_chargeback)],
)
def chargeback_draft_response(payload: ChargebackDraftIn, request: Request) -> ChargebackDraftOut:
    _require_model_loaded(request)
    state = request.app.state
    df = transaction_to_dataframe(payload.transaction)
    proba = float(state.champion.predict_proba(df)[0, 1])

    if proba >= state.threshold:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Refusing to draft a dispute letter: this transaction was scored as high-risk "
            f"by our own fraud model (fraud_probability={proba:.4f} >= threshold "
            f"{state.threshold:.4f}). This tool only assists disputes for transactions the "
            "system itself believes are legitimate -- the real 'friendly fraud' chargeback-"
            "defense use case. Drafting a defense for a system-flagged transaction would help "
            "evade fraud detection, which this project will not do.",
        )

    contributions = state.explainer.explain_one(df)
    letter = draft_dispute_letter(
        transaction_id=payload.transaction.transaction_id or "N/A",
        amount=payload.transaction.amount,
        currency=payload.currency,
        transaction_date=payload.transaction_date,
        top_contributors=[vars(c) for c in contributions],
        merchant_name=payload.merchant_name,
    )
    return ChargebackDraftOut(
        transaction_id=payload.transaction.transaction_id,
        fraud_probability=proba,
        threshold_used=state.threshold,
        letter=letter,
        model_version=state.metadata["model_version"],
    )


@app.post(
    "/chargeback/investigate",
    response_model=ChargebackInvestigateOut,
    dependencies=[Depends(require_api_key), Depends(rate_limit_investigate)],
)
def chargeback_investigate(payload: ChargebackInvestigateIn, request: Request) -> ChargebackInvestigateOut:
    """Agentic investigation: the LLM calls read-only tools (IP reputation,
    shipping status, prior chargeback history) and, if `allow_submit` is
    true, may call a mock evidence-submission tool as its final action.

    The high-risk refusal below is a deterministic code check, run before
    the agent loop even starts -- identical in spirit to
    /chargeback/draft-response's guard. The LLM decides how to investigate
    and whether to submit; it never decides whether it's allowed to run.
    """
    _require_model_loaded(request)
    state = request.app.state
    df = transaction_to_dataframe(payload.transaction)
    proba = float(state.champion.predict_proba(df)[0, 1])

    if proba >= state.threshold:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Refusing to investigate: this transaction was scored as high-risk by our own "
            f"fraud model (fraud_probability={proba:.4f} >= threshold {state.threshold:.4f}). "
            "This tool only assists disputes for transactions the system itself believes are "
            "legitimate. Investigating or submitting evidence for a system-flagged transaction "
            "would help evade fraud detection, which this project will not do.",
        )

    contributions = state.explainer.explain_one(df)
    result = investigate_and_respond(
        transaction_id=payload.transaction.transaction_id or "N/A",
        amount=payload.transaction.amount,
        currency=payload.currency,
        transaction_date=payload.transaction_date,
        merchant_name=payload.merchant_name,
        top_contributors=[vars(c) for c in contributions],
        customer_email=payload.customer_email,
        customer_ip=payload.customer_ip,
        tracking_number=payload.tracking_number,
        allow_submit=payload.allow_submit,
    )
    return ChargebackInvestigateOut(
        transaction_id=payload.transaction.transaction_id,
        fraud_probability=proba,
        threshold_used=state.threshold,
        investigation_trace=result["trace"],
        conclusion=result["conclusion"],
        submitted=result["submitted"],
        submission_reference=result["submission_reference"],
        model_version=state.metadata["model_version"],
    )
