"""Live Razorpay integration endpoints.

Demonstrates the actual production wiring pattern: a signature-verified
webhook receiver for real-time payment risk decisions (auth here is the
HMAC signature itself, exactly as Razorpay's own servers would call this
endpoint -- it deliberately does NOT also require our internal X-API-Key,
since real Razorpay webhook calls never carry one), plus a read-only proxy
for the dashboard so the Razorpay secret never has to leave this process.
"""
import json
import os

from fastapi import APIRouter, Depends, HTTPException, Request, status

from fraud_risk.razorpay_integration.client import RazorpayClient
from fraud_risk.razorpay_integration.webhook import verify_signature

from api.auth import require_api_key

router = APIRouter(tags=["razorpay"])


@router.post("/webhook/razorpay/payment")
async def razorpay_payment_webhook(request: Request):
    raw_body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")
    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")

    if not verify_signature(raw_body, signature, secret):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid webhook signature")

    event = json.loads(raw_body)
    payment = event.get("payload", {}).get("payment", {}).get("entity", {})
    if not payment:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No payment entity in webhook payload")

    result = request.app.state.razorpay_risk_engine.score(payment)
    return {
        "payment_id": payment.get("id"),
        "event": event.get("event"),
        "risk_score": result.risk_score,
        "risk_level": result.risk_level,
        "triggered_rules": [vars(r) for r in result.triggered_rules],
    }


@router.get("/razorpay/recent-orders", dependencies=[Depends(require_api_key)])
def recent_orders(count: int = 20):
    client = RazorpayClient()
    return {"orders": client.list_orders(count=count), "test_mode": client.is_test_mode}
