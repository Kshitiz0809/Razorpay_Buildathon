"""Builds realistic Razorpay `payment.authorized` webhook payloads.

Razorpay deliberately does not allow a payment to be completed server-side
without the client Checkout flow (PCI-DSS scope reduction) -- so a live
demo of the risk-scoring path uses simulated webhook events shaped exactly
like the real ones Razorpay sends, signed exactly the way Razorpay signs
them (see webhook.py). This is the same real event structure a production
integration would receive; only the source (this script vs. Razorpay's
servers) differs.
"""
import time
import uuid
from datetime import datetime

from fraud_risk.razorpay_integration.risk_rules import IST

PROFILES = {
    "normal": dict(
        amount=150000,  # paise -> INR 1,500
        method="upi",
        email="regular.customer@example.com",
        contact="+919812345670",
        international=False,
        hour_ist=14,
    ),
    "high_risk": dict(
        amount=25000000,  # paise -> INR 250,000
        method="card",
        email="new.suspicious@example.com",
        contact="+911234500000",
        international=True,
        hour_ist=3,
    ),
}


def build_payload(profile: dict) -> dict:
    now_ist = datetime.now(IST).replace(hour=profile["hour_ist"], minute=0, second=0, microsecond=0)
    created_at = int(now_ist.timestamp())

    payment = {
        "id": f"pay_sim_{uuid.uuid4().hex[:14]}",
        "entity": "payment",
        "amount": profile["amount"],
        "currency": "INR",
        "status": "authorized",
        "method": profile["method"],
        "captured": False,
        "email": profile["email"],
        "contact": profile["contact"],
        "created_at": created_at,
    }
    if profile["method"] == "card":
        payment["card"] = {
            "network": "Visa",
            "type": "credit",
            "international": profile["international"],
            "last4": "1111",
        }

    return {
        "entity": "event",
        "event": "payment.authorized",
        "contains": ["payment"],
        "payload": {"payment": {"entity": payment}},
        "created_at": int(time.time()),
    }
