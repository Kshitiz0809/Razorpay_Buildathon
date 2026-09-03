import json
from datetime import datetime

import pytest

from fraud_risk.razorpay_integration.risk_rules import IST, PaymentRiskEngine
from fraud_risk.razorpay_integration.webhook import sign_payload, verify_signature

CFG = {
    "currency_minor_unit_divisor": 100,
    "amount": {
        "high_amount_threshold": 50000,
        "high_amount_points": 25,
        "very_high_amount_threshold": 200000,
        "very_high_amount_points": 40,
    },
    "international_card": {"points": 20},
    "velocity": {"window_seconds": 600, "max_attempts_before_flag": 3, "points": 30},
    "new_payer": {"amount_threshold": 20000, "points": 15},
    "ip_velocity": {"window_seconds": 600, "max_attempts_before_flag": 3, "points": 35},
    "card_fingerprint_reuse": {"points": 35},
    "method_risk_points": {"card": 5, "netbanking": 3, "wallet": 8, "upi": 2, "emi": 5},
    "odd_hour": {"start_hour_ist": 1, "end_hour_ist": 5, "points": 10},
    "decision_thresholds": {"medium": 30, "high": 55},
}


def _ist_timestamp(hour: int) -> float:
    return datetime.now(IST).replace(hour=hour, minute=0, second=0, microsecond=0).timestamp()


def _payment(**overrides):
    base = {
        "amount": 10000,  # INR 100
        "email": "buyer@example.com",
        "contact": "+911111111111",
        "method": "upi",
        "created_at": _ist_timestamp(14),  # 2pm IST, outside odd-hour window
    }
    base.update(overrides)
    return base


# --- risk_rules.py -----------------------------------------------------


def test_low_amount_common_method_scores_low():
    engine = PaymentRiskEngine(CFG)
    result = engine.score(_payment(amount=10000, method="upi", email="a@example.com"))
    assert result.risk_level == "low"
    assert result.risk_score == 2  # upi method weight only


def test_very_high_amount_triggers_high_risk():
    engine = PaymentRiskEngine(CFG)
    result = engine.score(_payment(amount=25_000_000, method="upi", email="b@example.com"))  # INR 250,000
    rule_names = {r.rule for r in result.triggered_rules}
    assert "very_high_amount" in rule_names
    assert "new_payer_high_amount" in rule_names  # first-seen payer, amount above new_payer threshold
    assert result.risk_score == 40 + 15 + 2
    assert result.risk_level == "high"


def test_international_card_rule_triggers():
    engine = PaymentRiskEngine(CFG)
    result = engine.score(
        _payment(amount=10000, method="card", email="c@example.com", card={"international": True})
    )
    rule_names = {r.rule for r in result.triggered_rules}
    assert "international_card" in rule_names
    assert result.risk_score == 20 + 5  # international + card method weight


def test_velocity_rule_triggers_only_after_max_attempts():
    engine = PaymentRiskEngine(CFG)
    payer = "repeat.buyer@example.com"
    results = [engine.score(_payment(amount=1000, method="upi", email=payer)) for _ in range(4)]

    for r in results[:3]:
        assert "velocity" not in {rule.rule for rule in r.triggered_rules}
    assert "velocity" in {rule.rule for rule in results[3].triggered_rules}
    assert results[3].risk_level == "medium"


def test_new_payer_rule_fires_once_per_payer():
    engine = PaymentRiskEngine(CFG)
    payer = "big.spender@example.com"
    first = engine.score(_payment(amount=3_000_000, method="netbanking", email=payer))  # INR 30,000
    second = engine.score(_payment(amount=3_000_000, method="netbanking", email=payer))

    assert "new_payer_high_amount" in {r.rule for r in first.triggered_rules}
    assert "new_payer_high_amount" not in {r.rule for r in second.triggered_rules}


def test_ip_velocity_rule_triggers_across_different_payers():
    # The whole point of an IP-based rule: distinct emails/contacts sharing
    # one IP within the window should still trigger it, unlike the
    # per-payer velocity rule which is keyed by payer, not IP.
    engine = PaymentRiskEngine(CFG)
    shared_ip = "203.0.113.7"
    results = [
        engine.score(_payment(amount=1000, method="upi", email=f"ring{i}@example.com", notes={"ip_address": shared_ip}))
        for i in range(4)
    ]
    for r in results[:3]:
        assert "ip_velocity" not in {rule.rule for rule in r.triggered_rules}
    assert "ip_velocity" in {rule.rule for rule in results[3].triggered_rules}


def test_ip_velocity_rule_ignored_when_no_ip_present():
    engine = PaymentRiskEngine(CFG)
    for i in range(5):
        result = engine.score(_payment(amount=1000, method="upi", email=f"noip{i}@example.com"))
    assert "ip_velocity" not in {rule.rule for rule in result.triggered_rules}


def test_card_fingerprint_reuse_rule_triggers_on_second_distinct_payer():
    engine = PaymentRiskEngine(CFG)
    card = {"last4": "4242", "network": "Visa", "issuer": "HDFC"}
    first = engine.score(_payment(amount=1000, method="card", email="alice@example.com", card=card))
    second = engine.score(_payment(amount=1000, method="card", email="bob@example.com", card=card))

    assert "card_fingerprint_reuse" not in {r.rule for r in first.triggered_rules}
    assert "card_fingerprint_reuse" in {r.rule for r in second.triggered_rules}


def test_card_fingerprint_reuse_rule_ignored_for_same_payer_repeat_use():
    engine = PaymentRiskEngine(CFG)
    card = {"last4": "4242", "network": "Visa", "issuer": "HDFC"}
    first = engine.score(_payment(amount=1000, method="card", email="alice@example.com", card=card))
    second = engine.score(_payment(amount=1000, method="card", email="alice@example.com", card=card))

    assert "card_fingerprint_reuse" not in {r.rule for r in first.triggered_rules}
    assert "card_fingerprint_reuse" not in {r.rule for r in second.triggered_rules}


def test_odd_hour_rule_triggers_at_3am_ist():
    engine = PaymentRiskEngine(CFG)
    result = engine.score(_payment(amount=1000, method="upi", email="d@example.com", created_at=_ist_timestamp(3)))
    assert "odd_hour" in {r.rule for r in result.triggered_rules}


# --- webhook.py ----------------------------------------------------------


def test_valid_signature_verifies():
    body = b'{"event":"payment.authorized"}'
    secret = "test-secret"
    assert verify_signature(body, sign_payload(body, secret), secret) is True


def test_wrong_secret_fails_verification():
    body = b'{"event":"payment.authorized"}'
    sig = sign_payload(body, "correct-secret")
    assert verify_signature(body, sig, "wrong-secret") is False


def test_tampered_body_fails_verification():
    secret = "test-secret"
    sig = sign_payload(b'{"event":"payment.authorized"}', secret)
    assert verify_signature(b'{"event":"payment.captured"}', sig, secret) is False


def test_missing_signature_or_secret_fails():
    assert verify_signature(b"{}", "", "secret") is False
    assert verify_signature(b"{}", "somesig", "") is False


# --- API-level: health + webhook work without a trained fraud model -----

WEBHOOK_SECRET = "test-webhook-secret-for-pytest"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key-for-pytest")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", WEBHOOK_SECRET)
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


def _sample_event(**overrides):
    payment = {
        "id": "pay_test123",
        "amount": 10000,
        "currency": "INR",
        "method": "upi",
        "email": "buyer@example.com",
        "contact": "+911111111111",
        "created_at": _ist_timestamp(14),
    }
    payment.update(overrides)
    return {"event": "payment.authorized", "payload": {"payment": {"entity": payment}}}


def test_health_works_regardless_of_model_training_state(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_webhook_with_valid_signature_returns_risk_score(client):
    raw_body = json.dumps(_sample_event()).encode("utf-8")
    signature = sign_payload(raw_body, WEBHOOK_SECRET)
    resp = client.post(
        "/webhook/razorpay/payment",
        content=raw_body,
        headers={"Content-Type": "application/json", "X-Razorpay-Signature": signature},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["payment_id"] == "pay_test123"
    assert body["risk_level"] in {"low", "medium", "high"}
    assert isinstance(body["triggered_rules"], list)


def test_webhook_with_invalid_signature_rejected(client):
    raw_body = json.dumps(_sample_event()).encode("utf-8")
    resp = client.post(
        "/webhook/razorpay/payment",
        content=raw_body,
        headers={"Content-Type": "application/json", "X-Razorpay-Signature": "bogus"},
    )
    assert resp.status_code == 401


def test_webhook_with_missing_signature_header_rejected(client):
    raw_body = json.dumps(_sample_event()).encode("utf-8")
    resp = client.post("/webhook/razorpay/payment", content=raw_body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 401
