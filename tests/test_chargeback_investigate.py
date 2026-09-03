"""Tests the /chargeback/investigate safety guard -- must refuse to run the
agent at all on a transaction the fraud model itself scored as high-risk,
exactly like /chargeback/draft-response. Uses a mocked model and a mocked
agent call so this runs fast, deterministically, and without a real Groq
call or trained artifacts.
"""
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

API_KEY = "test-key-for-pytest"


@dataclass
class _FakeContribution:
    feature_name: str
    shap_value: float
    feature_value: float
    direction: str


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", API_KEY)
    from api.main import app

    with TestClient(app) as c:
        app.state.model_loaded = True
        app.state.threshold = 0.5
        app.state.metadata = {"model_version": "test-version"}
        app.state.champion = MagicMock()
        app.state.explainer = MagicMock()
        app.state.explainer.explain_one.return_value = [
            _FakeContribution("Amount", -0.3, 50.0, "decreases_risk"),
        ]
        yield c


def _payload(**overrides):
    txn = {"transaction_id": "txn_1", "time": 1000.0, "amount": 50.0}
    txn.update({f"v{i}": 0.0 for i in range(1, 29)})
    body = {
        "transaction": txn,
        "currency": "USD",
        "transaction_date": "2026-08-15",
        "merchant_name": "Acme",
        "customer_email": "buyer@example.com",
    }
    body.update(overrides)
    return body


def test_refuses_to_investigate_high_risk_transaction(client):
    from api.main import app

    app.state.champion.predict_proba.return_value = np.array([[0.1, 0.9]])  # proba 0.9 >= threshold 0.5

    resp = client.post("/chargeback/investigate", json=_payload(), headers={"X-API-Key": API_KEY})
    assert resp.status_code == 400
    assert "high-risk" in resp.json()["detail"]


def test_investigates_low_risk_transaction_without_calling_real_agent(client, monkeypatch):
    from api.main import app

    app.state.champion.predict_proba.return_value = np.array([[0.9, 0.1]])  # proba 0.1 < threshold 0.5

    fake_result = {
        "trace": [{"tool": "check_prior_chargebacks", "arguments": {"email": "buyer@example.com"}, "result": {"prior_chargebacks": 0}}],
        "conclusion": "Evidence supports the transaction being legitimate.",
        "submitted": False,
        "submission_reference": None,
    }
    monkeypatch.setattr("api.main.investigate_and_respond", lambda **kwargs: fake_result)

    resp = client.post("/chargeback/investigate", json=_payload(), headers={"X-API-Key": API_KEY})
    assert resp.status_code == 200
    body = resp.json()
    assert body["conclusion"] == fake_result["conclusion"]
    assert body["submitted"] is False
    assert len(body["investigation_trace"]) == 1
    assert body["investigation_trace"][0]["tool"] == "check_prior_chargebacks"


def test_boundary_at_exact_threshold_is_high_risk(client):
    from api.main import app

    app.state.champion.predict_proba.return_value = np.array([[0.5, 0.5]])  # proba == threshold exactly

    resp = client.post("/chargeback/investigate", json=_payload(), headers={"X-API-Key": API_KEY})
    assert resp.status_code == 400


def test_requires_api_key(client):
    resp = client.post("/chargeback/investigate", json=_payload())
    assert resp.status_code == 401


def test_allow_submit_defaults_to_false(client, monkeypatch):
    from api.main import app

    app.state.champion.predict_proba.return_value = np.array([[0.9, 0.1]])

    captured = {}

    def fake_investigate(**kwargs):
        captured.update(kwargs)
        return {"trace": [], "conclusion": "ok", "submitted": False, "submission_reference": None}

    monkeypatch.setattr("api.main.investigate_and_respond", fake_investigate)

    client.post("/chargeback/investigate", json=_payload(), headers={"X-API-Key": API_KEY})
    assert captured["allow_submit"] is False
