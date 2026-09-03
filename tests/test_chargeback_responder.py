"""Tests the /chargeback/draft-response safety guard: it must refuse to
draft a dispute letter for any transaction the fraud model itself scored
as high-risk. Uses a mocked model + mocked LLM call so this runs fast,
deterministically, and without needing real trained artifacts or a real
Groq API call -- the guard logic itself is what's under test here.
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
    body = {"transaction": txn, "currency": "USD", "transaction_date": "2026-08-15", "merchant_name": "Acme"}
    body.update(overrides)
    return body


def test_refuses_to_draft_for_high_risk_transaction(client):
    from api.main import app

    app.state.champion.predict_proba.return_value = np.array([[0.1, 0.9]])  # proba 0.9 >= threshold 0.5

    resp = client.post("/chargeback/draft-response", json=_payload(), headers={"X-API-Key": API_KEY})
    assert resp.status_code == 400
    assert "high-risk" in resp.json()["detail"]


def test_drafts_letter_for_low_risk_transaction(client, monkeypatch):
    from api.main import app

    app.state.champion.predict_proba.return_value = np.array([[0.9, 0.1]])  # proba 0.1 < threshold 0.5
    monkeypatch.setattr("api.main.draft_dispute_letter", lambda **kwargs: "DRAFTED LETTER TEXT")

    resp = client.post("/chargeback/draft-response", json=_payload(), headers={"X-API-Key": API_KEY})
    assert resp.status_code == 200
    body = resp.json()
    assert body["letter"] == "DRAFTED LETTER TEXT"
    assert body["fraud_probability"] == pytest.approx(0.1)


def test_boundary_at_exact_threshold_is_treated_as_high_risk(client):
    from api.main import app

    app.state.champion.predict_proba.return_value = np.array([[0.5, 0.5]])  # proba == threshold exactly

    resp = client.post("/chargeback/draft-response", json=_payload(), headers={"X-API-Key": API_KEY})
    assert resp.status_code == 400


def test_requires_api_key(client):
    resp = client.post("/chargeback/draft-response", json=_payload())
    assert resp.status_code == 401
