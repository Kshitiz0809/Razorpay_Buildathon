import os

import pytest
from fastapi.testclient import TestClient

from fraud_risk.config import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "metadata.json").exists(),
    reason="requires trained artifacts under models/champion/ (run the training pipeline first)",
)

TEST_API_KEY = "test-key-for-pytest"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", TEST_API_KEY)
    from api.main import app  # imported lazily so the env var is set first

    with TestClient(app) as c:
        yield c


def _sample_payload(**overrides):
    payload = {"transaction_id": "txn_test_1", "time": 1000.0, "amount": 149.99}
    payload.update({f"v{i}": 0.1 * i for i in range(1, 29)})
    payload.update(overrides)
    return payload


def test_health_is_open_and_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_score_without_api_key_is_unauthorized(client):
    resp = client.post("/score", json=_sample_payload())
    assert resp.status_code == 401


def test_score_with_valid_key_returns_probability(client):
    resp = client.post("/score", json=_sample_payload(), headers={"X-API-Key": TEST_API_KEY})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_flagged"], bool)


def test_score_rejects_malformed_payload(client):
    bad_payload = _sample_payload()
    del bad_payload["v1"]  # missing required field
    resp = client.post("/score", json=bad_payload, headers={"X-API-Key": TEST_API_KEY})
    assert resp.status_code == 422


def test_score_rejects_negative_amount(client):
    resp = client.post(
        "/score", json=_sample_payload(amount=-5.0), headers={"X-API-Key": TEST_API_KEY}
    )
    assert resp.status_code == 422


def test_explain_returns_top_contributors(client):
    resp = client.post("/explain", json=_sample_payload(), headers={"X-API-Key": TEST_API_KEY})
    assert resp.status_code == 200
    body = resp.json()
    assert 1 <= len(body["top_contributors"]) <= 6
    for c in body["top_contributors"]:
        assert c["direction"] in {"increases_risk", "decreases_risk", "neutral"}
