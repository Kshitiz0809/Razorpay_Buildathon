import json

import pytest
import requests

from fraud_risk.chargeback import mock_submission, tools


def test_shipping_status_is_deterministic():
    first = tools.check_shipping_status("TRACK123")
    second = tools.check_shipping_status("TRACK123")
    assert first == second


def test_shipping_status_returns_a_known_status():
    result = tools.check_shipping_status("ANY_TRACKING_NUMBER")
    assert result["status"] in {"delivered", "in_transit", "out_for_delivery", "delivery_exception"}
    assert result["delivered"] == (result["status"] == "delivered")


def test_prior_chargebacks_known_email_matches_seed_data():
    result = tools.check_prior_chargebacks("repeat.disputer@example.com")
    assert result["prior_chargebacks"] == 4


def test_prior_chargebacks_unknown_email_returns_clean_default():
    result = tools.check_prior_chargebacks("never.seen.before@example.com")
    assert result["prior_chargebacks"] == 0
    assert result["account_age_days"] is None


def test_ip_reputation_falls_back_when_network_unreachable(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.exceptions.ConnectionError("simulated network failure")

    monkeypatch.setattr(tools.requests, "get", _raise)
    result = tools.check_ip_reputation("1.2.3.4")
    assert result["source"].startswith("local_heuristic_fallback")
    assert result["classification"] in {"datacenter_or_hosting", "residential_or_business"}


def test_tool_schemas_are_valid_openai_function_format():
    for schema in tools.TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn and "description" in fn
        assert fn["name"] in tools.TOOL_FUNCTIONS
        assert fn["parameters"]["type"] == "object"
        assert set(fn["parameters"]["required"]).issubset(fn["parameters"]["properties"].keys())


def test_submit_dispute_evidence_appends_to_log(tmp_path, monkeypatch):
    log_path = tmp_path / "submitted_disputes.json"
    monkeypatch.setattr(mock_submission, "SUBMISSIONS_LOG_PATH", log_path)

    result = mock_submission.submit_dispute_evidence("txn_abc", "Dear Sir/Madam...")
    assert result["submitted"] is True
    assert result["confirmation_id"].startswith("mock_dispute_")

    logged = json.loads(log_path.read_text())
    assert len(logged) == 1
    assert logged[0]["transaction_id"] == "txn_abc"

    # A second submission appends rather than overwrites.
    mock_submission.submit_dispute_evidence("txn_def", "Another letter...")
    logged_again = json.loads(log_path.read_text())
    assert len(logged_again) == 2


def test_submit_tool_schema_matches_openai_format():
    schema = mock_submission.SUBMIT_TOOL_SCHEMA
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "submit_dispute_evidence"
    assert "letter" in schema["function"]["parameters"]["properties"]
