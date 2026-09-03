from unittest.mock import MagicMock

import pytest

from fraud_risk.chargeback import groq_client


def _fake_response(status_code, json_body=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_body or {}
    return resp


def test_succeeds_immediately_on_200(monkeypatch):
    ok_response = _fake_response(200, {"choices": [{"message": {"content": "hi"}}]})
    monkeypatch.setattr(groq_client.requests, "post", MagicMock(return_value=ok_response))
    monkeypatch.setattr(groq_client.time, "sleep", MagicMock())

    result = groq_client.call_groq("fake-key", {"model": "x"})
    assert result["choices"][0]["message"]["content"] == "hi"
    assert groq_client.requests.post.call_count == 1


def test_retries_on_429_then_succeeds(monkeypatch):
    rate_limited = _fake_response(429, text="rate limited")
    ok_response = _fake_response(200, {"choices": [{"message": {"content": "ok after retry"}}]})
    post_mock = MagicMock(side_effect=[rate_limited, ok_response])
    monkeypatch.setattr(groq_client.requests, "post", post_mock)
    sleep_mock = MagicMock()
    monkeypatch.setattr(groq_client.time, "sleep", sleep_mock)

    result = groq_client.call_groq("fake-key", {"model": "x"})
    assert result["choices"][0]["message"]["content"] == "ok after retry"
    assert post_mock.call_count == 2
    sleep_mock.assert_called_once()  # backed off exactly once before the successful retry


def test_raises_after_exhausting_retries_on_persistent_429(monkeypatch):
    always_limited = _fake_response(429, text="still rate limited")
    monkeypatch.setattr(groq_client.requests, "post", MagicMock(return_value=always_limited))
    monkeypatch.setattr(groq_client.time, "sleep", MagicMock())

    with pytest.raises(RuntimeError, match="rate limit"):
        groq_client.call_groq("fake-key", {"model": "x"})
    assert groq_client.requests.post.call_count == groq_client.MAX_RETRIES


def test_non_retryable_error_surfaces_response_body_immediately(monkeypatch):
    # A genuine 400 is not retried (retrying a malformed request doesn't
    # help), and the raised error must carry the actual response body --
    # a bare "400 Client Error" with no body was not enough to diagnose a
    # real issue found during development (see CHALLENGES.md).
    bad_request = _fake_response(400, text='{"error": {"message": "invalid tool schema"}}')
    post_mock = MagicMock(return_value=bad_request)
    monkeypatch.setattr(groq_client.requests, "post", post_mock)
    monkeypatch.setattr(groq_client.time, "sleep", MagicMock())

    with pytest.raises(RuntimeError, match="invalid tool schema"):
        groq_client.call_groq("fake-key", {"model": "x"})
    assert post_mock.call_count == 1  # no retry for a non-429 error
