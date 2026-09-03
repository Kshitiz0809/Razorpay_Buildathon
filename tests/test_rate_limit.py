import pytest

from api.rate_limit import RateLimiter


def test_allows_requests_under_the_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        limiter.check("key-a")  # should not raise


def test_blocks_requests_over_the_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        limiter.check("key-a")
    with pytest.raises(Exception) as exc_info:
        limiter.check("key-a")
    assert "429" in str(exc_info.value.status_code) or exc_info.value.status_code == 429


def test_limits_are_independent_per_key():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    limiter.check("key-a")
    limiter.check("key-a")
    limiter.check("key-b")  # different key, should not raise despite key-a being at its limit


def test_old_hits_outside_window_do_not_count(monkeypatch):
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    times = iter([100.0, 100.0, 200.0])  # third call is 100s later, well past the 10s window
    monkeypatch.setattr("api.rate_limit.time.monotonic", lambda: next(times))

    limiter.check("key-a")  # t=100, allowed (1st call consumed for the check)
    with pytest.raises(Exception):
        limiter.check("key-a")  # t=100 still, over the limit
    limiter.check("key-a")  # t=200, window has passed, allowed again
