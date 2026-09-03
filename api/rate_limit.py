"""Simple in-memory per-API-key sliding-window rate limiter.

Deliberately lightweight (no Redis) for this prototype -- a real
multi-instance deployment would back this with a shared store. Exists
specifically so /explain (which reveals per-transaction feature
contributions) can't be hammered to probe the model's decision boundary at
scale -- a defense-only requirement, not a performance feature. /score and
/chargeback/draft-response get looser limits for the same reason at lower
severity.
"""
import time
from collections import defaultdict
from threading import Lock

from fastapi import Header, HTTPException, status


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            recent = [t for t in self._hits[key] if now - t <= self.window_seconds]
            if len(recent) >= self.max_requests:
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    f"Rate limit exceeded: max {self.max_requests} requests per {self.window_seconds:.0f}s.",
                )
            recent.append(now)
            self._hits[key] = recent


score_limiter = RateLimiter(max_requests=120, window_seconds=60)
explain_limiter = RateLimiter(max_requests=20, window_seconds=60)
chargeback_limiter = RateLimiter(max_requests=10, window_seconds=60)


def rate_limit_score(x_api_key: str | None = Header(default=None)) -> None:
    score_limiter.check(x_api_key or "anonymous")


def rate_limit_explain(x_api_key: str | None = Header(default=None)) -> None:
    explain_limiter.check(x_api_key or "anonymous")


def rate_limit_chargeback(x_api_key: str | None = Header(default=None)) -> None:
    chargeback_limiter.check(x_api_key or "anonymous")
