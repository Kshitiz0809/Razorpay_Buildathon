"""Shared Groq chat-completions client with 429 retry/backoff.

Groq enforces a per-organization tokens-per-minute limit that is easy to
hit with a token-hungry reasoning model like openai/gpt-oss-120b --
confirmed live during development (`Limit 8000, Used 7188, Requested
1446`), especially across a multi-step tool-calling loop (agent.py) or
repeated calls during a demo/pitch recording. A 429 here is a normal,
expected operating condition, not a bug, so it's retried with backoff
rather than surfaced as a crash on the first hit.
"""
import time

import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5


def call_groq(api_key: str, payload: dict) -> dict:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if resp.status_code == 429:
            last_error = RuntimeError(f"Groq rate limit hit after {attempt + 1} attempt(s): {resp.text}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise last_error
        if resp.status_code >= 400:
            # Surface Groq's actual error body -- a bare "400 Client Error"
            # from raise_for_status() alone was not enough to diagnose a
            # real issue found during development.
            raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")
        return resp.json()
    raise last_error  # pragma: no cover -- loop always returns or raises above
