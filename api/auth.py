import os

from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Shared-secret gate on /score and /explain.

    A prototype stand-in for OAuth2/mTLS in a real deployment -- but present
    from day one so nothing in this service is a public unauthenticated
    scoring or explanation oracle an attacker could probe to reverse-engineer
    the model.
    """
    expected = os.environ.get("API_KEY")
    if not expected:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "API_KEY is not configured on the server"
        )
    if x_api_key != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")
