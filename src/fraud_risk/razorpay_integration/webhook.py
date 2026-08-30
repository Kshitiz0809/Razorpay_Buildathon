"""Razorpay webhook signature verification.

Razorpay signs each webhook POST body with HMAC-SHA256 using the webhook
secret configured in the Razorpay Dashboard (Settings > Webhooks), sent in
the `X-Razorpay-Signature` header as a hex digest. This is the exact,
publicly-documented verification algorithm -- verified against the raw
request body bytes, never a re-serialized copy, since re-serialization can
silently change byte content (key order, whitespace) and break the HMAC.
"""
import hashlib
import hmac


def verify_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    if not signature or not secret:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def sign_payload(raw_body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
