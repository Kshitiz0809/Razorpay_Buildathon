"""Mock chargeback evidence submission.

Simulates what a real evidence-submission call to a payment gateway's
Disputes API would look like -- deliberately local and inert (appends to a
local JSON log; never calls any real external system, and specifically
never Razorpay's real API). Real production evidence submission has real
stakes -- see README.md for why a genuine deployment should keep a human
approval step ahead of this, even though this prototype demonstrates the
technical pattern end to end.
"""
import json
import time
import uuid

from fraud_risk.config import ROOT

SUBMISSIONS_LOG_PATH = ROOT / "data" / "mock_services" / "submitted_disputes.json"


def submit_dispute_evidence(transaction_id: str, letter: str) -> dict:
    SUBMISSIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if SUBMISSIONS_LOG_PATH.exists():
        existing = json.loads(SUBMISSIONS_LOG_PATH.read_text())

    confirmation_id = f"mock_dispute_{uuid.uuid4().hex[:12]}"
    existing.append(
        {
            "confirmation_id": confirmation_id,
            "transaction_id": transaction_id,
            "letter": letter,
            "submitted_at": time.time(),
        }
    )
    SUBMISSIONS_LOG_PATH.write_text(json.dumps(existing, indent=2))

    return {
        "submitted": True,
        "confirmation_id": confirmation_id,
        "note": "Simulated submission -- logged locally, not sent to any real external system.",
    }


SUBMIT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_dispute_evidence",
        "description": (
            "Submit the compiled dispute evidence letter to contest the chargeback. Only call "
            "this once you have gathered enough evidence to support that this transaction was "
            "legitimate. This is a real, final action -- it cannot be undone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "letter": {"type": "string", "description": "The final dispute evidence letter to submit"},
            },
            "required": ["letter"],
        },
    },
}
