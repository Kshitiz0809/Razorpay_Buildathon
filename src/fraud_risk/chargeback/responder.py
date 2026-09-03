"""Chargeback evidence responder.

Given a transaction's fraud-risk explanation (the same per-transaction SHAP
output already generated for the Explainability page) plus basic
transaction facts, drafts a structured chargeback dispute/evidence letter a
merchant can submit to a card network -- directly implementing the
hackathon brief's "chargeback evidence responder" direction from the same
explainability output already built, rather than a second, disconnected
feature.

This is a drafting aid for a human to review, not an autonomous submission
system -- nothing here auto-submits a dispute. The caller (api/main.py) is
responsible for only invoking this on transactions the fraud model itself
scored as LOW risk (the real "friendly fraud" chargeback-defense use case);
drafting a defense for a transaction the system believes IS fraud would
help evade detection, which this project will not do -- see the guard in
api/main.py's /chargeback/draft-response endpoint.
"""
import os

import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def _format_contributors(top_contributors: list[dict]) -> str:
    lines = []
    for c in top_contributors:
        direction = c["direction"].replace("_", " ")
        lines.append(f"- {c['feature_name']}: {direction} (signed contribution {c['shap_value']:+.3f})")
    return "\n".join(lines)


def draft_dispute_letter(
    transaction_id: str,
    amount: float,
    currency: str,
    transaction_date: str,
    top_contributors: list[dict],
    merchant_name: str = "the merchant",
    model: str = DEFAULT_MODEL,
) -> str:
    api_key = os.environ["GROQ_API_KEY"]

    prompt = f"""You are drafting a chargeback dispute evidence letter for {merchant_name} to submit to a card network / acquiring bank, contesting a customer's chargeback claim on a transaction. Our automated fraud-risk model scored this transaction as LOW risk (i.e. our system assessed it as consistent with a legitimate purchase) at the time it was processed.

Transaction details:
- Transaction ID: {transaction_id}
- Amount: {currency} {amount:,.2f}
- Date: {transaction_date}

Our fraud-risk model's assessment (top contributing signals from a SHAP explanation, computed at transaction time):
{_format_contributors(top_contributors)}

Write a professional, factual dispute letter (250-350 words) that:
1. States the transaction was processed through standard authorization and passed automated fraud screening at the time of purchase.
2. References only the risk signals listed above as supporting evidence the transaction pattern was consistent with legitimate purchase behavior -- do not invent evidence not present in the data above.
3. Requests the chargeback be reversed pending review.
4. Uses a neutral, evidence-based tone with no speculation about the customer's intent.

Output only the letter text, no preamble or explanation."""

    resp = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 900,
            "reasoning_effort": "low",
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if not content:
        raise RuntimeError("Groq returned an empty completion for the dispute letter")
    return content
