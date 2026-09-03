import os

import requests
import streamlit as st
from dotenv import load_dotenv

from common import load_curated_examples, require_artifacts

load_dotenv()

st.set_page_config(page_title="Chargeback Evidence Responder", page_icon="📝", layout="wide")
st.title("📝 Chargeback Evidence Responder")
st.caption(
    "Drafts a chargeback dispute letter from the same per-transaction SHAP "
    "explanation already generated for the Explainability page — directly "
    "implementing the hackathon brief's \"chargeback evidence responder\" "
    "direction from explainability output already built elsewhere in this "
    "app, rather than a second, disconnected feature."
)

st.warning(
    "**Defense-only guard, not just a claim:** this refuses to draft a "
    "letter for any transaction our own fraud model scored above its "
    "flagging threshold — try a \"Correctly flagged fraud\" example below "
    "and watch it refuse. It only assists disputes for transactions the "
    "system itself believes are legitimate — the real 'friendly fraud' "
    "chargeback-defense use case. This is a drafting aid for a human to "
    "review and submit, not an autonomous system: nothing here submits "
    "anything on its own."
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")

require_artifacts("curated_examples.json")
examples = load_curated_examples()

category_labels = {
    "true_positive": "✅ Correctly flagged fraud (expect a refusal)",
    "false_positive": "⚠️ Wrongly flagged legitimate transaction (expect a refusal)",
    "false_negative": "❌ Missed fraud — system believed it legitimate (will draft; an honest edge case)",
    "true_negative": "◻️ Correctly passed legitimate transaction",
    "borderline": "🎯 Borderline (near the decision threshold)",
}

st.divider()
st.subheader("Draft a dispute letter")

options = [f"{category_labels.get(ex['category'], ex['category'])} — {ex['transaction_id']}" for ex in examples]
selected_idx = st.selectbox("Choose a transaction to dispute", range(len(examples)), format_func=lambda i: options[i])
example = examples[selected_idx]

col1, col2, col3 = st.columns(3)
merchant_name = col1.text_input("Merchant name", value="Acme Retail Pvt Ltd")
currency = col2.selectbox("Currency", ["USD", "INR", "EUR", "GBP"])
transaction_date = col3.date_input("Transaction date")

if st.button("Draft dispute letter"):
    payload = {
        "transaction": {"transaction_id": example["transaction_id"], **example["raw_features"]},
        "currency": currency,
        "transaction_date": str(transaction_date),
        "merchant_name": merchant_name,
    }
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chargeback/draft-response",
            json=payload,
            headers={"X-API-Key": API_KEY},
            timeout=30,
        )
    except requests.RequestException as e:
        st.error(f"Could not reach the API ({e}). Is it running?")
    else:
        if resp.status_code == 400:
            st.error(f"Refused: {resp.json()['detail']}")
        elif resp.status_code == 503:
            st.warning("Model not trained yet — run the training pipeline first.")
        elif resp.status_code == 429:
            st.warning("Rate limit hit — wait a moment and try again.")
        elif not resp.ok:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            body = resp.json()
            st.success(f"Drafted — fraud probability at the time: {body['fraud_probability']:.4f} (below threshold)")
            st.text_area("Dispute letter", value=body["letter"], height=400)
