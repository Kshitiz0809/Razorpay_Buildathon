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

st.divider()
st.subheader("🤖 Investigate & Auto-Respond (agentic)")
st.caption(
    "Instead of drafting from the SHAP explanation alone, this gives the "
    "LLM tools and lets it investigate before concluding: IP reputation "
    "(real lookup via ip-api.com, with a local fallback), shipment "
    "delivery status (simulated — no real order/carrier data exists for "
    "this dataset), and this merchant's own prior-chargeback history for "
    "the customer. The same high-risk refusal from above still applies, "
    "checked in code before the agent ever runs — the model decides how "
    "to investigate and whether to submit, never whether it's allowed to."
)

col1, col2, col3 = st.columns(3)
inv_email = col1.text_input(
    "Customer email", value="regular.customer@example.com", help="Try repeat.disputer@example.com for a red flag"
)
inv_ip = col2.text_input("Customer IP", value="8.8.8.8", help="Try a residential-looking IP for a different result")
inv_tracking = col3.text_input("Tracking number", value="TRACK123456")

allow_submit = st.checkbox(
    "Allow the agent to auto-submit evidence to the mock endpoint if it concludes the dispute should be contested",
    value=False,
)
if allow_submit:
    st.caption(
        "⚠️ With this checked, a click below can result in a real (simulated, local-only) "
        "submission — logged to data/mock_services/submitted_disputes.json, never sent externally."
    )

if st.button("Investigate & Respond"):
    payload = {
        "transaction": {"transaction_id": example["transaction_id"], **example["raw_features"]},
        "currency": currency,
        "transaction_date": str(transaction_date),
        "merchant_name": merchant_name,
        "customer_email": inv_email or None,
        "customer_ip": inv_ip or None,
        "tracking_number": inv_tracking or None,
        "allow_submit": allow_submit,
    }
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chargeback/investigate",
            json=payload,
            headers={"X-API-Key": API_KEY},
            timeout=60,
        )
    except requests.RequestException as e:
        st.error(f"Could not reach the API ({e}). Is it running?")
    else:
        if resp.status_code == 400:
            st.error(f"Refused: {resp.json()['detail']}")
        elif resp.status_code == 503:
            st.warning("Model not trained yet — run the training pipeline first.")
        elif resp.status_code == 429:
            st.warning("Rate limit hit (either this app's or Groq's) — wait a moment and try again.")
        elif not resp.ok:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            body = resp.json()
            st.success(f"Investigated — fraud probability at the time: {body['fraud_probability']:.4f} (below threshold)")

            st.markdown("**Investigation trace** (tools the agent chose to call, in order):")
            for i, step in enumerate(body["investigation_trace"], start=1):
                with st.expander(f"Step {i}: called `{step['tool']}`", expanded=True):
                    st.json({"arguments": step["arguments"], "result": step["result"]})

            if body["submitted"]:
                st.success(f"✅ Evidence submitted (simulated) — confirmation `{body['submission_reference']}`")
            elif allow_submit:
                st.info("The agent investigated but chose not to submit.")

            st.markdown("**Conclusion:**")
            st.markdown(body["conclusion"])
