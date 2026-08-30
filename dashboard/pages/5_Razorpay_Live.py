import json
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

from fraud_risk.razorpay_integration.simulate import PROFILES, build_payload
from fraud_risk.razorpay_integration.webhook import sign_payload

load_dotenv()

st.set_page_config(page_title="Razorpay Live Integration", page_icon="🔗", layout="wide")
st.title("🔗 Razorpay Live Integration")
st.caption(
    "A real integration against Razorpay's own Test Mode API — not a "
    "simulation of Razorpay, an actual client of it. The order lookup "
    "below hits the live API with real test credentials; the payment-risk "
    "demo sends signed webhook events shaped exactly like the real ones "
    "Razorpay's servers send, since completing an actual payment requires "
    "Razorpay's client-side Checkout flow by design (PCI-DSS scope "
    "reduction) — no backend can do that alone."
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")
WEBHOOK_SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")

st.divider()
st.subheader("Live account connection")

if st.button("Fetch recent orders from Razorpay"):
    try:
        resp = requests.get(
            f"{API_BASE_URL}/razorpay/recent-orders",
            headers={"X-API-Key": API_KEY},
            params={"count": 10},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        st.success(f"Connected — test mode: {data['test_mode']}")
        orders = data["orders"]
        if orders:
            df = pd.DataFrame(orders)[["id", "amount", "currency", "status", "receipt", "created_at"]]
            df["amount"] = df["amount"] / 100
            df["created_at"] = pd.to_datetime(df["created_at"], unit="s")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No orders in this account yet. Create a few real ones with:\n\n"
                "`python scripts/seed_razorpay_test_orders.py`"
            )
    except requests.RequestException as e:
        st.error(f"Could not reach the API or Razorpay ({e}). Is the API running?")

st.divider()
st.subheader("Live payment risk gate")
st.caption(
    "Stage 1 of a two-stage strategy: a transparent, auditable rule engine "
    "for a payment stream with no labeled fraud outcomes yet — there's no "
    "chargeback history for a brand-new integration to train on. Once real "
    "outcomes accumulate, this graduates to the same calibrated, "
    "precision/recall-measured model demonstrated elsewhere in this app."
)

col1, col2 = st.columns(2)
profile_choice = col1.selectbox("Simulated payment profile", list(PROFILES.keys()))
repeat = col2.number_input("Repeat (same payer — demos the velocity rule)", min_value=1, max_value=10, value=1)

if st.button("Send simulated payment.authorized webhook event"):
    last_result = None
    for _ in range(int(repeat)):
        event = build_payload(PROFILES[profile_choice])
        raw_body = json.dumps(event).encode("utf-8")
        signature = sign_payload(raw_body, WEBHOOK_SECRET)
        try:
            resp = requests.post(
                f"{API_BASE_URL}/webhook/razorpay/payment",
                data=raw_body,
                headers={"Content-Type": "application/json", "X-Razorpay-Signature": signature},
                timeout=10,
            )
            resp.raise_for_status()
            last_result = resp.json()
        except requests.RequestException as e:
            st.error(f"Could not reach the API ({e}). Is it running?")
            last_result = None
            break

    if last_result:
        level_color = {"low": "green", "medium": "orange", "high": "red"}[last_result["risk_level"]]
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Risk score", last_result["risk_score"])
        mcol2.markdown(f"Risk level: **:{level_color}[{last_result['risk_level'].upper()}]**")

        if last_result["triggered_rules"]:
            names = [r["rule"] for r in last_result["triggered_rules"]]
            points = [r["points"] for r in last_result["triggered_rules"]]
            reasons = [r["reason"] for r in last_result["triggered_rules"]]
            fig = go.Figure(go.Bar(x=points, y=names, orientation="h", text=reasons, textposition="auto"))
            fig.update_layout(xaxis_title="Points contributed", height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rules triggered — this event looked routine.")

        with st.expander("Raw webhook response"):
            st.json(last_result)
