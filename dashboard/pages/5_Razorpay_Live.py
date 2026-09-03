import json
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

from fraud_risk.razorpay_integration.simulate import PROFILES, build_payload, build_ring_event
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


def _send_event(event: dict):
    raw_body = json.dumps(event).encode("utf-8")
    signature = sign_payload(raw_body, WEBHOOK_SECRET)
    resp = requests.post(
        f"{API_BASE_URL}/webhook/razorpay/payment",
        data=raw_body,
        headers={"Content-Type": "application/json", "X-Razorpay-Signature": signature},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _render_risk_result(result: dict):
    level_color = {"low": "green", "medium": "orange", "high": "red"}[result["risk_level"]]
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Risk score", result["risk_score"])
    mcol2.markdown(f"Risk level: **:{level_color}[{result['risk_level'].upper()}]**")

    if result["triggered_rules"]:
        names = [r["rule"] for r in result["triggered_rules"]]
        points = [r["points"] for r in result["triggered_rules"]]
        reasons = [r["reason"] for r in result["triggered_rules"]]
        fig = go.Figure(go.Bar(x=points, y=names, orientation="h", text=reasons, textposition="auto"))
        fig.update_layout(xaxis_title="Points contributed", height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rules triggered — this event looked routine.")

    with st.expander("Raw webhook response"):
        st.json(result)


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
        try:
            last_result = _send_event(build_payload(PROFILES[profile_choice]))
        except requests.RequestException as e:
            st.error(f"Could not reach the API ({e}). Is it running?")
            last_result = None
            break

    if last_result:
        _render_risk_result(last_result)

st.divider()
st.subheader("Abuse-ring sentinel")
st.caption(
    "A coordinated ring probing with one device and one card against many "
    "synthetic identities doesn't look risky from any single transaction's "
    "amount or method — it only shows up across events. This fires N events "
    "sharing the same IP address and card fingerprint but a fresh email "
    "each time, and shows the card-reuse and IP-velocity rules trigger "
    "statefully as the ring plays out."
)

ring_size = st.slider("Ring size (distinct identities, same IP + card)", min_value=2, max_value=8, value=4)

if st.button("Simulate abuse ring"):
    results = []
    try:
        for i in range(ring_size):
            results.append(_send_event(build_ring_event(i)))
    except requests.RequestException as e:
        st.error(f"Could not reach the API ({e}). Is it running?")
        results = []

    if results:
        timeline = pd.DataFrame(
            [
                {
                    "event": i + 1,
                    "payment_id": r["payment_id"],
                    "risk_score": r["risk_score"],
                    "risk_level": r["risk_level"],
                    "triggered_rules": ", ".join(rule["rule"] for rule in r["triggered_rules"]) or "none",
                }
                for i, r in enumerate(results)
            ]
        )
        st.dataframe(timeline, use_container_width=True, hide_index=True)

        fig = go.Figure(go.Scatter(x=timeline["event"], y=timeline["risk_score"], mode="lines+markers"))
        fig.update_layout(
            xaxis_title="Event # in the ring", yaxis_title="Risk score", height=300, xaxis=dict(dtick=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Risk escalates as the same card/IP keeps reappearing under new identities — that's the signal.")
