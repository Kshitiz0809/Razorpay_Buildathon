import os
import time

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from fraud_risk.config import DATA_PROCESSED, FEATURE_COLUMNS, LABEL_COLUMN
from fraud_risk.cost.cost_model import cost_of_decisions
from common import load_cost_config

load_dotenv()

st.set_page_config(page_title="Live Transaction Stream", page_icon="🔴", layout="wide")
st.title("🔴 Live Transaction Stream")
st.caption(
    "Replays held-out TEST transactions through the live scoring API in "
    "chronological order — this is a replay of historical data the model "
    "never trained on, not live production traffic."
)

TEST_PATH = DATA_PROCESSED / "test.parquet"
if not TEST_PATH.exists():
    st.warning("No processed test split found. Run `python -m fraud_risk.models.train` first.")
    st.stop()

test_df = pd.read_parquet(TEST_PATH).reset_index(drop=True)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")

for key, default in [("stream_position", 0), ("stream_log", []), ("autoplay", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.subheader("Stream controls")
    batch_size = st.number_input("Batch size", min_value=1, max_value=100, value=10)
    playback_delay = st.slider("Autoplay delay (seconds)", 0.1, 3.0, 0.5)
    col_a, col_b = st.columns(2)
    step_clicked = col_a.button("Score next batch", use_container_width=True)
    reset_clicked = col_b.button("Reset", use_container_width=True)
    st.session_state.autoplay = st.checkbox("Autoplay", value=st.session_state.autoplay)

if reset_clicked:
    st.session_state.stream_position = 0
    st.session_state.stream_log = []
    st.session_state.autoplay = False
    st.rerun()


def _payload_from_row(row: pd.Series) -> dict:
    payload = {"transaction_id": f"txn_{row.name:06d}", "time": float(row["Time"]), "amount": float(row["Amount"])}
    for i in range(1, 29):
        payload[f"v{i}"] = float(row[f"V{i}"])
    return payload


def _score_next_batch():
    start = st.session_state.stream_position
    end = min(start + batch_size, len(test_df))
    if start >= end:
        st.session_state.autoplay = False
        return
    for idx in range(start, end):
        row = test_df.iloc[idx]
        payload = _payload_from_row(row)
        try:
            resp = requests.post(
                f"{API_BASE_URL}/score", json=payload, headers={"X-API-Key": API_KEY}, timeout=5
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as e:
            st.error(f"Could not reach the API at {API_BASE_URL}. Is it running? ({e})")
            st.session_state.autoplay = False
            return
        st.session_state.stream_log.append(
            {
                "transaction_id": result["transaction_id"],
                "amount": payload["amount"],
                "fraud_probability": result["fraud_probability"],
                "flagged": result["is_flagged"],
                "actual_label": "fraud" if row[LABEL_COLUMN] == 1 else "legitimate",
            }
        )
    st.session_state.stream_position = end


if step_clicked:
    _score_next_batch()

if st.session_state.autoplay:
    _score_next_batch()

log = st.session_state.stream_log
n_remaining = len(test_df) - st.session_state.stream_position

col1, col2, col3, col4 = st.columns(4)
col1.metric("Scored so far", len(log))
col2.metric("Flagged as fraud", sum(1 for r in log if r["flagged"]))
col3.metric("Remaining in stream", max(0, n_remaining))

if log:
    log_df = pd.DataFrame(log)
    y_true = (log_df["actual_label"] == "fraud").astype(int).values
    cost_cfg = load_cost_config()
    realized_cost = cost_of_decisions(y_true, log_df["amount"].values, log_df["flagged"].values, cost_cfg)
    never_flag_cost = cost_of_decisions(y_true, log_df["amount"].values, [False] * len(log_df), cost_cfg)
    col4.metric("$ saved so far vs. flagging nothing", f"${never_flag_cost - realized_cost:,.2f}")

    st.divider()
    st.subheader("Recent decisions")

    def _highlight(row):
        if row["flagged"] and row["actual_label"] == "fraud":
            return ["background-color: rgba(0, 200, 0, 0.15)"] * len(row)
        if row["flagged"] and row["actual_label"] != "fraud":
            return ["background-color: rgba(255, 165, 0, 0.15)"] * len(row)
        if not row["flagged"] and row["actual_label"] == "fraud":
            return ["background-color: rgba(255, 0, 0, 0.15)"] * len(row)
        return [""] * len(row)

    display_df = log_df.iloc[::-1].head(50)
    st.dataframe(
        display_df.style.apply(_highlight, axis=1).format({"fraud_probability": "{:.4f}", "amount": "${:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Click **Score next batch** in the sidebar (or enable Autoplay) to start the stream.")

if st.session_state.autoplay and n_remaining > 0:
    time.sleep(playback_delay)
    st.rerun()
