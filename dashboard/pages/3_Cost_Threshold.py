import plotly.graph_objects as go
import streamlit as st

from fraud_risk.cost.cost_model import select_optimal_threshold, cost_flag_none, cost_flag_all
from common import load_test_predictions, load_cost_config, require_artifacts

st.set_page_config(page_title="Cost & Threshold Optimization", page_icon="💰", layout="wide")
st.title("💰 Cost & Threshold Optimization")
st.caption(
    "The dollar-cost model behind the chosen operating threshold — every "
    "assumption below is editable and auditable, never hardcoded silently inside the model."
)

require_artifacts("test_predictions.parquet")
preds = load_test_predictions()
default_cfg = load_cost_config()

st.subheader("Assumptions")
st.caption(
    "Loaded from configs/cost_config.yaml — adjust below to see how the "
    "optimal threshold changes in real time."
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**False Negative — missed fraud**")
    fn_fee = st.number_input(
        "Chargeback processing fee ($)",
        value=float(default_cfg["false_negative"]["flat_chargeback_fee"]),
        step=1.0,
    )
    st.caption("Plus the real transaction Amount, always.")

    st.markdown("**True Positive — correctly flagged**")
    tp_cost = st.number_input(
        "Manual investigation cost ($)",
        value=float(default_cfg["true_positive"]["investigation_cost"]),
        step=0.5,
    )

with col2:
    st.markdown("**False Positive — wrongly declined**")
    fp_friction = st.number_input(
        "Support / friction cost ($)", value=float(default_cfg["false_positive"]["avg_friction_cost"]), step=0.5
    )
    fp_pct = st.number_input(
        "Lost margin (% of amount)",
        value=float(default_cfg["false_positive"]["lost_revenue_pct_of_amount"]),
        step=0.01,
        format="%.2f",
    )
    fp_churn = st.number_input(
        "Churn risk cost ($)", value=float(default_cfg["false_positive"]["churn_risk_cost"]), step=0.5
    )

cfg = {
    "false_negative": {"flat_chargeback_fee": fn_fee},
    "false_positive": {
        "avg_friction_cost": fp_friction,
        "lost_revenue_pct_of_amount": fp_pct,
        "churn_risk_cost": fp_churn,
    },
    "true_positive": {"investigation_cost": tp_cost},
    "true_negative": {"cost": 0.0},
}

y_true = preds["Class"].values
amount = preds["Amount"].values
scores = preds["champion_score"].values

best_t, best_cost, curve = select_optimal_threshold(y_true, amount, scores, cfg)
naive_none = cost_flag_none(y_true, amount, cfg)
naive_all = cost_flag_all(y_true, amount, cfg)

st.divider()
col1, col2, col3 = st.columns(3)
col1.metric("Optimal threshold", f"{best_t:.3f}")
col2.metric("Cost at optimal threshold", f"${best_cost:,.2f}")
col3.metric("Savings vs. flagging nothing", f"${naive_none - best_cost:,.2f}")

st.subheader("Cost vs. threshold")
fig = go.Figure()
fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["total_cost"], mode="lines", name="expected cost"))
fig.add_vline(x=best_t, line_dash="dash", line_color="red", annotation_text=f"optimal t={best_t:.3f}")
fig.add_hline(y=naive_none, line_dash="dot", line_color="gray", annotation_text="flag nothing")
fig.add_hline(y=naive_all, line_dash="dot", line_color="orange", annotation_text="flag everything")
fig.update_layout(xaxis_title="Threshold", yaxis_title="Total expected cost ($) on TEST", height=450)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "This curve is recomputed live on held-out TEST predictions purely to "
    "demonstrate sensitivity to assumptions. The threshold actually shipped "
    "with the model was selected on the VALIDATION block using "
    "configs/cost_config.yaml — TEST is used here only for this illustrative "
    "view, never to pick the deployed threshold."
)
