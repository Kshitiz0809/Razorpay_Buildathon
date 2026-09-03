import plotly.graph_objects as go
import streamlit as st

from fraud_risk.cost.cost_model import select_optimal_threshold, cost_flag_none, cost_flag_all, total_cost
from fraud_risk.evaluation.metrics import classification_metrics
from common import load_test_predictions, load_cost_config, require_artifacts

NAIVE_THRESHOLD = 0.5

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
best_t_is_flag_none = best_t == float("inf")

st.divider()
col1, col2, col3 = st.columns(3)
col1.metric("Optimal threshold", "∞ (flag nothing)" if best_t_is_flag_none else f"{best_t:.3f}")
col2.metric("Cost at optimal threshold", f"${best_cost:,.2f}")
col3.metric("Savings vs. flagging nothing", f"${naive_none - best_cost:,.2f}")

if best_t_is_flag_none:
    st.warning(
        "Under these assumptions, flagging **nothing** minimizes total cost — "
        "the false-positive cost outweighs the fraud losses being prevented. "
        "This is the threshold sweep correctly considering 'do nothing' as a "
        "candidate policy, not a bug."
    )

st.divider()
st.subheader("Naive vs. cost-optimized: the $ difference")
st.caption(
    "Same model, same scores, same test set — the only difference is which "
    "threshold gets shipped. 'Naive' is the textbook default (t = 0.5) most "
    "tutorials stop at; 'cost-optimized' is the threshold that minimizes "
    "the dollar-cost model above."
)

naive_cost = total_cost(y_true, amount, scores, NAIVE_THRESHOLD, cfg)
naive_metrics = classification_metrics(y_true, scores, NAIVE_THRESHOLD)
optimized_metrics = classification_metrics(y_true, scores, best_t)

naive_col, optimized_col = st.columns(2)
with naive_col:
    st.markdown("#### 😐 Naive model &nbsp; (t = 0.50)")
    st.metric("Total cost", f"${naive_cost:,.2f}")
    st.metric("Precision", f"{naive_metrics['precision']:.1%}")
    st.metric("Recall", f"{naive_metrics['recall']:.1%}")

with optimized_col:
    st.markdown(f"#### 🎯 Cost-optimized model &nbsp; (t = {'∞' if best_t_is_flag_none else f'{best_t:.3f}'})")
    st.metric("Total cost", f"${best_cost:,.2f}", delta=f"{best_cost - naive_cost:,.2f}", delta_color="inverse")
    st.metric("Precision", f"{optimized_metrics['precision']:.1%}")
    st.metric("Recall", f"{optimized_metrics['recall']:.1%}")

fig = go.Figure(
    go.Bar(
        x=["Naive (t=0.5)", "Cost-optimized"],
        y=[naive_cost, best_cost],
        marker_color=["#C44E52", "#4C72B0"],
        text=[f"${naive_cost:,.0f}", f"${best_cost:,.0f}"],
        textposition="outside",
    )
)
fig.update_layout(yaxis_title="Total expected cost ($) on TEST", height=350)
st.plotly_chart(fig, use_container_width=True)

if naive_cost > 0:
    savings_pct = (naive_cost - best_cost) / naive_cost * 100
    st.success(
        f"Cost-aware thresholding saves **${naive_cost - best_cost:,.2f}** "
        f"({savings_pct:.1f}%) over the naive default on this test set — "
        "identical model, identical scores, only the threshold changed."
    )

st.subheader("Cost vs. threshold")
finite_curve = curve[curve["threshold"].apply(lambda t: t != float("inf"))]
fig = go.Figure()
fig.add_trace(go.Scatter(x=finite_curve["threshold"], y=finite_curve["total_cost"], mode="lines", name="expected cost"))
if not best_t_is_flag_none:
    fig.add_vline(x=best_t, line_dash="dash", line_color="red", annotation_text=f"optimal t={best_t:.3f}")
fig.add_hline(y=naive_none, line_dash="dot", line_color="gray", annotation_text="flag nothing")
fig.add_hline(y=naive_all, line_dash="dot", line_color="orange", annotation_text="flag everything")
# Log scale: "flag everything" costs orders of magnitude more than the
# useful operating range, and on a linear axis that squashes the
# actually-interesting part of the curve flat.
fig.update_layout(
    xaxis_title="Threshold",
    yaxis_title="Total expected cost ($, log scale) on TEST",
    yaxis_type="log",
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "This curve is recomputed live on held-out TEST predictions purely to "
    "demonstrate sensitivity to assumptions. The threshold actually shipped "
    "with the model was selected on the VALIDATION block using "
    "configs/cost_config.yaml — TEST is used here only for this illustrative "
    "view, never to pick the deployed threshold."
)
