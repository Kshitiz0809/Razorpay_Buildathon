import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud_risk.evaluation.metrics import classification_metrics, pr_curve_points, calibration_curve_points
from common import load_test_predictions, load_evaluation_report, require_artifacts

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
st.title("📊 Model Performance")
st.caption(
    "All numbers on this page are computed on the held-out TEST block, "
    "touched exactly once during evaluation — never during training or threshold selection."
)

require_artifacts("test_predictions.parquet", "evaluation_report.json")

preds = load_test_predictions()
report = load_evaluation_report()
default_threshold = report["champion"]["threshold"]

threshold = st.slider(
    "Decision threshold", 0.0, 1.0, value=float(default_threshold), step=0.001, format="%.3f"
)
st.caption(f"Cost-optimal threshold selected on validation: **{default_threshold:.3f}**")

y_true = preds["Class"].values
scores = preds["champion_score"].values
metrics = classification_metrics(y_true, scores, threshold)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Precision", f"{metrics['precision']:.1%}")
col2.metric("Recall", f"{metrics['recall']:.1%}")
col3.metric("F1", f"{metrics['f1']:.3f}")
col4.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
col5.metric("Flagged", f"{metrics['n_flagged']} / {metrics['n_total']}")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Precision-Recall curve")
    pr = pr_curve_points(y_true, scores)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="PR curve"))
    fig.add_trace(
        go.Scatter(
            x=[metrics["recall"]],
            y=[metrics["precision"]],
            mode="markers",
            marker=dict(size=12, color="red"),
            name="current threshold",
        )
    )
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=400)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Confusion matrix")
    cm = [
        [metrics["true_negatives"], metrics["false_positives"]],
        [metrics["false_negatives"], metrics["true_positives"]],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred Legitimate", "Pred Fraud"],
            y=["Actual Legitimate", "Actual Fraud"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Calibration curve")
calib = calibration_curve_points(y_true, scores)
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="perfect calibration")
)
fig.add_trace(
    go.Scatter(x=calib["mean_predicted"], y=calib["mean_actual"], mode="lines+markers", name="champion (calibrated)")
)
fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed fraud rate", height=400)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Baseline comparison (honesty anchor)")
baseline_scores = preds["baseline_score"].values
baseline_metrics = classification_metrics(y_true, baseline_scores, 0.5)
comparison = pd.DataFrame(
    [
        {
            "Model": "Logistic Regression (baseline, t=0.5)",
            **{k: baseline_metrics[k] for k in ["precision", "recall", "f1", "pr_auc"]},
        },
        {
            "Model": f"LightGBM (champion, t={threshold:.3f})",
            **{k: metrics[k] for k in ["precision", "recall", "f1", "pr_auc"]},
        },
    ]
)
st.dataframe(
    comparison.style.format({"precision": "{:.1%}", "recall": "{:.1%}", "f1": "{:.3f}", "pr_auc": "{:.3f}"}),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "The baseline is reported as an honesty anchor, not a candidate for deployment — "
    "it shows the champion's lift is real, not an artifact of an easy dataset."
)
