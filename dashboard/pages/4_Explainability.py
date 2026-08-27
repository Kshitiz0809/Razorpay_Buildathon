import plotly.graph_objects as go
import streamlit as st

from common import load_curated_examples, require_artifacts

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")
st.title("🔍 Explainability")
st.caption(
    "Per-transaction SHAP explanations for hand-picked examples — this is "
    "what a risk analyst sees to justify why a specific transaction was "
    "flagged. No global feature-importance view is exposed anywhere in this "
    "app or its API, by design: that would hand an attacker a map of "
    "exactly which signals to spoof to evade the model."
)

require_artifacts("curated_examples.json")
examples = load_curated_examples()

category_labels = {
    "true_positive": "✅ Correctly flagged fraud",
    "false_positive": "⚠️ Wrongly flagged legitimate transaction",
    "false_negative": "❌ Missed fraud",
    "true_negative": "◻️ Correctly passed legitimate transaction",
    "borderline": "🎯 Borderline (near the decision threshold)",
}

options = [f"{category_labels.get(ex['category'], ex['category'])} — {ex['transaction_id']}" for ex in examples]
selected_idx = st.selectbox("Choose a transaction", range(len(examples)), format_func=lambda i: options[i])
example = examples[selected_idx]

col1, col2, col3, col4 = st.columns(4)
col1.metric("True label", example["true_label"])
col2.metric("Model decision", "FLAGGED" if example["flagged"] else "passed")
col3.metric("Fraud probability", f"{example['champion_score']:.4f}")
col4.metric("Amount", f"${example['amount']:.2f}")

st.divider()
st.subheader("Why this decision?")

contributors = example["top_contributors"]
names = [c["feature_name"] for c in contributors]
values = [c["shap_value"] for c in contributors]
colors = ["#C44E52" if v > 0 else "#4C72B0" for v in values]

fig = go.Figure(
    go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    )
)
fig.update_layout(
    xaxis_title="SHAP contribution  (push toward fraud →,  ← push toward legitimate)",
    yaxis=dict(autorange="reversed"),
    height=350,
)
st.plotly_chart(fig, use_container_width=True)

top = contributors[0]
direction_word = "increased" if top["direction"] == "increases_risk" else "decreased"
st.markdown(
    f"**Primary driver:** `{top['feature_name']}` = {top['feature_value']:.3f} most strongly "
    f"{direction_word} this transaction's fraud risk."
)

with st.expander("Raw contribution values"):
    st.json(contributors)
