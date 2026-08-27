import streamlit as st

from common import load_evaluation_report

st.set_page_config(page_title="Fraud Risk Manager", page_icon="🛡️", layout="wide")

st.title("🛡️ Real-Time Fraud Risk Scorer")
st.caption("Razorpay AI Risk Manager track — a payment-gateway-grade transaction fraud detector")

st.markdown(
    "Built the way a payment gateway's own risk team would build it: a "
    "calibrated, cost-optimized, explainable fraud classifier evaluated with "
    "a strictly **chronological** train/validation/test split, so the "
    "reported numbers reflect what the model would have actually caught in "
    "production — not an optimistic random split."
)

report = load_evaluation_report()

if report is None:
    st.info(
        "No trained model found yet. Run the pipeline below, then reload this page."
    )
    st.code(
        "python -m fraud_risk.models.train\n"
        "python -m fraud_risk.evaluation.report\n"
        "python scripts/build_curated_examples.py",
        language="bash",
    )
else:
    champ = report["champion"]
    cost = report["cost_usd"]
    meta = report["model_metadata"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision (TEST)", f"{champ['precision']:.1%}")
    col2.metric("Recall (TEST)", f"{champ['recall']:.1%}")
    col3.metric("PR-AUC (TEST)", f"{champ['pr_auc']:.3f}")
    col4.metric("$ saved vs. flagging nothing", f"${cost['savings_vs_flag_nothing']:,.0f}")

    st.divider()
    st.markdown(
        f"Model version `{meta['model_version']}` — trained on **{meta['train_rows']:,}** "
        f"transactions ({meta['train_frauds']} fraud), calibrated on "
        f"**{meta['val_rows']:,}** transactions ({meta['val_frauds']} fraud), and "
        f"evaluated **exactly once** on a chronologically-later, held-out "
        f"**{meta['test_rows']:,}**-transaction block ({meta['test_frauds']} fraud) "
        f"that the model never influenced training or threshold selection with."
    )

st.divider()
st.markdown(
    """
### Pages
- **Live Transaction Stream** — replays held-out test transactions through the live scoring API in real time
- **Model Performance** — precision/recall, PR curve, calibration curve, confusion matrix with an interactive threshold slider
- **Cost & Threshold Optimization** — the auditable dollar-cost model behind the chosen operating threshold
- **Explainability** — SHAP-based per-transaction explanations for hand-picked examples

*Use the sidebar to navigate between pages.*
"""
)

st.divider()
st.caption(
    "Defense-only prototype: scores and explains individual transactions. "
    "No global feature importances or model internals are exposed anywhere in this app or its API."
)
