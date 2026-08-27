import json

import pandas as pd
import streamlit as st

from fraud_risk.config import REPORTS_DIR, CONFIGS_DIR, load_yaml


def load_evaluation_report() -> dict | None:
    path = REPORTS_DIR / "evaluation_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_test_predictions() -> pd.DataFrame | None:
    path = REPORTS_DIR / "test_predictions.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_curated_examples() -> list | None:
    path = REPORTS_DIR / "curated_examples.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_cost_config() -> dict:
    return load_yaml(CONFIGS_DIR / "cost_config.yaml")


def require_artifacts(*names: str) -> None:
    """Friendly stop-and-explain if required generated reports are missing."""
    missing = [n for n in names if not (REPORTS_DIR / n).exists()]
    if missing:
        st.warning(f"Missing: {', '.join(missing)}")
        st.code(
            "python -m fraud_risk.models.train\n"
            "python -m fraud_risk.evaluation.report\n"
            "python scripts/build_curated_examples.py",
            language="bash",
        )
        st.stop()
