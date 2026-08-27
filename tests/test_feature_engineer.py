import numpy as np
import pandas as pd
import pytest

from fraud_risk.features.engineer import FeatureEngineer, CHAMPION_COLUMNS, BASELINE_COLUMNS
from fraud_risk.config import V_COLUMNS


def _toy_frame():
    row = {"Time": 0.0, "Amount": np.e - 1}  # log1p(Amount) == 1.0
    row.update({v: 0.0 for v in V_COLUMNS})
    return pd.DataFrame([row])


def test_midnight_hour_has_zero_sin_and_one_cos():
    fe = FeatureEngineer(CHAMPION_COLUMNS)
    out = fe.transform(_toy_frame())
    assert out["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-9)


def test_log1p_amount_applied_for_baseline():
    fe = FeatureEngineer(BASELINE_COLUMNS)
    out = fe.transform(_toy_frame())
    assert out["Amount_log1p"].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_raw_time_is_dropped_from_output():
    fe = FeatureEngineer(CHAMPION_COLUMNS)
    out = fe.transform(_toy_frame())
    assert "Time" not in out.columns


def test_output_column_order_matches_requested():
    fe = FeatureEngineer(CHAMPION_COLUMNS)
    out = fe.transform(_toy_frame())
    assert list(out.columns) == CHAMPION_COLUMNS
