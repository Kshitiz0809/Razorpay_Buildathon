"""Shared feature engineering, wrapped as a sklearn transformer so it can be
serialized inside the same Pipeline as the model -- this makes it
structurally impossible to apply a mismatched transform at inference time.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from fraud_risk.config import V_COLUMNS

# Trees are scale-invariant and don't need the log1p amount; raw Amount is enough.
CHAMPION_COLUMNS = ["Amount", "hour_sin", "hour_cos"] + V_COLUMNS
# Logistic regression benefits from de-skewing the heavy-tailed Amount.
BASELINE_COLUMNS = ["Amount_log1p", "hour_sin", "hour_cos"] + V_COLUMNS


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds hour-of-day cyclical features and a log1p Amount; drops raw Time.

    Raw `Time` (absolute seconds since dataset start) is never used as a
    model input -- only its cyclical, position-independent hour-of-day
    component is, so the model can't key off "this is late in the dataset."
    """

    def __init__(self, output_columns: list[str]):
        self.output_columns = output_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        hour = (df["Time"] % 86400) / 3600.0
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["Amount_log1p"] = np.log1p(df["Amount"])
        return df[self.output_columns]
