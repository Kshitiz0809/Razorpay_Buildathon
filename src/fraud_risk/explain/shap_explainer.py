"""Per-transaction SHAP explanations.

Deliberately exposes only top-N per-transaction feature contributions, never
a global feature-importance summary and never the raw feature vector
alongside the explanation -- a global importance endpoint would hand an
attacker a map of exactly which signals to spoof to evade the model, which
would make this offense-capable and disqualify it under the hackathon's
defense-only rule.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from fraud_risk.features.engineer import CHAMPION_COLUMNS

TOP_N_DEFAULT = 6


@dataclass
class FeatureContribution:
    feature_name: str
    shap_value: float
    feature_value: float
    direction: str


class ShapExplainer:
    def __init__(self, calibrated_champion):
        # Calibration only rescales probabilities; the feature-contribution
        # story is explained against the underlying tree model, not the
        # calibration wrapper (SHAP's TreeExplainer needs a tree model).
        pipeline = calibrated_champion.calibrated_classifiers_[0].estimator
        self._feature_engineer = pipeline.named_steps["features"]
        self._model = pipeline.named_steps["model"]
        self._tree_explainer = shap.TreeExplainer(
            self._model, feature_perturbation="tree_path_dependent"
        )

    def _shap_values_for_positive_class(self, transformed: pd.DataFrame) -> np.ndarray:
        # shap emits a UserWarning on every call here noting its return shape
        # for binary classifiers "has changed" -- verified empirically that
        # for our pinned shap/lightgbm versions it returns a plain ndarray
        # already aligned to the positive class, which the branch below
        # handles; the warning is pure log noise at this pinned version.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            raw = self._tree_explainer.shap_values(transformed)
        # Older shap versions return [class0_values, class1_values] for binary
        # classifiers; newer versions return a single array already aligned
        # to the positive class. Handle both.
        if isinstance(raw, list):
            return np.asarray(raw[1])
        return np.asarray(raw)

    def explain_batch(self, raw_df: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> list[list[FeatureContribution]]:
        transformed = self._feature_engineer.transform(raw_df)
        values = self._shap_values_for_positive_class(transformed)

        results = []
        for i in range(len(transformed)):
            row_values = values[i]
            row_features = transformed.iloc[i]
            order = np.argsort(-np.abs(row_values))[:top_n]
            results.append(
                [
                    FeatureContribution(
                        feature_name=CHAMPION_COLUMNS[j],
                        shap_value=float(row_values[j]),
                        feature_value=float(row_features.iloc[j]),
                        direction=(
                            "increases_risk"
                            if row_values[j] > 0
                            else "decreases_risk"
                            if row_values[j] < 0
                            else "neutral"
                        ),
                    )
                    for j in order
                ]
            )
        return results

    def explain_one(self, raw_row_df: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> list[FeatureContribution]:
        return self.explain_batch(raw_row_df, top_n=top_n)[0]
