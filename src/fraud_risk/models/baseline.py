from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_risk.features.engineer import FeatureEngineer, BASELINE_COLUMNS


def build_baseline_pipeline(config: dict) -> Pipeline:
    """LogisticRegression(class_weight='balanced') baseline.

    This is the "honesty anchor" -- a simple, well-understood model reported
    alongside the champion so the champion's lift is credible rather than a
    black-box claim.
    """
    return Pipeline(
        steps=[
            ("features", FeatureEngineer(BASELINE_COLUMNS)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty=config.get("penalty", "l2"),
                    C=config.get("C", 1.0),
                    max_iter=config.get("max_iter", 1000),
                    class_weight=config.get("class_weight", "balanced"),
                    random_state=42,
                ),
            ),
        ]
    )
