import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import joblib

from fraud_risk.config import MODELS_DIR


@dataclass
class ArtifactMetadata:
    model_version: str
    trained_at: str
    train_rows: int
    val_rows: int
    test_rows: int
    train_frauds: int
    val_frauds: int
    test_frauds: int
    threshold: float
    feature_columns: list[str]


def save_bundle(
    calibrated_champion,
    baseline_pipeline,
    threshold: float,
    metadata_fields: dict,
) -> ArtifactMetadata:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibrated_champion, MODELS_DIR / "champion_calibrated.joblib")
    joblib.dump(baseline_pipeline, MODELS_DIR / "baseline.joblib")

    metadata = ArtifactMetadata(
        model_version=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        trained_at=datetime.now(timezone.utc).isoformat(),
        threshold=threshold,
        **metadata_fields,
    )
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(asdict(metadata), f, indent=2)

    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    return metadata


def load_champion():
    return joblib.load(MODELS_DIR / "champion_calibrated.joblib")


def load_baseline():
    return joblib.load(MODELS_DIR / "baseline.joblib")


def load_metadata() -> dict:
    with open(MODELS_DIR / "metadata.json") as f:
        return json.load(f)


def load_threshold() -> float:
    with open(MODELS_DIR / "threshold.json") as f:
        return json.load(f)["threshold"]
