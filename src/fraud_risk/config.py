from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "champion"
REPORTS_DIR = ROOT / "reports"
CONFIGS_DIR = ROOT / "configs"

RAW_CSV = DATA_RAW / "creditcard.csv"

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time", "Amount"] + V_COLUMNS
LABEL_COLUMN = "Class"


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_config() -> dict:
    return load_yaml(CONFIGS_DIR / "train_config.yaml")


def cost_config() -> dict:
    return load_yaml(CONFIGS_DIR / "cost_config.yaml")
