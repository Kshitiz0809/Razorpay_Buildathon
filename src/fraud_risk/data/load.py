import pandas as pd

from fraud_risk.config import RAW_CSV
from fraud_risk.data.schema import validate_raw


def load_raw_transactions(path=RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it first with:\n"
            "  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip"
        )
    df = pd.read_csv(path)
    df = validate_raw(df)
    return df.sort_values("Time").reset_index(drop=True)
