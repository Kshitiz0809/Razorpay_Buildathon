import pandas as pd

from fraud_risk.config import FEATURE_COLUMNS


def transaction_to_dataframe(txn) -> pd.DataFrame:
    row = {"Time": txn.time, "Amount": txn.amount}
    for i in range(1, 29):
        row[f"V{i}"] = getattr(txn, f"v{i}")
    return pd.DataFrame([row])[FEATURE_COLUMNS]
