import numpy as np
import pandas as pd
import pytest

from fraud_risk.config import V_COLUMNS


def _make_synthetic_transactions(n=2000, fraud_rate=0.02, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_rate))

    time = np.sort(rng.uniform(0, 172792, size=n))
    amount = np.abs(rng.normal(50, 40, size=n))
    label = np.zeros(n, dtype=int)
    fraud_idx = rng.choice(n, size=n_fraud, replace=False)
    label[fraud_idx] = 1

    data = {"Time": time, "Amount": amount}
    for v in V_COLUMNS:
        col = rng.normal(0, 1, size=n)
        col[fraud_idx] += rng.normal(3, 1, size=n_fraud)  # separable signal for fraud rows
        data[v] = col
    data["Class"] = label

    return pd.DataFrame(data).sort_values("Time").reset_index(drop=True)


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    return _make_synthetic_transactions()
