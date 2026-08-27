"""Chronological (time-ordered) train/val/test split.

Fraud scoring is a forecasting problem: in production the model only ever
sees the past. A random stratified split would let training rows sit after
test rows in time, which both leaks distributional information into
training and misrepresents deployment reality. So we sort by `Time` and cut
three non-overlapping blocks instead.
"""
from dataclasses import dataclass

import pandas as pd

from fraud_risk.config import LABEL_COLUMN


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    min_test_frauds: int = 80,
) -> SplitResult:
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1, each in (0, 1)")

    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    n_test_frauds = int(test[LABEL_COLUMN].sum())
    if n_test_frauds < min_test_frauds:
        raise ValueError(
            f"Chronological test block has only {n_test_frauds} frauds "
            f"(< min_test_frauds={min_test_frauds}). Widen test_frac in "
            "configs/train_config.yaml based on this count -- never based "
            "on resulting model metrics."
        )

    assert train[LABEL_COLUMN].sum() > 0, "train block has zero fraud examples"
    assert val[LABEL_COLUMN].sum() > 0, "val block has zero fraud examples"

    return SplitResult(train=train, val=val, test=test)


def assert_no_time_leakage(split: SplitResult) -> None:
    assert split.train["Time"].max() <= split.val["Time"].min(), (
        "train/val time ranges overlap"
    )
    assert split.val["Time"].max() <= split.test["Time"].min(), (
        "val/test time ranges overlap"
    )
