import pandera.errors
import pytest

from fraud_risk.data.schema import validate_raw


def test_valid_synthetic_frame_passes(synthetic_transactions):
    validated = validate_raw(synthetic_transactions)
    assert len(validated) == len(synthetic_transactions)


def test_bad_class_value_rejected(synthetic_transactions):
    bad = synthetic_transactions.copy()
    bad.loc[0, "Class"] = 2
    with pytest.raises(pandera.errors.SchemaError):
        validate_raw(bad)


def test_negative_amount_rejected(synthetic_transactions):
    bad = synthetic_transactions.copy()
    bad.loc[0, "Amount"] = -10.0
    with pytest.raises(pandera.errors.SchemaError):
        validate_raw(bad)


def test_missing_column_rejected(synthetic_transactions):
    bad = synthetic_transactions.drop(columns=["V1"])
    with pytest.raises(pandera.errors.SchemaError):
        validate_raw(bad)
