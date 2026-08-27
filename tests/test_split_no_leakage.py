import pytest

from fraud_risk.data.split import chronological_split, assert_no_time_leakage


def test_chronological_blocks_do_not_overlap(synthetic_transactions):
    split = chronological_split(synthetic_transactions, min_test_frauds=1)
    assert_no_time_leakage(split)  # should not raise


def test_split_covers_every_row_exactly_once(synthetic_transactions):
    split = chronological_split(synthetic_transactions, min_test_frauds=1)
    assert len(split.train) + len(split.val) + len(split.test) == len(synthetic_transactions)


def test_raises_when_test_block_has_too_few_frauds(synthetic_transactions):
    with pytest.raises(ValueError, match="frauds"):
        chronological_split(synthetic_transactions, min_test_frauds=10_000)


def test_train_and_val_each_contain_fraud_examples(synthetic_transactions):
    split = chronological_split(synthetic_transactions, min_test_frauds=1)
    assert split.train["Class"].sum() > 0
    assert split.val["Class"].sum() > 0
