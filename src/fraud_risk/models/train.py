"""Training entrypoint.

load raw -> chronological split -> fit baseline + champion -> calibrate
champion on val -> select cost-minimizing threshold on val -> save artifacts.

Run:  python -m fraud_risk.models.train
"""
from fraud_risk.config import train_config, cost_config, DATA_PROCESSED, LABEL_COLUMN, FEATURE_COLUMNS
from fraud_risk.data.load import load_raw_transactions
from fraud_risk.data.split import chronological_split, assert_no_time_leakage
from fraud_risk.models.baseline import build_baseline_pipeline
from fraud_risk.models.champion import fit_champion_pipeline
from fraud_risk.models.calibration import calibrate_on_validation
from fraud_risk.cost.cost_model import select_optimal_threshold
from fraud_risk.artifacts import save_bundle


def main():
    cfg = train_config()
    cost_cfg = cost_config()

    df = load_raw_transactions()
    split = chronological_split(
        df,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
        min_test_frauds=cfg["split"]["min_test_frauds"],
    )
    assert_no_time_leakage(split)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    split.train.to_parquet(DATA_PROCESSED / "train.parquet")
    split.val.to_parquet(DATA_PROCESSED / "val.parquet")
    split.test.to_parquet(DATA_PROCESSED / "test.parquet")

    X_train, y_train = split.train[FEATURE_COLUMNS], split.train[LABEL_COLUMN]
    X_val, y_val = split.val[FEATURE_COLUMNS], split.val[LABEL_COLUMN]

    print(
        f"train={len(X_train)} ({int(y_train.sum())} fraud) | "
        f"val={len(X_val)} ({int(y_val.sum())} fraud) | "
        f"test={len(split.test)} ({int(split.test[LABEL_COLUMN].sum())} fraud)"
    )

    print("Fitting baseline (LogisticRegression)...")
    baseline = build_baseline_pipeline(cfg["logistic_baseline"])
    baseline.fit(X_train, y_train)

    print("Fitting champion (LightGBM, early stopping on val)...")
    champion = fit_champion_pipeline(cfg["lightgbm_champion"], X_train, y_train, X_val, y_val)

    print("Calibrating champion on val...")
    calibrated_champion = calibrate_on_validation(
        champion, X_val, y_val, method=cfg["calibration"]["method"]
    )

    val_scores = calibrated_champion.predict_proba(X_val)[:, 1]
    best_t, best_cost, _ = select_optimal_threshold(
        y_val.values, X_val["Amount"].values, val_scores, cost_cfg
    )
    print(f"Selected threshold={best_t:.4f} (val expected cost=${best_cost:,.2f})")

    metadata = save_bundle(
        calibrated_champion=calibrated_champion,
        baseline_pipeline=baseline,
        threshold=best_t,
        metadata_fields=dict(
            train_rows=len(split.train),
            val_rows=len(split.val),
            test_rows=len(split.test),
            train_frauds=int(y_train.sum()),
            val_frauds=int(y_val.sum()),
            test_frauds=int(split.test[LABEL_COLUMN].sum()),
            feature_columns=FEATURE_COLUMNS,
        ),
    )
    print(f"Saved artifacts, model_version={metadata.model_version}")


if __name__ == "__main__":
    main()
