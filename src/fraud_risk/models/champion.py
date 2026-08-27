import lightgbm as lgb
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud_risk.features.engineer import FeatureEngineer, CHAMPION_COLUMNS


def fit_champion_pipeline(
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Pipeline:
    """Fit LightGBM with class-weighting and early stopping on VALIDATION.

    scale_pos_weight is computed from TRAIN only (never from val/test) so
    the imbalance-correction itself can't leak information about held-out data.
    """
    fe = FeatureEngineer(CHAMPION_COLUMNS)
    fe.fit(X_train)
    X_train_t = fe.transform(X_train)
    X_val_t = fe.transform(X_val)

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos

    model = lgb.LGBMClassifier(
        n_estimators=config.get("n_estimators", 500),
        learning_rate=config.get("learning_rate", 0.05),
        num_leaves=config.get("num_leaves", 31),
        max_depth=config.get("max_depth", -1),
        min_child_samples=config.get("min_child_samples", 20),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        reg_alpha=config.get("reg_alpha", 0.0),
        reg_lambda=config.get("reg_lambda", 0.0),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train_t,
        y_train,
        eval_set=[(X_val_t, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(config.get("early_stopping_rounds", 50), verbose=False)],
    )

    return Pipeline(steps=[("features", fe), ("model", model)])
