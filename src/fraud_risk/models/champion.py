import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from fraud_risk.features.engineer import FeatureEngineer, CHAMPION_COLUMNS


def _sklearn_pr_auc_eval(y_true, y_pred):
    """Custom eval metric for LightGBM's early stopping.

    LightGBM's built-in string metric name "average_precision" does NOT
    match sklearn's average_precision_score under scale_pos_weight -- an
    empirically-confirmed discrepancy (see CHALLENGES.md) that silently
    picked a badly-overfit early-stopping iteration in an earlier version
    of this code (reported val "average_precision" of 0.65 while the
    actual sklearn PR-AUC on the same predictions was 0.07). Using this
    explicit callable guarantees early stopping optimizes the exact metric
    this project reports and selects a threshold against everywhere else.
    """
    return "sklearn_pr_auc", average_precision_score(y_true, y_pred), True


def fit_champion_pipeline(
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Pipeline:
    """Fit LightGBM with early stopping on VALIDATION.

    scale_pos_weight is a configured value (configs/train_config.yaml), not
    computed from the train imbalance ratio -- see that file's comment for
    why: a VAL-only hyperparameter search found the textbook auto-computed
    weight (~447x for this split) caused severe overfitting that a
    class-imbalance-naive alternative did not.
    """
    fe = FeatureEngineer(CHAMPION_COLUMNS)
    fe.fit(X_train)
    X_train_t = fe.transform(X_train)
    X_val_t = fe.transform(X_val)

    model = lgb.LGBMClassifier(
        n_estimators=config.get("n_estimators", 500),
        learning_rate=config.get("learning_rate", 0.02),
        num_leaves=config.get("num_leaves", 15),
        max_depth=config.get("max_depth", -1),
        min_child_samples=config.get("min_child_samples", 30),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        reg_alpha=config.get("reg_alpha", 0.5),
        reg_lambda=config.get("reg_lambda", 0.5),
        scale_pos_weight=config.get("scale_pos_weight", 1.0),
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train_t,
        y_train,
        eval_set=[(X_val_t, y_val)],
        eval_metric=_sklearn_pr_auc_eval,
        callbacks=[lgb.early_stopping(config.get("early_stopping_rounds", 50), verbose=False)],
    )

    return Pipeline(steps=[("features", fe), ("model", model)])
