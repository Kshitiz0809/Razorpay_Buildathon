import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


def calibrate_on_validation(
    fitted_pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Fit a calibrator on VALIDATION only, wrapping an already-fitted pipeline.

    Class-weighted GBM output is a decision-relevant score but is not a
    true probability -- it must be calibrated before it can feed a
    dollar-cost formula. Sigmoid (Platt) is used over isotonic because
    isotonic is nonparametric and needs more minority-class points per bin
    than the validation block's ~100-150 frauds can reliably supply.
    """
    calibrator = CalibratedClassifierCV(fitted_pipeline, method=method, cv="prefit")
    calibrator.fit(X_val, y_val)
    return calibrator
