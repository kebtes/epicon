"""
---------------------------------------------------------
REGRESSION METRICS
---------------------------------------------------------

Standard metrics for evaluating regression models.
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Compute the mean squared error (MSE).

    Parameters:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: MSE value.

    Examples:
        >>> from nnf.metrics import mean_squared_error
        >>> mean_squared_error([1, 2, 3], [1.1, 1.9, 3.2])
        0.02
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """
    Compute the mean absolute error (MAE).

    Parameters:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: MAE value.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Compute the coefficient of determination R^2.

    Parameters:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: R^2 score. Best possible value is 1.0.

    Examples:
        >>> from nnf.metrics import r2_score
        >>> r2_score([1, 2, 3], [1, 2, 3])
        1.0
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + 1e-15))
