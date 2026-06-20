"""
---------------------------------------------------------
RIDGE REGRESSION MODEL
---------------------------------------------------------

Linear Regression with L2 regularization (Tikhonov
regularization). Adds a penalty proportional to the squared
magnitude of the coefficients to prevent overfitting.

Objective:
    min_w ||Xw - y||^2 + alpha * ||w||^2

Closed-form solution:
    w = (X^T X + alpha * I)^{-1} X^T y
"""

import numpy as np


class Ridge:
    """
    Linear Regression with L2 regularization.

    Parameters:
        alpha (float): Regularization strength. Must be >= 0.
                       Higher values mean stronger regularization.
                       Defaults to 1.0.
        fit_intercept (bool): Whether to fit an intercept term.
                              Defaults to True.

    Attributes:
        coef_ (np.ndarray): Learned coefficients, shape (n_features,).
        intercept_ (float): Learned intercept term.

    Examples:
        >>> import numpy as np
        >>> from epicon.linear_model import Ridge
        >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        >>> y = np.array([6, 8, 9, 11])
        >>> model = Ridge(alpha=0.1)
        >>> model.fit(X, y)
        >>> model.predict(np.array([[3, 5]]))
        array([16.])
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        """
        Fit the Ridge regression model using the closed-form solution.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,) or (n_samples, 1).

        Returns:
            self: The fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_aug = np.c_[np.ones(n_samples), X]
            n_params = n_features + 1
        else:
            X_aug = X
            n_params = n_features

        # Ridge closed-form: (X^T X + alpha * I)^{-1} X^T y
        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ y

        # Add regularization (don't regularize the bias term)
        reg_matrix = self.alpha * np.eye(n_params)
        if self.fit_intercept:
            reg_matrix[0, 0] = 0.0  # No regularization on intercept

        try:
            w = np.linalg.solve(XtX + reg_matrix, Xty)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(XtX + reg_matrix) @ Xty

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def predict(self, X):
        """
        Predict target values for input samples.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        Calculate the coefficient of determination R^2.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): True target values of shape (n_samples,).

        Returns:
            float: R^2 score.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - (ss_res / (ss_tot + 1e-15))
