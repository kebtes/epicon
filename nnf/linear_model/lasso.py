"""
---------------------------------------------------------
LASSO REGRESSION MODEL
---------------------------------------------------------

Linear Regression with L1 regularization (Lasso). Adds a
penalty proportional to the absolute magnitude of the
coefficients, which can drive some coefficients to exactly
zero, performing feature selection.

Objective:
    min_w 0.5 * ||Xw - y||^2 + alpha * ||w||_1

Uses coordinate descent with soft-thresholding for optimization.
"""

import numpy as np


class Lasso:
    """
    Linear Regression with L1 regularization (Lasso).

    Parameters:
        alpha (float): Regularization strength. Must be >= 0.
                       Higher values mean stronger regularization
                       and more coefficients driven to zero.
                       Defaults to 1.0.
        fit_intercept (bool): Whether to fit an intercept term.
                              Defaults to True.
        max_iter (int): Maximum number of coordinate descent iterations.
                        Defaults to 1000.
        tol (float): Convergence tolerance. Defaults to 1e-4.

    Attributes:
        coef_ (np.ndarray): Learned coefficients, shape (n_features,).
        intercept_ (float): Learned intercept term.

    Examples:
        >>> import numpy as np
        >>> from nnf.linear_model import Lasso
        >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        >>> y = np.array([6, 8, 9, 11])
        >>> model = Lasso(alpha=0.01)
        >>> model.fit(X, y)
        >>> model.predict(np.array([[3, 5]]))
        array([16.])
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0

    def _soft_threshold(self, x, threshold):
        """
        Soft-thresholding operator used in Lasso coordinate descent.

            S(x, gamma) = sign(x) * max(|x| - gamma, 0)
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

    def fit(self, X, y):
        """
        Fit the Lasso model using coordinate descent.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,) or (n_samples, 1).

        Returns:
            self: The fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape

        # For alpha = 0, use closed-form OLS
        if self.alpha == 0.0:
            if self.fit_intercept:
                X_aug = np.c_[np.ones(n_samples), X]
            else:
                X_aug = X

            try:
                w_full = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                w_full = np.linalg.pinv(X_aug) @ y

            if self.fit_intercept:
                self.intercept_ = w_full[0]
                self.coef_ = w_full[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = w_full

            return self

        # Center the data if fitting intercept
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
            self.intercept_ = y_mean
        else:
            X_centered = X
            y_centered = y

        # Initialize coefficients
        w = np.zeros(n_features)

        # Coordinate descent
        for iteration in range(self.max_iter):
            w_prev = w.copy()

            for j in range(n_features):
                # Compute partial residual (remove effect of feature j)
                residual = y_centered - (X_centered @ w - X_centered[:, j] * w[j])
                rho = X_centered[:, j] @ residual

                # Update coefficient using soft-thresholding
                z = X_centered[:, j] @ X_centered[:, j]
                if z == 0:
                    continue

                w[j] = self._soft_threshold(rho, self.alpha * n_samples) / z

            # Check convergence
            if np.max(np.abs(w - w_prev)) < self.tol:
                break

        self.coef_ = w

        if not self.fit_intercept:
            self.intercept_ = 0.0

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
