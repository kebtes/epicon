"""
---------------------------------------------------------
LINEAR REGRESSION MODEL
---------------------------------------------------------

Closed-form (Normal Equation) and Gradient Descent based
Linear Regression for predicting continuous target variables.

The model minimizes the sum of squared residuals between
the observed targets and the linear predictions.

Formula:
    y = X @ w + b

Where:
    - X is the input matrix of shape (n_samples, n_features)
    - w is the weight vector of shape (n_features,)
    - b is the bias term (scalar)
    - y is the predicted output of shape (n_samples,)
"""

import numpy as np


class LinearRegression:
    """
    Ordinary Least Squares Linear Regression.

    Supports both closed-form solution via the Normal Equation
    and iterative optimization via Gradient Descent for large
    feature spaces.

    Parameters:
        fit_intercept (bool): Whether to fit an intercept term.
                              Defaults to True.
        method (str): Solver to use. 'normal_eq' for the closed-form
                      solution, 'gd' for gradient descent. Defaults to 'normal_eq'.
        learning_rate (float): Step size for gradient descent. Defaults to 0.01.
        epochs (int): Number of passes over the data for gradient descent.
                      Defaults to 1000.
        tol (float): Tolerance for convergence. If the loss improvement is
                     below this threshold, training stops early. Defaults to 1e-8.

    Attributes:
        coef_ (np.ndarray): Learned coefficients (weights), shape (n_features,).
        intercept_ (float): Learned intercept term.

    Examples:
        >>> import numpy as np
        >>> from epicon.linear_model import LinearRegression
        >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        >>> y = np.dot(X, np.array([1, 2])) + 3
        >>> model = LinearRegression()
        >>> model.fit(X, y)
        >>> model.predict(np.array([[3, 5]]))
        array([16.])
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        method: str = "normal_eq",
        learning_rate: float = 0.01,
        epochs: int = 1000,
        tol: float = 1e-8,
    ):
        self.fit_intercept = fit_intercept
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol

        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,) or (n_samples, 1).

        Returns:
            self: The fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape

        if self.method == "normal_eq":
            self._fit_normal_equation(X, y)
        elif self.method == "gd":
            self._fit_gradient_descent(X, y, n_samples, n_features)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'normal_eq' or 'gd'.")

        return self

    def _fit_normal_equation(self, X, y):
        """
        Fit using the closed-form Normal Equation:
            w = (X^T X)^{-1} X^T y

        This provides an exact solution in O(n_features^3) time.
        """
        if self.fit_intercept:
            X_aug = np.c_[np.ones(X.shape[0]), X]
        else:
            X_aug = X

        # w = (X^T X)^{-1} X^T y
        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ y

        try:
            w = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            w = np.linalg.pinv(XtX) @ Xty

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

    def _fit_gradient_descent(self, X, y, n_samples, n_features):
        """
        Fit using batch gradient descent.

        Suitable for datasets with a very large number of features
        where the Normal Equation becomes computationally expensive.
        """
        if self.fit_intercept:
            X_aug = np.c_[np.ones(n_samples), X]
            n_features += 1
        else:
            X_aug = X

        w = np.zeros(n_features)
        prev_loss = np.inf

        for _epoch in range(self.epochs):
            y_pred = X_aug @ w
            error = y_pred - y
            loss = np.mean(error**2)

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

            gradient = (2.0 / n_samples) * (X_aug.T @ error)
            w -= self.learning_rate * gradient

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

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
            float: R^2 score. Best possible value is 1.0.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - (ss_res / (ss_tot + 1e-15))
