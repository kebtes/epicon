"""
---------------------------------------------------------
LOGISTIC REGRESSION MODEL
---------------------------------------------------------

Binary classification model that models class probabilities
using a logistic (sigmoid) function. The model is trained by
minimizing the binary cross-entropy loss via gradient descent.

Formula:
    p(y=1 | X) = sigmoid(X @ w + b)
    sigmoid(z) = 1 / (1 + exp(-z))

The decision boundary is at p = 0.5.
"""

import numpy as np


class LogisticRegression:
    """
    Binary Logistic Regression classifier.

    Uses gradient descent to minimize binary cross-entropy loss
    and predict binary outcomes (0 or 1).

    Parameters:
        fit_intercept (bool): Whether to fit an intercept term.
                              Defaults to True.
        learning_rate (float): Step size for gradient descent. Defaults to 0.01.
        epochs (int): Number of passes over the data. Defaults to 1000.
        tol (float): Convergence tolerance. Defaults to 1e-8.
        C (float): Inverse regularization strength (smaller = stronger).
                   1/C is the regularization coefficient. Defaults to 1.0.
        penalty (str or None): Regularization type ('l1', 'l2', or None).
                               Defaults to None.

    Attributes:
        coef_ (np.ndarray): Learned coefficients, shape (n_features,).
        intercept_ (float): Learned intercept term.

    Examples:
        >>> import numpy as np
        >>> from epicon.linear_model import LogisticRegression
        >>> X = np.array([[1], [2], [3], [4]])
        >>> y = np.array([0, 0, 1, 1])
        >>> model = LogisticRegression()
        >>> model.fit(X, y)
        >>> model.predict(np.array([[1.5], [3.5]]))
        array([0, 1])
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        tol: float = 1e-8,
        C: float = 1.0,
        penalty: str = None,
    ):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.C = C
        self.penalty = penalty

        self.coef_ = None
        self.intercept_ = 0.0
        self._classes = None

    def _sigmoid(self, z):
        """
        Compute the sigmoid function with numerical stability clipping.
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Binary target values of shape (n_samples,)
                            with values in {0, 1}.

        Returns:
            self: The fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self._classes = np.unique(y)
        if len(self._classes) > 2:
            raise ValueError(
                f"LogisticRegression supports binary classification only, "
                f"but found {len(self._classes)} classes: {self._classes}"
            )

        # Map labels to {0, 1}
        y = np.where(y == self._classes[0], 0, 1).astype(np.float64)

        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_aug = np.c_[np.ones(n_samples), X]
            n_params = n_features + 1
        else:
            X_aug = X
            n_params = n_features

        w = np.zeros(n_params)
        prev_loss = np.inf

        for _epoch in range(self.epochs):
            logits = X_aug @ w
            y_pred = self._sigmoid(logits)

            # Binary cross-entropy loss
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

            gradient = (1.0 / n_samples) * (X_aug.T @ (y_pred - y))

            # Regularization (skip bias term if fit_intercept)
            if self.penalty == "l2":
                reg_start = 1 if self.fit_intercept else 0
                gradient[reg_start:] += (1.0 / self.C) * w[reg_start:] / n_samples
            elif self.penalty == "l1":
                reg_start = 1 if self.fit_intercept else 0
                gradient[reg_start:] += (1.0 / self.C) * np.sign(w[reg_start:]) / n_samples

            w -= self.learning_rate * gradient

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability estimates of shape (n_samples, 2).
                        Column 0 is P(y=0 | X), column 1 is P(y=1 | X).
        """
        X = np.asarray(X, dtype=np.float64)
        logits = X @ self.coef_ + self.intercept_
        proba_pos = self._sigmoid(logits)
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        """
        Calculate the mean accuracy on the given test data and labels.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): True labels of shape (n_samples,).

        Returns:
            float: Accuracy score.
        """
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
