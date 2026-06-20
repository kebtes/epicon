"""
---------------------------------------------------------
GAUSSIAN NAIVE BAYES CLASSIFIER
---------------------------------------------------------

A probabilistic classifier based on Bayes' theorem with the
"naive" assumption of conditional independence between features.

For Gaussian Naive Bayes, the likelihood of each feature
given a class is assumed to follow a normal (Gaussian)
distribution.

Formula:
    P(y | X) ∝ P(y) * ∏ P(x_i | y)
    P(x_i | y) = N(x_i | μ_yi, σ_yi²)
"""

import numpy as np


class GaussianNB:
    """
    Gaussian Naive Bayes classifier.

    Parameters:
        var_smoothing (float): Portion of the largest variance of all
                               features added to variances for numerical
                               stability. Defaults to 1e-9.

    Attributes:
        classes_ (np.ndarray): Unique class labels.
        class_prior_ (np.ndarray): Prior probability of each class.
        theta_ (np.ndarray): Mean of each feature per class,
                             shape (n_classes, n_features).
        var_ (np.ndarray): Variance of each feature per class,
                           shape (n_classes, n_features).

    Examples:
        >>> import numpy as np
        >>> from epicon.naive_bayes import GaussianNB
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> y = np.array([0, 0, 0, 1, 1, 1])
        >>> model = GaussianNB()
        >>> model.fit(X, y)
        >>> model.predict(np.array([[-0.8, -1]]))
        array([0])
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None
        self.var_ = None

    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model to the training data.

        Computes the mean, variance, and prior probability for each
        class based on the training data.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Returns:
            self: The fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[i] = np.mean(X_c, axis=0)
            self.var_[i] = np.var(X_c, axis=0)
            self.class_prior_[i] = X_c.shape[0] / X.shape[0]

        # Add smoothing to variances for numerical stability
        self.var_ += self.var_smoothing * np.max(self.var_, axis=1, keepdims=True)

        return self

    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        joint_log_likelihood = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability estimates of shape (n_samples, n_classes).
        """
        joint_log_likelihood = self._joint_log_likelihood(X)

        # Convert log-likelihood to probabilities using softmax
        log_proba = joint_log_likelihood - np.max(joint_log_likelihood, axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= np.sum(proba, axis=1, keepdims=True)

        return proba

    def _joint_log_likelihood(self, X):
        """
        Compute the joint log likelihood of each sample for each class.

        log(P(y)) + sum(log(P(x_i | y)))

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Joint log likelihood of shape (n_samples, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        joint_log_likelihood = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # log(P(y))
            joint_log_likelihood[:, i] = np.log(self.class_prior_[i] + 1e-15)

            # sum(log(P(x_i | y))) for each feature
            # log of Gaussian PDF: -0.5 * log(2 * pi * var) - (x - mean)^2 / (2 * var)
            diff = X - self.theta_[i]
            var = self.var_[i]
            log_likelihood = -0.5 * np.log(2 * np.pi * var + 1e-15) - (diff ** 2) / (2 * var + 1e-15)
            joint_log_likelihood[:, i] += np.sum(log_likelihood, axis=1)

        return joint_log_likelihood

    def score(self, X, y):
        """
        Calculate the mean accuracy on test data.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): True labels of shape (n_samples,).

        Returns:
            float: Accuracy score.
        """
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
