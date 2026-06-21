"""
---------------------------------------------------------
K-NEAREST NEIGHBORS REGRESSOR
---------------------------------------------------------

A non-parametric regression algorithm that predicts
the target value of a sample as the mean (or weighted mean)
of the target values of its k nearest neighbors.
"""

import numpy as np


class KNeighborsRegressor:
    """
    K-Nearest Neighbors regressor.

    Parameters:
        n_neighbors (int): Number of neighbors to use. Defaults to 5.
        metric (str): Distance metric to use. One of 'euclidean' or
                      'manhattan'. Defaults to 'euclidean'.
        weights (str): Weight function used in prediction. 'uniform' for
                       equal weights, 'distance' for inverse distance weighting.
                       Defaults to 'uniform'.

    Attributes:
        X_train (np.ndarray): Stored training data.
        y_train (np.ndarray): Stored training targets.

    Examples:
        >>> import numpy as np
        >>> from epicon.neighbors import KNeighborsRegressor
        >>> X = np.array([[1], [2], [3], [4]])
        >>> y = np.array([1.5, 2.0, 3.5, 4.0])
        >>> model = KNeighborsRegressor(n_neighbors=3)
        >>> model.fit(X, y)
        >>> model.predict(np.array([[2.5]]))
        array([2.666...])
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean", weights: str = "uniform"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data for lazy inference.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).

        Returns:
            self: The fitted model.
        """
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64).ravel()
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
        predictions = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            predictions[i] = self._predict_single(X[i])

        return predictions

    def _predict_single(self, x):
        """
        Predict the target value for a single sample.

        Parameters:
            x (np.ndarray): A single test sample.

        Returns:
            float: Predicted value.
        """
        if self.metric == "euclidean":
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.metric == "manhattan":
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError(f"Unknown metric '{self.metric}'. Use 'euclidean' or 'manhattan'.")

        nearest_indices = np.argsort(distances)[: self.n_neighbors]
        nearest_values = self.y_train[nearest_indices]

        if self.weights == "uniform":
            return np.mean(nearest_values)
        elif self.weights == "distance":
            k_distances = distances[nearest_indices] + 1e-15
            weights = 1.0 / k_distances
            return np.average(nearest_values, weights=weights)
        else:
            raise ValueError(f"Unknown weights '{self.weights}'. Use 'uniform' or 'distance'.")

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
