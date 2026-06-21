"""
---------------------------------------------------------
K-NEAREST NEIGHBORS CLASSIFIER
---------------------------------------------------------

A non-parametric classification algorithm that predicts
the class of a sample based on the majority class among its
k nearest neighbors in the feature space.

The model stores the entire training set and performs
prediction by computing distances to all training points.
"""

import numpy as np


class KNeighborsClassifier:
    """
    K-Nearest Neighbors classifier.

    Parameters:
        n_neighbors (int): Number of neighbors to use. Defaults to 5.
        metric (str): Distance metric to use. One of 'euclidean' or
                      'manhattan'. Defaults to 'euclidean'.
        weights (str): Weight function used in prediction. 'uniform' for
                       equal weights, 'distance' for inverse distance weighting.
                       Defaults to 'uniform'.

    Attributes:
        X_train (np.ndarray): Stored training data.
        y_train (np.ndarray): Stored training labels.
        classes_ (np.ndarray): Unique class labels.

    Examples:
        >>> import numpy as np
        >>> from epicon.neighbors import KNeighborsClassifier
        >>> X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
        >>> y = np.array([0, 0, 1, 1])
        >>> model = KNeighborsClassifier(n_neighbors=3)
        >>> model.fit(X, y)
        >>> model.predict(np.array([[2, 2]]))
        array([0])
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean", weights: str = "uniform"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Store the training data for lazy inference.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Returns:
            self: The fitted model.
        """
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        predictions = np.zeros(X.shape[0], dtype=self.y_train.dtype)

        for i in range(X.shape[0]):
            predictions[i] = self._predict_single(X[i])

        return predictions

    def _predict_single(self, x):
        """
        Predict the class for a single sample.

        Parameters:
            x (np.ndarray): A single test sample.

        Returns:
            The predicted class label.
        """
        # Compute distances to all training points
        if self.metric == "euclidean":
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.metric == "manhattan":
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError(f"Unknown metric '{self.metric}'. Use 'euclidean' or 'manhattan'.")

        # Get k nearest neighbors
        nearest_indices = np.argsort(distances)[: self.n_neighbors]
        nearest_labels = self.y_train[nearest_indices]

        if self.weights == "uniform":
            # Majority vote
            counts = np.bincount(nearest_labels.astype(np.int64))
            return np.argmax(counts)
        elif self.weights == "distance":
            # Inverse distance weighting
            k_distances = distances[nearest_indices] + 1e-15
            weights = 1.0 / k_distances

            weighted_votes = np.zeros(len(self.classes_), dtype=np.float64)
            for label, weight in zip(nearest_labels, weights, strict=False):
                idx = np.where(self.classes_ == label)[0][0]
                weighted_votes[idx] += weight

            return self.classes_[np.argmax(weighted_votes)]
        else:
            raise ValueError(f"Unknown weights '{self.weights}'. Use 'uniform' or 'distance'.")

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability estimates of shape (n_samples, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        probas = np.zeros((X.shape[0], len(self.classes_)))
        len(self.classes_)

        for i in range(X.shape[0]):
            if self.metric == "euclidean":
                distances = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis=1))
            else:
                distances = np.sum(np.abs(self.X_train - X[i]), axis=1)

            nearest_indices = np.argsort(distances)[: self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]

            for label in nearest_labels:
                idx = np.where(self.classes_ == label)[0][0]
                probas[i, idx] += 1.0

            probas[i] /= self.n_neighbors

        return probas

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
