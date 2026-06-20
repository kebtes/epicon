"""
---------------------------------------------------------
FEATURE SCALING UTILITIES
---------------------------------------------------------

StandardScaler: Standardize features by removing the mean
                and scaling to unit variance.

MinMaxScaler:   Transform features by scaling each feature to
                a given range, typically [0, 1].
"""

import numpy as np


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample x is:
        z = (x - mean) / std

    Parameters:
        with_mean (bool): Whether to center the data. Defaults to True.
        with_std (bool): Whether to scale to unit variance. Defaults to True.

    Attributes:
        mean_ (np.ndarray): Per-feature mean.
        std_ (np.ndarray): Per-feature standard deviation.
        n_features_in_ (int): Number of features seen during fit.

    Examples:
        >>> import numpy as np
        >>> from nnf.preprocessing import StandardScaler
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> scaler = StandardScaler()
        >>> scaler.fit_transform(X)
        array([[-1.224..., -1.224...],
               [ 0.        ,  0.        ],
               [ 1.224...,  1.224...]])
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Compute the mean and standard deviation for each feature.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).

        Returns:
            self: The fitted scaler.
        """
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])

        if self.with_std:
            self.std_ = np.std(X, axis=0)
        else:
            self.std_ = np.ones(X.shape[1])

        # Prevent division by zero
        self.std_[self.std_ == 0] = 1.0

        return self

    def transform(self, X):
        """
        Perform standardization by centering and scaling.

        Parameters:
            X (np.ndarray): Data to transform of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data.
        """
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit to the data and then transform it.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """
        Scale back to the original representation.

        Parameters:
            X (np.ndarray): Data to inverse transform.

        Returns:
            np.ndarray: Original data representation.
        """
        X = np.asarray(X, dtype=np.float64)
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range.

    The transformation is:
        X_std = (X - X.min) / (X.max - X.min)
        X_scaled = X_std * (max - min) + min

    Parameters:
        feature_range (tuple): Desired range of transformed data.
                               Defaults to (0, 1).

    Attributes:
        min_ (np.ndarray): Per-feature minimum.
        scale_ (np.ndarray): Per-feature scaling factor (1 / range).
        data_min_ (np.ndarray): Per-feature minimum seen in the data.
        data_max_ (np.ndarray): Per-feature maximum seen in the data.

    Examples:
        >>> import numpy as np
        >>> from nnf.preprocessing import MinMaxScaler
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> scaler = MinMaxScaler()
        >>> scaler.fit_transform(X)
        array([[0., 0.],
               [0.5, 0.5],
               [1., 1.]])
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X):
        """
        Compute the minimum and maximum for each feature.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).

        Returns:
            self: The fitted scaler.
        """
        X = np.asarray(X, dtype=np.float64)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        data_range = self.data_max_ - self.data_min_

        # Prevent division by zero
        data_range[data_range == 0] = 1.0

        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_

        return self

    def transform(self, X):
        """
        Scale features to the specified range.

        Parameters:
            X (np.ndarray): Data to transform of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data.
        """
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.min_

    def fit_transform(self, X):
        """
        Fit to the data and then transform it.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """
        Scale back to the original representation.

        Parameters:
            X (np.ndarray): Data to inverse transform.

        Returns:
            np.ndarray: Original data representation.
        """
        X = np.asarray(X, dtype=np.float64)
        return (X - self.min_) / self.scale_
