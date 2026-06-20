"""
---------------------------------------------------------
LABEL & ONE-HOT ENCODING UTILITIES
---------------------------------------------------------

LabelEncoder:   Encode categorical labels as integers (0 to n_classes-1).

OneHotEncoder:  Encode categorical features as a one-hot numeric array.
"""

import numpy as np


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1.

    Attributes:
        classes_ (np.ndarray): The unique classes seen during fit.

    Examples:
        >>> from nnf.preprocessing import LabelEncoder
        >>> encoder = LabelEncoder()
        >>> encoder.fit_transform(['cat', 'dog', 'bird', 'dog'])
        array([2, 1, 0, 1])
        >>> encoder.inverse_transform([0, 1, 2])
        array(['bird', 'dog', 'cat'], dtype='<U4')
    """

    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        """
        Fit the encoder by finding unique classes.

        Parameters:
            y (array-like): Target values of shape (n_samples,).

        Returns:
            self: The fitted encoder.
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        """
        Transform labels to normalized encoding.

        Parameters:
            y (array-like): Target values of shape (n_samples,).

        Returns:
            np.ndarray: Encoded labels of shape (n_samples,).
        """
        y = np.asarray(y)
        indices = np.searchsorted(self.classes_, y)
        return indices.astype(np.int64)

    def fit_transform(self, y):
        """
        Fit the encoder and then transform the labels.

        Parameters:
            y (array-like): Target values of shape (n_samples,).

        Returns:
            np.ndarray: Encoded labels of shape (n_samples,).
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        """
        Transform labels back to original encoding.

        Parameters:
            y (array-like): Encoded labels of shape (n_samples,).

        Returns:
            np.ndarray: Original labels.
        """
        y = np.asarray(y).ravel().astype(np.int64)
        return self.classes_[y]


class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.

    Attributes:
        categories_ (list): The unique categories per feature seen during fit.

    Examples:
        >>> import numpy as np
        >>> from nnf.preprocessing import OneHotEncoder
        >>> encoder = OneHotEncoder()
        >>> encoder.fit_transform(np.array([['cat'], ['dog'], ['bird']]))
        array([[0., 1., 0.],
               [0., 0., 1.],
               [1., 0., 0.]])
    """

    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        """
        Fit the encoder by finding unique categories.

        Parameters:
            X (array-like): Data of shape (n_samples, n_features).

        Returns:
            self: The fitted encoder.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.categories_ = []
        for i in range(X.shape[1]):
            self.categories_.append(np.unique(X[:, i]))

        return self

    def transform(self, X):
        """
        Transform data to one-hot encoding.

        Parameters:
            X (array-like): Data of shape (n_samples, n_features).

        Returns:
            np.ndarray: One-hot encoded data.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = []
        for i in range(X.shape[1]):
            cat = self.categories_[i]
            mapping = {v: j for j, v in enumerate(cat)}
            col = np.array([mapping.get(x, -1) for x in X[:, i]])
            one_hot = np.zeros((len(col), len(cat)))
            valid = col >= 0
            one_hot[valid, col[valid]] = 1.0
            result.append(one_hot)

        return np.hstack(result)

    def fit_transform(self, X):
        """
        Fit the encoder and then transform the data.

        Parameters:
            X (array-like): Data of shape (n_samples, n_features).

        Returns:
            np.ndarray: One-hot encoded data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """
        Convert one-hot encoded data back to original labels.

        Parameters:
            X (array-like): One-hot encoded data.

        Returns:
            np.ndarray: Original labels.
        """
        X = np.asarray(X)
        result = []
        start = 0
        for cat in self.categories_:
            end = start + len(cat)
            col = X[:, start:end]
            indices = np.argmax(col, axis=1)
            result.append(np.array([cat[i] for i in indices]))
            start = end

        return np.column_stack(result)
