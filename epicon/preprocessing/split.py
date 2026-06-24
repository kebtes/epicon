"""
---------------------------------------------------------
TRAIN/TEST SPLIT UTILITY
---------------------------------------------------------

Split arrays into random train and test subsets.
"""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets.

    Parameters:
        X (array-like): Input data of shape (n_samples, n_features).
        y (array-like): Target values of shape (n_samples,).
        test_size (float or int): If float, should be between 0.0 and 1.0
                                  and represents the proportion of the dataset
                                  to include in the test split. If int,
                                  represents the absolute number of test samples.
                                  Defaults to 0.2.
        random_state (int or None): Controls the shuffling. Defaults to None.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)

    Examples:
        >>> import numpy as np
        >>> from epicon.preprocessing import train_test_split
        >>> X = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        >>> len(X_train)
        3
        >>> len(X_test)
        2
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples = len(X)

    if isinstance(test_size, float):
        n_test = int(np.ceil(n_samples * test_size))
    else:
        n_test = int(test_size)

    n_test = min(n_test, n_samples - 1)

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
