"""
---------------------------------------------------------
NUMBA JIT COMPILED KERNELS FOR PERFORMANCE-CRITICAL LOOPS
---------------------------------------------------------

Optional Numba acceleration for tree-based models, KNN,
and other algorithms with non-vectorizable loops.
If Numba is not installed, pure Python fallbacks are used.
"""

import numpy as np

try:
    import numba as nb

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None


def _jit_if_available(func=None, *, nopython=True, parallel=False, **kwargs):
    """
    Decorator that applies Numba JIT only if Numba is installed.
    Can be used with or without arguments.

    Usage:
        @_jit_if_available
        def my_func(x): ...

        @_jit_if_available(nopython=True, parallel=True)
        def my_func(x): ...
    """

    def decorator(f):
        if HAS_NUMBA:
            return nb.jit(f, nopython=nopython, parallel=parallel, **kwargs)
        return f

    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# Tree split helpers
# ---------------------------------------------------------------------------


@_jit_if_available
def _gini_impurity(left_counts, right_counts, total_left, total_right, total_samples):
    """
    Compute weighted Gini impurity for a binary split.

    Args:
        left_counts (np.ndarray): Class counts in left child.
        right_counts (np.ndarray): Class counts in right child.
        total_left (int): Total samples in left child.
        total_right (int): Total samples in right child.
        total_samples (int): Total samples in parent.

    Returns:
        float: Weighted Gini impurity of the split.
    """
    if total_left == 0 or total_right == 0:
        return 0.0

    left_gini = 1.0 - np.sum((left_counts / total_left) ** 2)
    right_gini = 1.0 - np.sum((right_counts / total_right) ** 2)

    return (total_left / total_samples) * left_gini + (total_right / total_samples) * right_gini


@_jit_if_available
def _entropy_impurity(left_counts, right_counts, total_left, total_right, total_samples):
    """
    Compute weighted entropy for a binary split.

    Args:
        left_counts (np.ndarray): Class counts in left child.
        right_counts (np.ndarray): Class counts in right child.
        total_left (int): Total samples in left child.
        total_right (int): Total samples in right child.
        total_samples (int): Total samples in parent.

    Returns:
        float: Weighted entropy of the split.
    """
    if total_left == 0 or total_right == 0:
        return 0.0

    def _entropy(counts, total):
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    left_ent = _entropy(left_counts, total_left)
    right_ent = _entropy(right_counts, total_right)

    return (total_left / total_samples) * left_ent + (total_right / total_samples) * right_ent


@_jit_if_available
def _mse_split(left_y, right_y):
    """
    Compute weighted MSE for a binary split.

    Args:
        left_y (np.ndarray): Target values in left child.
        right_y (np.ndarray): Target values in right child.

    Returns:
        float: Weighted MSE of the split.
    """
    total = len(left_y) + len(right_y)
    if total == 0:
        return 0.0

    left_mse = np.var(left_y) * len(left_y) if len(left_y) > 0 else 0.0
    right_mse = np.var(right_y) * len(right_y) if len(right_y) > 0 else 0.0

    return (left_mse + right_mse) / total


@_jit_if_available
def _best_split_numeric(X_column, y, num_classes, criterion, min_samples_split, min_samples_leaf):
    """
    Find the best threshold for a numeric feature.

    Args:
        X_column (np.ndarray): Feature values (sorted).
        y (np.ndarray): Target values.
        num_classes (int): Number of classes (1 for regression).
        criterion (str): 'gini', 'entropy', or 'mse'.
        min_samples_split (int): Minimum samples to split.
        min_samples_leaf (int): Minimum samples per leaf.

    Returns:
        tuple: (best_threshold, best_impurity) or (None, inf).
    """
    n = len(X_column)
    best_threshold = None
    best_score = np.inf

    if n < min_samples_split:
        return None, np.inf

    # Unique thresholds at midpoints between sorted values
    unique_vals = np.unique(X_column)
    if len(unique_vals) == 1:
        return None, np.inf

    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

    for threshold in thresholds:
        left_mask = X_column <= threshold
        right_mask = ~left_mask

        left_count = np.sum(left_mask)
        right_count = np.sum(right_mask)

        if left_count < min_samples_leaf or right_count < min_samples_leaf:
            continue

        if criterion == "mse":
            score = _mse_split(y[left_mask], y[right_mask])
        elif criterion in ("gini", "entropy"):
            left_y = y[left_mask].astype(np.int64)
            right_y = y[right_mask].astype(np.int64)

            left_counts = np.zeros(num_classes, dtype=np.float64)
            right_counts = np.zeros(num_classes, dtype=np.float64)

            for c in range(num_classes):
                left_counts[c] = np.sum(left_y == c)
                right_counts[c] = np.sum(right_y == c)

            if criterion == "gini":
                score = _gini_impurity(left_counts, right_counts, left_count, right_count, n)
            else:
                score = _entropy_impurity(left_counts, right_counts, left_count, right_count, n)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        if score < best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


# ---------------------------------------------------------------------------
# KNN distance helpers
# ---------------------------------------------------------------------------


@_jit_if_available
def _euclidean_distance(x1, x2):
    """
    Compute Euclidean distance between two vectors.
    """
    diff = x1 - x2
    return np.sqrt(np.dot(diff, diff))


@_jit_if_available
def _manhattan_distance(x1, x2):
    """
    Compute Manhattan distance between two vectors.
    """
    return np.sum(np.abs(x1 - x2))


@_jit_if_available
def _knn_predict_single(x_test, X_train, y_train, k, metric="euclidean"):
    """
    Predict label for a single test point using KNN.

    Args:
        x_test (np.ndarray): Single test sample.
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        k (int): Number of neighbors.
        metric (str): 'euclidean' or 'manhattan'.

    Returns:
        float: Predicted label (for classification, returns the majority class).
    """
    n_train = X_train.shape[0]
    distances = np.zeros(n_train)

    for i in range(n_train):
        if metric == "euclidean":
            distances[i] = _euclidean_distance(x_test, X_train[i])
        else:
            distances[i] = _manhattan_distance(x_test, X_train[i])

    # Get indices of k smallest distances
    indices = np.argsort(distances)[:k]
    k_nearest = y_train[indices]

    # For regression, return mean; for classification, return mode
    if np.issubdtype(y_train.dtype, np.floating) or len(np.unique(y_train)) > 20:
        return np.mean(k_nearest)
    else:
        # Return mode (most common)
        counts = np.bincount(k_nearest.astype(np.int64))
        return np.argmax(counts)
