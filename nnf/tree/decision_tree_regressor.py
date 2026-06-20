"""
---------------------------------------------------------
DECISION TREE REGRESSOR
---------------------------------------------------------

A decision tree for regression tasks. The tree splits the
feature space into regions and assigns the mean target value
of the training samples in each region as the prediction.

Uses MSE (Mean Squared Error) or MAE (Mean Absolute Error)
as the splitting criterion.
"""

import numpy as np
from nnf._jit import _best_split_numeric, _mse_split


class _RegNode:
    """
    Internal tree node for regression.

    Attributes:
        feature_idx (int): Index of the feature to split on.
        threshold (float): Threshold value for the split.
        left (_RegNode): Left child node.
        right (_RegNode): Right child node.
        value (float): Predicted value for leaf nodes.
        is_leaf (bool): Whether this node is a leaf.
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf


class DecisionTreeRegressor:
    """
    Decision Tree regressor using MSE as the splitting criterion.

    Parameters:
        criterion (str): Split quality measure. One of 'mse' or
                         'mae'. Defaults to 'mse'.
        max_depth (int or None): Maximum depth of the tree. If None,
                                 nodes are expanded until all leaves
                                 contain fewer samples than
                                 min_samples_split. Defaults to None.
        min_samples_split (int): Minimum number of samples required to
                                 split an internal node. Defaults to 2.
        min_samples_leaf (int): Minimum number of samples required to
                                be at a leaf node. Defaults to 1.
        max_features (int or None): Number of features to consider when
                                    looking for the best split. If None,
                                    all features are used. Defaults to None.
        random_state (int or None): Controls the randomness of the estimator.
                                    Defaults to None.

    Attributes:
        tree_ (_RegNode): The root node of the fitted tree.
        feature_importances_ (np.ndarray): Feature importance scores.

    Examples:
        >>> import numpy as np
        >>> from nnf.tree import DecisionTreeRegressor
        >>> X = np.array([[1], [2], [3], [4]])
        >>> y = np.array([1.0, 2.0, 3.0, 4.0])
        >>> model = DecisionTreeRegressor(max_depth=3)
        >>> model.fit(X, y)
        >>> model.predict(np.array([[2.5]]))
        array([2.5])
    """

    def __init__(
        self,
        criterion: str = 'mse',
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int = None,
        random_state: int = None,
    ):
        if criterion not in ('mse', 'mae'):
            raise ValueError(f"criterion must be 'mse' or 'mae', got '{criterion}'")
        self.criterion = criterion
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tree_ = None
        self.feature_importances_ = None
        self._n_features = None

    def fit(self, X, y):
        """
        Build a decision tree regressor from the training data.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).

        Returns:
            self: The fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self._n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self._n_features)

        self.tree_ = self._grow_tree(X, y, 0)

        # Normalize feature importances
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance

        return self

    def _grow_tree(self, X, y, depth):
        """
        Recursively grow the regression tree.

        Parameters:
            X (np.ndarray): Data at the current node.
            y (np.ndarray): Target values at the current node.
            depth (int): Current depth in the tree.

        Returns:
            _RegNode: The root node of the subtree.
        """
        n_samples, n_features = X.shape

        # Check stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return _RegNode(value=np.mean(y), is_leaf=True)

        # Determine which features to consider
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        else:
            n_selected = min(self.max_features, n_features)
            feature_indices = np.random.choice(n_features, n_selected, replace=False)

        # Find the best split
        best_feature = None
        best_threshold = None
        best_score = np.inf

        for feature_idx in feature_indices:
            X_column = X[:, feature_idx]
            sort_idx = np.argsort(X_column)
            X_sorted = X_column[sort_idx]
            y_sorted = y[sort_idx]

            if self.criterion == 'mse':
                threshold, score = _best_split_numeric(
                    X_sorted, y_sorted, 1, 'mse',
                    self.min_samples_split, self.min_samples_leaf
                )
            else:
                # MAE criterion
                threshold, score = self._best_split_mae(X_sorted, y_sorted)

            if score < best_score:
                best_score = score
                best_feature = feature_idx
                best_threshold = threshold

        if best_feature is None or best_threshold is None:
            return _RegNode(value=np.mean(y), is_leaf=True)

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Track feature importance
        parent_var = np.var(y) * n_samples
        left_var = np.var(y[left_mask]) * np.sum(left_mask) if np.sum(left_mask) > 0 else 0
        right_var = np.var(y[right_mask]) * np.sum(right_mask) if np.sum(right_mask) > 0 else 0
        self.feature_importances_[best_feature] += parent_var - (left_var + right_var)

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _RegNode(feature_idx=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _best_split_mae(self, X_sorted, y_sorted):
        """
        Find the best threshold for MAE criterion.

        Parameters:
            X_sorted (np.ndarray): Sorted feature values.
            y_sorted (np.ndarray): Target values sorted by feature.

        Returns:
            tuple: (best_threshold, best_score).
        """
        n = len(X_sorted)
        best_threshold = None
        best_score = np.inf

        if n < self.min_samples_split:
            return None, np.inf

        unique_vals = np.unique(X_sorted)
        if len(unique_vals) == 1:
            return None, np.inf

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for threshold in thresholds:
            left_mask = X_sorted <= threshold
            right_mask = ~left_mask

            left_count = np.sum(left_mask)
            right_count = np.sum(right_mask)

            if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                continue

            left_mae = np.mean(np.abs(y_sorted[left_mask] - np.mean(y_sorted[left_mask])))
            right_mae = np.mean(np.abs(y_sorted[right_mask] - np.mean(y_sorted[right_mask])))

            score = (left_count / n) * left_mae + (right_count / n) * right_mae

            if score < best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score

    def predict(self, X):
        """
        Predict target values for input samples.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._traverse(x, self.tree_) for x in X])

    def _traverse(self, x, node):
        """
        Traverse the tree to make a prediction for a single sample.

        Parameters:
            x (np.ndarray): A single sample.
            node (_RegNode): Current tree node.

        Returns:
            float: Predicted value.
        """
        if node.is_leaf:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

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
