"""
---------------------------------------------------------
DECISION TREE CLASSIFIER
---------------------------------------------------------

A non-parametric supervised learning method that builds a
tree-like model of decisions. The tree splits the feature
space into regions, assigning the majority class in each
region as the prediction.

Supports both Gini impurity and information gain (entropy)
as splitting criteria.
"""

import numpy as np
from epicon._jit import _best_split_numeric


class _Node:
    """
    Internal tree node representation.

    Attributes:
        feature_idx (int): Index of the feature to split on.
        threshold (float): Threshold value for the split.
        left (_Node): Left child node.
        right (_Node): Right child node.
        value (any): Predicted value (class) for leaf nodes.
        is_leaf (bool): Whether this node is a leaf.
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf


class DecisionTreeClassifier:
    """
    Decision Tree classifier supporting Gini and Entropy criteria.

    Parameters:
        criterion (str): Split quality measure. One of 'gini' or
                         'entropy'. Defaults to 'gini'.
        max_depth (int or None): Maximum depth of the tree. If None,
                                 nodes are expanded until all leaves
                                 are pure or contain fewer samples than
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
        classes_ (np.ndarray): Unique class labels.
        n_classes_ (int): Number of classes.
        feature_importances_ (np.ndarray): Feature importance scores.
        tree_ (_Node): The root node of the fitted tree.

    Examples:
        >>> import numpy as np
        >>> from epicon.tree import DecisionTreeClassifier
        >>> X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        >>> y = np.array([0, 0, 1, 1])
        >>> model = DecisionTreeClassifier(max_depth=3)
        >>> model.fit(X, y)
        >>> model.predict(np.array([[0, 0], [1, 1]]))
        array([0, 0])
    """

    def __init__(
        self,
        criterion: str = 'gini',
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int = None,
        random_state: int = None,
    ):
        if criterion not in ('gini', 'entropy'):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got '{criterion}'")
        self.criterion = criterion
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.classes_ = None
        self.n_classes_ = None
        self.feature_importances_ = None
        self.tree_ = None
        self._n_features = None

    def fit(self, X, y):
        """
        Build a decision tree classifier from the training data.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Returns:
            self: The fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self._n_features)

        # Map original labels to {0, ..., n_classes-1} for internal use
        y_mapped = np.searchsorted(self.classes_, y).astype(np.float64)

        self.tree_ = self._grow_tree(X, y_mapped, depth=0)

        # Normalize feature importances
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance

        return self

    def _grow_tree(self, X, y, depth):
        """
        Recursively grow the decision tree.

        Parameters:
            X (np.ndarray): Data at the current node.
            y (np.ndarray): Labels at the current node (mapped to 0..n_classes-1).
            depth (int): Current depth in the tree.

        Returns:
            _Node: The root node of the subtree.
        """
        n_samples, n_features = X.shape
        n_classes = self.n_classes_

        # Check stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return _Node(value=self._most_common_class(y), is_leaf=True)

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

            threshold, score = _best_split_numeric(
                X_sorted, y_sorted, n_classes, self.criterion,
                self.min_samples_split, self.min_samples_leaf
            )

            if score < best_score:
                best_score = score
                best_feature = feature_idx
                best_threshold = threshold

        if best_feature is None or best_threshold is None:
            return _Node(value=self._most_common_class(y), is_leaf=True)

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Track feature importance (proportional to impurity reduction)
        parent_impurity = self._compute_impurity(y)
        left_impurity = self._compute_impurity(y[left_mask])
        right_impurity = self._compute_impurity(y[right_mask])
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        weighted_impurity = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity
        self.feature_importances_[best_feature] += (parent_impurity - weighted_impurity) * n_samples

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(feature_idx=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _compute_impurity(self, y):
        """
        Compute the impurity of a set of labels.

        Parameters:
            y (np.ndarray): Labels (mapped to 0..n_classes-1).

        Returns:
            float: Impurity value.
        """
        if len(y) == 0:
            return 0.0

        counts = np.bincount(y.astype(np.int64), minlength=self.n_classes_)
        probs = counts / len(y)
        probs = probs[probs > 0]

        if self.criterion == 'gini':
            return 1.0 - np.sum(probs ** 2)
        else:
            return -np.sum(probs * np.log2(probs))

    def _most_common_class(self, y):
        """
        Return the most common class in a set of labels.

        Parameters:
            y (np.ndarray): Labels.

        Returns:
            The most frequent label.
        """
        counts = np.bincount(y.astype(np.int64), minlength=self.n_classes_)
        return np.argmax(counts)

    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        y_internal = np.array([self._traverse(x, self.tree_) for x in X])
        return self.classes_[y_internal.astype(np.int64)]

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        Returns the proportion of samples of each class in the leaf
        node that each sample falls into.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability estimates of shape (n_samples, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        probas = np.zeros((X.shape[0], self.n_classes_))

        for i in range(X.shape[0]):
            leaf = self._find_leaf(X[i], self.tree_)
            # We stored the class distribution in the leaf value
            # (best approximation: class with count 1 at that class index)
            if leaf.is_leaf:
                probas[i, int(leaf.value)] = 1.0

        return probas

    def _traverse(self, x, node):
        """
        Traverse the tree to make a prediction for a single sample.

        Parameters:
            x (np.ndarray): A single sample.
            node (_Node): Current tree node.

        Returns:
            Predicted class index.
        """
        if node.is_leaf:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

    def _find_leaf(self, x, node):
        """
        Find the leaf node that a sample falls into.

        Parameters:
            x (np.ndarray): A single sample.
            node (_Node): Current tree node.

        Returns:
            _Node: The leaf node.
        """
        if node.is_leaf:
            return node

        if x[node.feature_idx] <= node.threshold:
            return self._find_leaf(x, node.left)
        else:
            return self._find_leaf(x, node.right)

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
