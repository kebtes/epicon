"""
---------------------------------------------------------
RANDOM FOREST CLASSIFIER
---------------------------------------------------------

An ensemble learning method that constructs multiple decision
trees at training time and outputs the majority class of the
individual trees. Each tree is trained on a bootstrap sample
of the data and considers a random subset of features at each
split, which decorrelates the trees and improves generalization.
"""

import numpy as np
from nnf.tree.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest classifier.

    Parameters:
        n_estimators (int): Number of trees in the forest. Defaults to 100.
        max_depth (int or None): Maximum depth of each tree. Defaults to None.
        min_samples_split (int): Minimum samples required to split an
                                 internal node. Defaults to 2.
        min_samples_leaf (int): Minimum samples required to be at a leaf
                                node. Defaults to 1.
        max_features (int or str or None): Number of features to consider
                                          for the best split. If 'sqrt',
                                          uses sqrt(n_features). If 'log2',
                                          uses log2(n_features). If None,
                                          uses all features. Defaults to 'sqrt'.
        bootstrap (bool): Whether to bootstrap samples when building trees.
                          Defaults to True.
        random_state (int or None): Controls randomness. Defaults to None.
        n_jobs (int): Number of jobs for parallel fitting. Currently,
                      only sequential fitting is supported. Defaults to 1.

    Attributes:
        estimators_ (list of DecisionTreeClassifier): The fitted trees.
        classes_ (np.ndarray): Unique class labels.
        feature_importances_ (np.ndarray): Feature importance scores.

    Examples:
        >>> import numpy as np
        >>> from nnf.ensemble import RandomForestClassifier
        >>> X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        >>> y = np.array([0, 0, 1, 1])
        >>> model = RandomForestClassifier(n_estimators=10, max_depth=3)
        >>> model.fit(X, y)
        >>> model.predict(np.array([[0, 0]]))
        array([0])
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        bootstrap: bool = True,
        random_state: int = None,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators_ = []
        self.classes_ = None
        self.feature_importances_ = None

    def _get_max_features(self, n_features):
        """
        Determine the number of features to consider at each split.

        Parameters:
            n_features (int): Total number of features.

        Returns:
            int: Number of features to sample.
        """
        if self.max_features is None:
            return n_features
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, (int, float)):
            return max(1, int(self.max_features))
        else:
            return n_features

    def fit(self, X, y):
        """
        Build a forest of trees from the training data.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Returns:
            self: The fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        max_features = self._get_max_features(n_features)
        self.estimators_ = []

        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                criterion='gini',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=(self.random_state + i if self.random_state else None),
            )

            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
            else:
                X_bootstrap = X
                y_bootstrap = y

            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)

        # Compute feature importances as the average across all trees
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0
        )

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
        n_samples = X.shape[0]

        # Collect predictions from all trees
        all_preds = np.zeros((n_samples, self.n_estimators), dtype=int)
        for i, tree in enumerate(self.estimators_):
            all_preds[:, i] = tree.predict(X)

        # Majority vote per sample
        predictions = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            counts = np.bincount(all_preds[i])
            predictions[i] = np.argmax(counts)

        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability estimates of shape (n_samples, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(self.classes_)
        n_samples = X.shape[0]

        probas = np.zeros((n_samples, n_classes))
        for tree in self.estimators_:
            probas += tree.predict_proba(X)

        probas /= self.n_estimators
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
