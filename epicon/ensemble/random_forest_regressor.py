"""
---------------------------------------------------------
RANDOM FOREST REGRESSOR
---------------------------------------------------------

An ensemble learning method that constructs multiple decision
trees at training time and outputs the mean prediction of the
individual trees. Each tree is trained on a bootstrap sample
of the data, and a random subset of features is considered at
each split to decorrelate the trees.
"""

import numpy as np

from epicon.tree.decision_tree_regressor import DecisionTreeRegressor


class RandomForestRegressor:
    """
    Random Forest regressor.

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

    Attributes:
        estimators_ (list of DecisionTreeRegressor): The fitted trees.
        feature_importances_ (np.ndarray): Feature importance scores.

    Examples:
        >>> import numpy as np
        >>> from epicon.ensemble import RandomForestRegressor
        >>> X = np.array([[1], [2], [3], [4]])
        >>> y = np.array([1.0, 2.0, 3.0, 4.0])
        >>> model = RandomForestRegressor(n_estimators=10, max_depth=3)
        >>> model.fit(X, y)
        >>> model.predict(np.array([[2.5]]))
        array([2.5])
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        bootstrap: bool = True,
        random_state: int = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_ = []
        self.feature_importances_ = None

    def _get_max_features(self, n_features):
        """
        Determine the number of features to consider at each split.
        """
        if self.max_features is None:
            return n_features
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int | float):
            return max(1, int(self.max_features))
        else:
            return n_features

    def fit(self, X, y):
        """
        Build a forest of trees from the training data.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).

        Returns:
            self: The fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        max_features = self._get_max_features(n_features)
        self.estimators_ = []

        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                criterion="mse",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=(self.random_state + i if self.random_state else None),
            )

            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
            else:
                X_bootstrap = X
                y_bootstrap = y

            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)

        # Average feature importances across all trees
        self.feature_importances_ = np.mean([tree.feature_importances_ for tree in self.estimators_], axis=0)

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
        n_samples = X.shape[0]

        all_preds = np.zeros((n_samples, self.n_estimators))
        for i, tree in enumerate(self.estimators_):
            all_preds[:, i] = tree.predict(X)

        return np.mean(all_preds, axis=1)

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
