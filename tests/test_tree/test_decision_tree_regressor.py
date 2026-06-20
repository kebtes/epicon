import numpy as np
import pytest
from epicon.tree import DecisionTreeRegressor


@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return X, y


def test_decision_tree_regressor_fit_predict(simple_data):
    X, y = simple_data
    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_decision_tree_regressor_mse():
    np.random.seed(42)
    X = np.random.randn(100, 1) * 10
    y = 2 * X.ravel() + 1 + np.random.randn(100) * 0.5
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X, y)
    assert model.score(X, y) > 0.8


def test_decision_tree_regressor_constant():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([5.0, 5.0, 5.0, 5.0])
    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X, y)
    preds = model.predict(np.array([[5], [6]]))
    np.testing.assert_array_almost_equal(preds, [5.0, 5.0])


def test_decision_tree_regressor_feature_importances():
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = X[:, 0] * 2 + np.random.randn(50) * 0.1
    model = DecisionTreeRegressor(max_depth=4)
    model.fit(X, y)
    assert model.feature_importances_ is not None
    assert model.feature_importances_.shape[0] == 5
    # Feature 0 should be most important
    assert model.feature_importances_[0] > 0
