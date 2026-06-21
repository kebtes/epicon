import numpy as np
import pytest

from epicon.tree import DecisionTreeClassifier


@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0], [2, 2], [3, 3]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_decision_tree_classifier_fit_predict(simple_data):
    X, y = simple_data
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert model.score(X, y) > 0.8


def test_decision_tree_classifier_entropy(simple_data):
    X, y = simple_data
    model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    model.fit(X, y)
    assert model.score(X, y) > 0.8


def test_decision_tree_classifier_pure():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 0, 0])
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    preds = model.predict(np.array([[5], [6]]))
    np.testing.assert_array_equal(preds, [0, 0])


def test_decision_tree_classifier_depth_control(simple_data):
    X, y = simple_data
    model_shallow = DecisionTreeClassifier(max_depth=1)
    model_shallow.fit(X, y)
    model_deep = DecisionTreeClassifier(max_depth=10)
    model_deep.fit(X, y)
    # Deep tree should have at least as good training accuracy
    assert model_deep.score(X, y) >= model_shallow.score(X, y)


def test_decision_tree_classifier_feature_importances(simple_data):
    X, y = simple_data
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    assert model.feature_importances_ is not None
    assert model.feature_importances_.shape[0] == X.shape[1]
    np.testing.assert_almost_equal(np.sum(model.feature_importances_), 1.0)
