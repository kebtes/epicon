import numpy as np
import pytest
from nnf.ensemble import RandomForestClassifier


@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0], [5, 5], [6, 6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_random_forest_classifier_fit_predict(simple_data):
    X, y = simple_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_random_forest_classifier_score(simple_data):
    X, y = simple_data
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.8


def test_random_forest_classifier_estimators(simple_data):
    X, y = simple_data
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)
    assert len(model.estimators_) == 5


def test_random_forest_classifier_predict_proba(simple_data):
    X, y = simple_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    probas = model.predict_proba(X)
    assert probas.shape[0] == len(X)
    assert probas.shape[1] == 2
    np.testing.assert_array_almost_equal(np.sum(probas, axis=1), np.ones(len(X)))
