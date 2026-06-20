import numpy as np
import pytest
from nnf.neighbors import KNeighborsClassifier


@pytest.fixture
def simple_data():
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_knn_classifier_fit_predict(simple_data):
    X, y = simple_data
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert model.score(X, y) > 0.5


def test_knn_classifier_k1(simple_data):
    X, y = simple_data
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y)
    preds = model.predict(X)
    np.testing.assert_array_equal(preds, y)


def test_knn_classifier_manhattan(simple_data):
    X, y = simple_data
    model = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
    model.fit(X, y)
    preds = model.predict(X)
    np.testing.assert_array_equal(preds, y)


def test_knn_classifier_predict_proba(simple_data):
    X, y = simple_data
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    probas = model.predict_proba(X)
    assert probas.shape == (len(X), 2)
    np.testing.assert_array_almost_equal(np.sum(probas, axis=1), np.ones(len(X)))


def test_knn_classifier_distance_weighted(simple_data):
    X, y = simple_data
    model = KNeighborsClassifier(n_neighbors=3, weights='distance')
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
