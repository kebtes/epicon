import numpy as np
import pytest
from nnf.naive_bayes import GaussianNB


@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_gaussian_nb_fit_predict(simple_data):
    X, y = simple_data
    model = GaussianNB()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_gaussian_nb_accuracy(simple_data):
    X, y = simple_data
    model = GaussianNB()
    model.fit(X, y)
    assert model.score(X, y) > 0.8


def test_gaussian_nb_predict_proba(simple_data):
    X, y = simple_data
    model = GaussianNB()
    model.fit(X, y)
    probas = model.predict_proba(X)
    assert probas.shape == (len(X), 2)
    np.testing.assert_array_almost_equal(np.sum(probas, axis=1), np.ones(len(X)))


def test_gaussian_nb_parameters(simple_data):
    X, y = simple_data
    model = GaussianNB()
    model.fit(X, y)
    assert model.theta_ is not None
    assert model.var_ is not None
    assert model.class_prior_ is not None
    assert model.classes_ is not None
    assert model.theta_.shape == (2, 2)
    assert model.var_.shape == (2, 2)
    assert len(model.class_prior_) == 2
