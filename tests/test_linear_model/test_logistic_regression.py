import numpy as np
import pytest

from epicon.linear_model import LogisticRegression


@pytest.fixture
def binary_data():
    np.random.seed(42)
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_logistic_regression_fit_predict(binary_data):
    X, y = binary_data
    model = LogisticRegression(epochs=5000)
    model.fit(X, y)
    preds = model.predict(X)
    np.testing.assert_array_equal(preds, y)


def test_logistic_regression_predict_proba(binary_data):
    X, y = binary_data
    model = LogisticRegression(epochs=2000)
    model.fit(X, y)
    probas = model.predict_proba(X)
    assert probas.shape == (len(X), 2)
    assert np.all(probas >= 0) and np.all(probas <= 1)
    np.testing.assert_array_almost_equal(np.sum(probas, axis=1), np.ones(len(X)))


def test_logistic_regression_score(binary_data):
    X, y = binary_data
    model = LogisticRegression(epochs=2000)
    model.fit(X, y)
    assert model.score(X, y) > 0.8


def test_logistic_regression_l2_penalty(binary_data):
    X, y = binary_data
    model = LogisticRegression(epochs=2000, C=0.1, penalty="l2")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_logistic_regression_coefficients(binary_data):
    X, y = binary_data
    model = LogisticRegression(epochs=2000)
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.coef_.shape[0] == X.shape[1]
    assert isinstance(model.intercept_, float)
