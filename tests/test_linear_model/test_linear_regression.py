import numpy as np
import pytest
from epicon.linear_model import LinearRegression


@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    return X, y


def test_linear_regression_fit_predict(simple_data):
    X, y = simple_data
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    np.testing.assert_array_almost_equal(preds, y, decimal=5)


def test_linear_regression_coefficients(simple_data):
    X, y = simple_data
    model = LinearRegression()
    model.fit(X, y)
    np.testing.assert_array_almost_equal(model.coef_, [1, 2], decimal=3)
    np.testing.assert_almost_equal(model.intercept_, 3, decimal=3)


def test_linear_regression_gradient_descent(simple_data):
    X, y = simple_data
    model = LinearRegression(method='gd', epochs=10000, learning_rate=0.05)
    model.fit(X, y)
    preds = model.predict(X)
    np.testing.assert_array_almost_equal(preds, y, decimal=2)


def test_linear_regression_score(simple_data):
    X, y = simple_data
    model = LinearRegression()
    model.fit(X, y)
    assert abs(model.score(X, y) - 1.0) < 1e-5


def test_linear_regression_no_intercept():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    true_w = np.array([1.5, -2.0, 0.5])
    y = X @ true_w
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(model.coef_, true_w, decimal=3)
    assert abs(model.intercept_) < 1e-10


def test_linear_regression_single_feature():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    model = LinearRegression()
    model.fit(X, y)
    np.testing.assert_almost_equal(model.coef_[0], 2.0, decimal=3)
    np.testing.assert_almost_equal(model.intercept_, 0.0, decimal=3)
