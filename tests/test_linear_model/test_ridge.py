import numpy as np
import pytest

from epicon.linear_model import Ridge


def test_ridge_fit_predict():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = X @ true_w + 0.1 * np.random.randn(100)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)
    assert model.score(X, y) > 0.9


def test_ridge_regularization():
    np.random.seed(42)
    # Create collinear features to test regularization
    X = np.random.randn(50, 10)
    true_w = np.random.randn(10)
    y = X @ true_w + 0.1 * np.random.randn(50)

    ridge_high = Ridge(alpha=10.0)
    ridge_low = Ridge(alpha=0.001)
    ridge_high.fit(X, y)
    ridge_low.fit(X, y)

    # High alpha should shrink coefficients more
    assert np.linalg.norm(ridge_high.coef_) < np.linalg.norm(ridge_low.coef_)


def test_ridge_no_alpha():
    np.random.seed(42)
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = Ridge(alpha=0.0)
    model.fit(X, y)
    preds = model.predict(X)
    np.testing.assert_array_almost_equal(preds, y, decimal=4)


def test_ridge_invalid_alpha():
    with pytest.raises(ValueError):
        Ridge(alpha=-0.5)


def test_ridge_no_intercept():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X @ np.array([1.5, -2.0, 0.5])
    model = Ridge(alpha=0.1, fit_intercept=False)
    model.fit(X, y)
    assert abs(model.intercept_) < 1e-10
