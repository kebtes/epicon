import numpy as np
import pytest
from epicon.linear_model import Lasso


def test_lasso_fit_predict():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_w = np.array([1.0, 0.0, 0.0, 0.0, 3.0])
    y = X @ true_w + 0.1 * np.random.randn(100)
    model = Lasso(alpha=0.1, max_iter=2000)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)
    assert model.score(X, y) > 0.9


def test_lasso_sparsity():
    np.random.seed(42)
    X = np.random.randn(100, 20)
    true_w = np.zeros(20)
    true_w[:3] = [1.5, -2.0, 0.5]
    y = X @ true_w + 0.1 * np.random.randn(100)

    model = Lasso(alpha=0.5, max_iter=2000)
    model.fit(X, y)
    # With strong regularization, many coefficients should be near zero
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-3)
    assert n_nonzero <= 10


def test_lasso_alpha_zero():
    np.random.seed(42)
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = Lasso(alpha=0.0)
    model.fit(X, y)
    preds = model.predict(X)
    np.testing.assert_array_almost_equal(preds, y, decimal=2)


def test_lasso_invalid_alpha():
    with pytest.raises(ValueError):
        Lasso(alpha=-0.5)
