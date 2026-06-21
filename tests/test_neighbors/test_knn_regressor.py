import numpy as np
import pytest

from epicon.neighbors import KNeighborsRegressor


@pytest.fixture
def simple_data():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.5, 2.0, 3.5, 4.0, 5.5])
    return X, y


def test_knn_regressor_fit_predict(simple_data):
    X, y = simple_data
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_knn_regressor_k1(simple_data):
    X, y = simple_data
    model = KNeighborsRegressor(n_neighbors=1)
    model.fit(X, y)
    preds = model.predict(X)
    np.testing.assert_array_almost_equal(preds, y)


def test_knn_regressor_score(simple_data):
    X, y = simple_data
    model = KNeighborsRegressor(n_neighbors=2)
    model.fit(X, y)
    assert model.score(X, y) > 0.5
