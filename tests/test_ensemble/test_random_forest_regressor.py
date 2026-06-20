import numpy as np
import pytest
from epicon.ensemble import RandomForestRegressor


@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return X, y


def test_random_forest_regressor_fit_predict(simple_data):
    X, y = simple_data
    model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_random_forest_regressor_score(simple_data):
    X, y = simple_data
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    r2 = model.score(X, y)
    assert r2 > 0.8


def test_random_forest_regressor_estimators(simple_data):
    X, y = simple_data
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(X, y)
    assert len(model.estimators_) == 5
