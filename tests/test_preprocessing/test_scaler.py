import numpy as np
import pytest
from nnf.preprocessing import StandardScaler, MinMaxScaler


class TestStandardScaler:
    @pytest.fixture
    def data(self):
        return np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)

    def test_fit_transform(self, data):
        scaler = StandardScaler()
        transformed = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(np.mean(transformed, axis=0), [0, 0], decimal=5)
        np.testing.assert_array_almost_equal(np.std(transformed, axis=0), [1, 1], decimal=5)

    def test_inverse_transform(self, data):
        scaler = StandardScaler()
        transformed = scaler.fit_transform(data)
        reconstructed = scaler.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(reconstructed, data, decimal=5)

    def test_fit_attributes(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        np.testing.assert_array_almost_equal(scaler.mean_, [3, 4])
        np.testing.assert_array_almost_equal(scaler.std_, [np.std([1, 3, 5]), np.std([2, 4, 6])])

    def test_single_sample(self):
        X = np.array([[5, 10]])
        scaler = StandardScaler()
        scaler.fit(X)
        transformed = scaler.transform(X)
        np.testing.assert_array_almost_equal(transformed, [[0, 0]], decimal=5)


class TestMinMaxScaler:
    @pytest.fixture
    def data(self):
        return np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)

    def test_fit_transform(self, data):
        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(transformed[0], [0, 0])
        np.testing.assert_array_almost_equal(transformed[-1], [1, 1])

    def test_inverse_transform(self, data):
        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(data)
        reconstructed = scaler.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(reconstructed, data, decimal=5)

    def test_custom_range(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        transformed = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(transformed[0], [-1, -1])
        np.testing.assert_array_almost_equal(transformed[-1], [1, 1])
