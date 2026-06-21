import numpy as np

from epicon.datasets import make_classification, make_regression


class TestMakeRegression:
    def test_shape(self):
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        assert X.shape == (50, 5)
        assert y.shape == (50,)

    def test_reproducible(self):
        X1, y1 = make_regression(n_samples=100, random_state=42)
        X2, y2 = make_regression(n_samples=100, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestMakeClassification:
    def test_shape(self):
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        assert X.shape == (50, 4)
        assert y.shape == (50,)

    def test_binary(self):
        X, y = make_classification(n_samples=100, random_state=42)
        assert len(np.unique(y)) == 2

    def test_multiclass(self):
        X, y = make_classification(n_samples=100, n_classes=3, random_state=42)
        assert len(np.unique(y)) == 3
