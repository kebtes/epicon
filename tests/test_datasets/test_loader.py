import numpy as np
import pytest
from epicon.datasets import load_iris


def test_load_iris_shape():
    X, y = load_iris(return_X_y=True)
    assert X.shape == (150, 4)
    assert y.shape == (150,)


def test_load_iris_bunch():
    bunch = load_iris(return_X_y=False)
    assert bunch.data.shape == (150, 4)
    assert bunch.target.shape == (150,)
    assert len(bunch.target_names) == 3
    assert len(bunch.feature_names) == 4


def test_load_iris_classes():
    X, y = load_iris(return_X_y=True)
    assert len(np.unique(y)) == 3
    assert np.min(y) == 0
    assert np.max(y) == 2
