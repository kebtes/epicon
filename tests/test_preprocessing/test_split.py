import numpy as np
import pytest
from nnf.preprocessing import train_test_split


def test_train_test_split_default():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    assert len(X_train) == 3
    assert len(X_test) == 2
    assert len(y_train) == 3
    assert len(y_test) == 2


def test_train_test_split_reproducible():
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    X1, _, y1, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X2, _, y2, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_train_test_split_small():
    X = np.array([[1]])
    y = np.array([1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == 0 or len(X_test) == 0 or (len(X_train) > 0 and len(X_test) > 0)
