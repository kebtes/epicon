import numpy as np
import pytest
from epicon.metrics import mean_squared_error, mean_absolute_error, r2_score


class TestMeanSquaredError:
    def test_perfect(self):
        assert mean_squared_error([1, 2, 3], [1, 2, 3]) == 0.0

    def test_small_error(self):
        mse = mean_squared_error([1, 2, 3], [1.1, 1.9, 3.2])
        assert abs(mse - 0.02) < 1e-10


class TestMeanAbsoluteError:
    def test_perfect(self):
        assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0.0

    def test_value(self):
        mae = mean_absolute_error([1, 2, 3], [2, 3, 4])
        assert mae == 1.0


class TestR2Score:
    def test_perfect(self):
        assert r2_score([1, 2, 3], [1, 2, 3]) == 1.0

    def test_mean_baseline(self):
        y = [1, 2, 3]
        r2 = r2_score(y, [2, 2, 2])
        assert abs(r2) < 1e-10
