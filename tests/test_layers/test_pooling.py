import numpy as np
import pytest

from epicon.layers import MaxPooling2D


@pytest.fixture
def pool():
    return MaxPooling2D(pool_size=2, stride=2)


@pytest.fixture
def sample_input():
    np.random.seed(42)
    return np.random.randn(2, 4, 8, 8)


class TestMaxPooling2D:
    def test_forward_shape(self, pool, sample_input):
        out = pool.forward(sample_input)
        assert out.shape == (2, 4, 4, 4)

    def test_forward_takes_max(self):
        pool = MaxPooling2D(pool_size=2, stride=2)
        x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float64)  # (1, 1, 2, 2)
        out = pool.forward(x)
        assert out[0, 0, 0, 0] == 4.0

    def test_backward_shape(self, pool, sample_input):
        out = pool.forward(sample_input)
        dout = np.random.randn(*out.shape)
        dx = pool.backward(dout)
        assert dx.shape == sample_input.shape

    def test_backward_routes_to_max(self):
        pool = MaxPooling2D(pool_size=2, stride=2)
        x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float64)
        out = pool.forward(x)
        dout = np.ones_like(out)
        dx = pool.backward(dout)
        # Only position (1, 1) had max=4, so only that gets gradient
        assert dx[0, 0, 1, 1] == 1.0
        assert dx[0, 0, 0, 0] == 0.0
        assert dx[0, 0, 0, 1] == 0.0
        assert dx[0, 0, 1, 0] == 0.0

    def test_not_trainable(self, pool):
        assert not pool.trainable

    def test_overlapping_pooling(self):
        pool = MaxPooling2D(pool_size=3, stride=1)
        x = np.random.randn(1, 1, 5, 5)
        out = pool.forward(x)
        assert out.shape == (1, 1, 3, 3)
        dx = pool.backward(np.ones_like(out))
        assert dx.shape == x.shape
