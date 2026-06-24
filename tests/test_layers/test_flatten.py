import numpy as np
import pytest

from epicon.layers import Flatten


@pytest.fixture
def flatten():
    return Flatten()


class TestFlatten:
    def test_forward_flattens_2d(self, flatten):
        x = np.random.randn(3, 4, 8, 8)
        out = flatten.forward(x)
        assert out.shape == (3, 256)

    def test_forward_flattens_3d(self, flatten):
        x = np.random.randn(5, 10, 20)
        out = flatten.forward(x)
        assert out.shape == (5, 200)

    def test_backward_restores_shape(self, flatten):
        x = np.random.randn(3, 4, 8, 8)
        out = flatten.forward(x)
        dout = np.random.randn(*out.shape)
        dx = flatten.backward(dout)
        assert dx.shape == x.shape

    def test_backward_values_preserved(self, flatten):
        x = np.random.randn(2, 3, 4)
        out = flatten.forward(x)
        dout = np.ones_like(out)
        dx = flatten.backward(dout)
        np.testing.assert_array_almost_equal(dx, np.ones_like(x))

    def test_not_trainable(self, flatten):
        assert not flatten.trainable
