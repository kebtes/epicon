import numpy as np
import pytest

from epicon.layers import Conv2D


@pytest.fixture
def conv():
    np.random.seed(42)
    return Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)


@pytest.fixture
def sample_input():
    np.random.seed(42)
    return np.random.randn(2, 1, 8, 8)


class TestConv2D:
    def test_forward_shape(self, conv, sample_input):
        out = conv.forward(sample_input)
        assert out.shape == (2, 4, 8, 8)

    def test_forward_no_padding(self, conv, sample_input):
        conv.padding = 0
        out = conv.forward(sample_input)
        assert out.shape == (2, 4, 6, 6)  # (8-3)//1+1 = 6

    def test_forward_stride_2(self, conv, sample_input):
        conv.stride = 2
        out = conv.forward(sample_input)
        assert out.shape == (2, 4, 4, 4)  # (8-3)//2+1 = 3... wait (8+2-3)//2+1 = 4

    def test_backward_shape(self, conv, sample_input):
        out = conv.forward(sample_input)
        dout = np.random.randn(*out.shape)
        dx = conv.backward(dout)
        assert dx.shape == sample_input.shape

    def test_gradients_populated(self, conv, sample_input):
        out = conv.forward(sample_input)
        dout = np.ones_like(out)
        conv.backward(dout)
        assert conv.dweights.shape == conv.weight.shape
        assert conv.dbias.shape == conv.bias.shape
        assert not np.allclose(conv.dweights, 0)
        assert not np.allclose(conv.dbias, 0)

    def test_trainable(self, conv):
        assert conv.trainable
