import numpy as np
import pytest

from epicon.layers import BatchNormalization


@pytest.fixture
def bn():
    return BatchNormalization(n_features=4)


@pytest.fixture
def sample_input():
    np.random.seed(42)
    return np.random.randn(16, 4)


class TestBatchNormalization:
    def test_forward_training_normalizes(self, bn, sample_input):
        bn.training = True
        out = bn.forward(sample_input)
        assert out.shape == sample_input.shape
        assert np.allclose(np.mean(out, axis=0), 0, atol=1e-7)
        assert np.allclose(np.std(out, axis=0), 1, atol=1e-7)

    def test_forward_inference_uses_running_stats(self, bn, sample_input):
        bn.training = True
        bn.forward(sample_input)
        bn.training = False
        out = bn.forward(sample_input)
        assert out.shape == sample_input.shape

    def test_gamma_beta_learnable(self, bn):
        assert bn.trainable
        assert bn.gamma.shape == (4,)
        assert bn.beta.shape == (4,)
        assert np.allclose(bn.gamma, 1.0)
        assert np.allclose(bn.beta, 0.0)

    def test_backward_produces_gradients(self, bn, sample_input):
        bn.training = True
        out = bn.forward(sample_input)
        dout = np.ones_like(out)
        dx = bn.backward(dout)
        assert dx.shape == sample_input.shape
        assert bn.dgamma.shape == (4,)
        assert bn.dbeta.shape == (4,)

    def test_running_stats_update(self, bn, sample_input):
        bn.training = True
        bn.forward(sample_input)
        assert not np.allclose(bn.running_mean, 0)
        assert not np.allclose(bn.running_var, 1)

    def test_n_features_mismatch_raises(self):
        bn = BatchNormalization(n_features=4)
        x = np.random.randn(8, 6)
        with pytest.raises((ValueError, TypeError)):
            bn.forward(x)
