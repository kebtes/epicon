import numpy as np

from epicon.layers import Dense
from epicon.optimizers import Adam


def test_adam_update_params():
    np.random.seed(42)
    layer = Dense(3, 2)
    initial_weights = layer.weights.copy()
    initial_biases = layer.biases.copy()

    layer.dweights = np.random.randn(3, 2) * 0.1
    layer.dbiases = np.random.randn(1, 2) * 0.1

    optimizer = Adam(learning_rate=0.1)
    optimizer.pre_update_params()
    optimizer.update_params(layer)
    optimizer.post_update_params()

    assert not np.array_equal(layer.weights, initial_weights)
    assert not np.array_equal(layer.biases, initial_biases)


def test_adam_get_params():
    optimizer = Adam(learning_rate=0.01, decay=1e-4, beta1=0.9, beta2=0.999)
    params = optimizer.get_params()
    assert params["type"] == "Adam"
    assert params["attrs"]["learning_rate"] == 0.01
    assert params["attrs"]["beta1"] == 0.9
    assert params["attrs"]["beta2"] == 0.999


def test_adam_learning_rate_decay():
    optimizer = Adam(learning_rate=0.1, decay=0.5)
    optimizer.pre_update_params()
    assert optimizer.current_learning_rate == 0.1 / (1.0 + 0.5 * 1)
    optimizer.post_update_params()
    optimizer.pre_update_params()
    assert optimizer.current_learning_rate == 0.1 / (1.0 + 0.5 * 3)
