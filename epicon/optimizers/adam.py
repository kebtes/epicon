"""
---------------------------------------------------------
ADAM OPTIMIZER
---------------------------------------------------------

Adam (Adaptive Moment Estimation) is an optimization algorithm
that computes adaptive learning rates for each parameter by
estimating the first and second moments of the gradients.

It combines the advantages of Momentum (exponential moving
average of gradients) and RMSProp (adaptive learning rates).

Reference:
    Kingma, D. P., & Ba, J. (2014). Adam: A Method for
    Stochastic Optimization. arXiv:1412.6980.
"""

import numpy as np

from epicon.layers.base import Layer
from epicon.optimizers.base import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer with adaptive learning rates.

    Parameters:
        learning_rate (float): Initial learning rate. Defaults to 0.001.
        decay (float): Learning rate decay factor. Defaults to 0.
        beta1 (float): Exponential decay rate for first moment
                       estimates. Defaults to 0.9.
        beta2 (float): Exponential decay rate for second moment
                       estimates. Defaults to 0.999.
        epsilon (float): Small constant for numerical stability.
                         Defaults to 1e-7.

    Attributes:
        iterations (int): Number of parameter update iterations.
        current_learning_rate (float): Decayed learning rate.

    Examples:
        >>> from epicon.optimizers import Adam
        >>> optimizer = Adam(learning_rate=0.001)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay: float = 0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
    ):
        super().__init__(learning_rate, decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.cache = {}

    def update_params(self, layer: Layer):
        """
        Update layer parameters using the Adam algorithm.

        Parameters:
            layer (Layer): The layer whose parameters to update.
        """
        if layer not in self.cache:
            self.cache[layer] = {
                "m_weights": np.zeros_like(layer.weights),
                "v_weights": np.zeros_like(layer.weights),
                "m_biases": (np.zeros_like(layer.biases) if layer.biases is not None else None),
                "v_biases": (np.zeros_like(layer.biases) if layer.biases is not None else None),
            }

        cache = self.cache[layer]
        t = self.iterations + 1  # time step (1-indexed)

        # Update biased first and second moment estimates for weights
        cache["m_weights"] = self.beta1 * cache["m_weights"] + (1 - self.beta1) * layer.dweights
        cache["v_weights"] = self.beta2 * cache["v_weights"] + (1 - self.beta2) * (layer.dweights**2)

        # Bias correction
        m_hat = cache["m_weights"] / (1 - self.beta1**t)
        v_hat = cache["v_weights"] / (1 - self.beta2**t)

        # Update weights
        layer.weights -= self.current_learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Update biases if they exist
        if layer.biases is not None:
            cache["m_biases"] = self.beta1 * cache["m_biases"] + (1 - self.beta1) * layer.dbiases
            cache["v_biases"] = self.beta2 * cache["v_biases"] + (1 - self.beta2) * (layer.dbiases**2)

            m_hat_b = cache["m_biases"] / (1 - self.beta1**t)
            v_hat_b = cache["v_biases"] / (1 - self.beta2**t)

            layer.biases -= self.current_learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def pre_update_params(self):
        """
        Decay learning rate and increment iteration counter.
        """
        self.iterations += 1
        self.current_learning_rate = self.learning_rate / (1.0 + self.decay * self.iterations)

    def get_params(self):
        return {
            "type": "Adam",
            "attrs": {
                "learning_rate": self.learning_rate,
                "decay": self.decay,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
            },
        }

    def set_params(self, params: dict):
        for key, val in params.items():
            setattr(self, key, val)
