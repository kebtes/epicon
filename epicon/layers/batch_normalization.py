import numpy as np

from epicon.layers import Layer


class BatchNormalization(Layer):
    """
    Batch normalization layer.

    Normalizes the input to have zero mean and unit variance, then applies
    a learnable scale (gamma) and shift (beta). During training, statistics
    are computed per batch. During inference, running averages are used.

    Parameters:
        n_features (int): Number of input features.
        epsilon (float): Small constant for numerical stability. Defaults to 1e-5.
        momentum (float): Momentum for running mean/var update. Defaults to 0.9.

    Attributes:
        gamma (np.ndarray): Learnable scale parameter.
        beta (np.ndarray): Learnable shift parameter.
        running_mean (np.ndarray): Running mean for inference.
        running_var (np.ndarray): Running variance for inference.
        training (bool): Whether the layer is in training mode.
    """

    def __init__(self, n_features: int, epsilon: float = 1e-5, momentum: float = 0.9):
        super().__init__()
        self.n_inputs = n_features
        self.n_neurons = n_features
        self.epsilon = epsilon
        self.momentum = momentum

        self.trainable = True
        self.training = True

        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)

        self.params = n_features + n_features

        self.dgamma = None
        self.dbeta = None

        self.x_hat = None
        self.std = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass for batch normalization.

        Parameters:
            inputs (np.ndarray): Input of shape (batch_size, n_features).

        Returns:
            np.ndarray: Normalized, scaled, and shifted output.
        """
        self.inputs = inputs

        if self.training:
            mean = np.mean(inputs, axis=0)
            var = np.var(inputs, axis=0)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        self.std = np.sqrt(var + self.epsilon)
        self.x_hat = (inputs - mean) / self.std
        return self.gamma * self.x_hat + self.beta

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        Backward pass for batch normalization.

        Parameters:
            dvalues (np.ndarray): Upstream gradient of shape (batch_size, n_features).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        batch_size = dvalues.shape[0]

        self.dgamma = np.sum(dvalues * self.x_hat, axis=0)
        self.dbeta = np.sum(dvalues, axis=0)

        x_centered = self.inputs - np.mean(self.inputs, axis=0)
        dx_hat = dvalues * self.gamma
        dvar = np.sum(dx_hat * x_centered * -0.5 * self.std ** (-3), axis=0)
        dmean = np.sum(dx_hat * -1.0 / self.std, axis=0) + dvar * np.mean(-2.0 * x_centered, axis=0)

        self.dinputs = dx_hat / self.std + dvar * 2.0 * x_centered / batch_size + dmean / batch_size

        return self.dinputs
