import numpy as np

from epicon.layers import Layer


class Flatten(Layer):
    """
    Flatten layer.

    Reshapes the input tensor into a 2D array of shape
    (batch_size, channels * height * width) for transition
    from convolutional layers to dense layers.

    No learnable parameters.
    """

    def __init__(self):
        super().__init__()
        self._input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: flatten input to (batch_size, ...).

        Parameters:
            x (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Flattened output.
        """
        self.inputs = x
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        Backward pass: reshape gradient back to original input shape.

        Parameters:
            dvalues (np.ndarray): Upstream gradient.

        Returns:
            np.ndarray: Reshaped gradient matching the original input shape.
        """
        self.dinputs = dvalues.reshape(self._input_shape)
        return self.dinputs
