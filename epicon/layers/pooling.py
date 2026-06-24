import numpy as np

from epicon.layers import Layer


class MaxPooling2D(Layer):
    """
    2D max pooling layer.

    Downsamples the input by taking the maximum value over a
    spatial window of size pool_size x pool_size.

    Parameters:
        pool_size (int): Size of the pooling window (square).
        stride (int): Stride of the pooling. Defaults to pool_size
                      (non-overlapping windows).

    Attributes:
        mask (np.ndarray): Max position indices for backward pass.
    """

    def __init__(self, pool_size: int, stride: int = None):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for max pooling.

        Parameters:
            x (np.ndarray): Input of shape (batch_size, channels, height, width).

        Returns:
            np.ndarray: Pooled output of shape
                        (batch_size, channels, out_h, out_w).
        """
        self.inputs = x
        batch_size, channels, in_h, in_w = x.shape

        out_h = (in_h - self.pool_size) // self.stride + 1
        out_w = (in_w - self.pool_size) // self.stride + 1

        out = np.zeros((batch_size, channels, out_h, out_w))
        self.mask = np.zeros_like(x)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        window = x[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        out[b, c, i, j] = max_val

                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        self.mask[b, c, h_start + max_idx[0], w_start + max_idx[1]] = 1

        return out

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        """
        Backward pass for max pooling.

        Routes the upstream gradient to the position of each max value.

        Parameters:
            doutput (np.ndarray): Upstream gradient.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        batch_size, channels, out_h, out_w = doutput.shape

        dx = np.zeros_like(self.inputs)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_slice = slice(h_start, h_start + self.pool_size)
                        w_slice = slice(w_start, w_start + self.pool_size)
                        dx[b, c, h_slice, w_slice] += doutput[b, c, i, j] * self.mask[b, c, h_slice, w_slice]

        self.dinputs = dx
        return self.dinputs
