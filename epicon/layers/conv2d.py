import numpy as np

from epicon.layers import Layer


class Conv2D(Layer):
    """
    2D convolution layer.

    Applies a 2D convolution over an input signal composed of multiple
    input planes (channels). Commonly used for image processing.

    Parameters:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of kernels (output channels).
        kernel_size (int): Size of the convolving kernel (square).
        stride (int): Stride of the convolution. Defaults to 1.
        padding (int): Zero-padding added to both sides. Defaults to 0.

    Attributes:
        weight (np.ndarray): Kernel weights of shape
            (out_channels, in_channels, kernel_size, kernel_size).
        bias (np.ndarray): Bias of shape (out_channels,).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.trainable = True

        limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size))
        self.weight = np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels)

        self.params = self.weight.size + self.bias.size

        self.dweights = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)

        self.input_padded = None

    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        if self.padding == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode="constant")

    def _out_shape(self, input_height: int, input_width: int) -> tuple[int, int]:
        out_h = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        return out_h, out_w

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for 2D convolution.

        Parameters:
            x (np.ndarray): Input of shape (batch_size, in_channels, height, width).

        Returns:
            np.ndarray: Output of shape (batch_size, out_channels, out_height, out_width).
        """
        self.inputs = x
        batch_size, _, in_h, in_w = x.shape

        x_padded = self._pad_input(x)
        self.input_padded = x_padded

        out_h, out_w = self._out_shape(in_h, in_w)
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        region = x_padded[b, :, h_start:h_end, w_start:w_end]
                        out[b, oc, i, j] = np.sum(region * self.weight[oc]) + self.bias[oc]

        return out

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        """
        Backward pass for 2D convolution.

        Parameters:
            doutput (np.ndarray): Upstream gradient of shape
                                  (batch_size, out_channels, out_height, out_width).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        batch_size, _, _, _ = self.inputs.shape
        _, _, out_h, out_w = doutput.shape

        self.dweights.fill(0)
        self.dbias.fill(0)
        dx_padded = np.zeros_like(self.input_padded)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        region = self.input_padded[b, :, h_start:h_end, w_start:w_end]
                        self.dweights[oc] += doutput[b, oc, i, j] * region
                        dx_padded[b, :, h_start:h_end, w_start:w_end] += doutput[b, oc, i, j] * self.weight[oc]

                self.dbias[oc] += np.sum(doutput[b, oc, :, :])

        if self.padding > 0:
            self.dinputs = dx_padded[:, :, self.padding : -self.padding, self.padding : -self.padding]
        else:
            self.dinputs = dx_padded

        return self.dinputs
