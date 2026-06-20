import numpy as np

from nnf.layers import Layer

class Conv1D(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Xavier initialization
        limit = np.sqrt(6 / (in_channels + out_channels))
        self.weight = np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size))
        self.bias = np.zeros(out_channels)

        # Gradients
        self.dweights = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)
        self.grad_input = None

        self.input = None
        self.input_padded = None

    def _pad_input(self, x):
        if self.padding == 0:
            return x
        
        return np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)), mode='constant')
    
    def forward(self, x):
        self.input = x
        batch_size, _, input_width = x.shape

        x_padded = self._pad_input(x)
        self.input_padded = x_padded

        out_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, self.out_channels, out_width))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_width):
                    start = i * self.stride
                    end = start + self.kernel_size

                    region = x_padded[b, :, start:end] # (in_channels, kernel_size)
                    out[b, oc, i] = np.sum(region * self.weight[oc]) + self.bias[oc]
        
        return out
    
    def backward(self, doutput):
        batch_size, _, input_width = self.input.shape
        _, _, output_width = doutput.shape

        # Gradient initialization
        self.dweights.fill(0)
        self.dbias.fill(0)
        dx_padded = np.zeros_like(self.input_padded)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(output_width):
                    start = i * self.stride
                    end = start + self.kernel_size

                    self.dweights[oc] += doutput[b, oc, i] * self.input_padded[b, :, start: end]

                    dx_padded[b, :, start: end] += doutput[b, oc, i] * self.weight[oc]

                self.dbias[oc] += np.sum(doutput[b, oc, :])
        
        if self.padding > 0:
            self.dinputs = dx_padded[:, :, self.padding: - self.padding]
        else:
            self.dinputs = dx_padded
        
        return self.dinputs
