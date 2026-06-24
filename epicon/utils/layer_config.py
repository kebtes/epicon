from epicon.activations import LeakyReLU, ReLU, Sigmoid, Softmax, Tanh
from epicon.layers import BatchNormalization, Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from epicon.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from epicon.optimizers import Adam, GradientDescent, Momentum

LAYER_REGISTERY = {
    "Dense": Dense,
    "Dropout": Dropout,
    "BatchNormalization": BatchNormalization,
    "Conv1D": Conv1D,
    "Conv2D": Conv2D,
    "MaxPooling2D": MaxPooling2D,
    "Flatten": Flatten,
    "ReLU": ReLU,
    "LeakyReLU": LeakyReLU,
    "Sigmoid": Sigmoid,
    "Softmax": Softmax,
    "Tanh": Tanh,
    "GradientDescent": GradientDescent,
    "Momentum": Momentum,
    "Adam": Adam,
    "MSE": MSE,
    "BinaryCrossEntropy": BinaryCrossEntropy,
    "CategoricalCrossEntropy": CategoricalCrossEntropy,
}
