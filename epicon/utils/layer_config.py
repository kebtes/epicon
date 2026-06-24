from epicon.activations import LeakyReLU, ReLU, Sigmoid, Softmax, Tanh
from epicon.layers import Conv1D, Dense, Dropout
from epicon.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from epicon.optimizers import Adam, GradientDescent, Momentum

LAYER_REGISTERY = {
    "Dense": Dense,
    "Dropout": Dropout,
    "Conv1D": Conv1D,
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
