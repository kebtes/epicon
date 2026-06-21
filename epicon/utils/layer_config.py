from epicon.activations import LeakyReLU, ReLU, Sigmoid, Softmax, Tanh
from epicon.layers import Dense
from epicon.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from epicon.optimizers import GradientDescent, Momentum

LAYER_REGISTERY = {
    "Dense": Dense,
    "ReLU": ReLU,
    "LeakyRelU": LeakyReLU,
    "Sigmoid": Sigmoid,
    "Softmax": Softmax,
    "Tanh": Tanh,
    "GradientDescent": GradientDescent,
    "Momentum": Momentum,
    "MSE": MSE,
    "BinaryCrossEntropy": BinaryCrossEntropy,
    "CategoricalCrossEntropy": CategoricalCrossEntropy,
}
