from epicon.activations import LeakyReLU
from epicon.activations import ReLU
from epicon.activations import Sigmoid
from epicon.activations import Softmax
from epicon.activations import Tanh

from epicon.layers import Dense

from epicon.losses import MSE
from epicon.losses import BinaryCrossEntropy
from epicon.losses import CategoricalCrossEntropy

from epicon.optimizers import GradientDescent
from epicon.optimizers import Momentum

LAYER_REGISTERY = {
    "Dense"                     : Dense,
    "ReLU"                      : ReLU,
    "LeakyRelU"                 : LeakyReLU,
    "Sigmoid"                   : Sigmoid,
    "Softmax"                   : Softmax,
    "Tanh"                      : Tanh,
    "GradientDescent"           : GradientDescent,
    "Momentum"                  : Momentum,
    "MSE"                       : MSE,
    "BinaryCrossEntropy"        : BinaryCrossEntropy,
    "CategoricalCrossEntropy"   : CategoricalCrossEntropy
}