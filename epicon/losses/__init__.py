from epicon.losses.base import Loss
from epicon.losses.binary_cross_entropy import BinaryCrossEntropy
from epicon.losses.categorical_cross_entropy import CategoricalCrossEntropy
from epicon.losses.mse import MSE

__all__ = ["MSE", "BinaryCrossEntropy", "CategoricalCrossEntropy"]
