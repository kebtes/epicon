import pytest
from nnf.utils import ModelBuilder
from nnf.utils import LAYER_REGISTERY
from nnf.layers import Layer
from nnf.models import Model

class DummyLayer(Layer):
    def __init__(self, units=0, activation=None):
        self.units = units
        self.activation = activation
    
    def forward(self, x):
        return x
    
    def backward(self, dvalues):
        return super().backward(dvalues)

LAYER_REGISTERY["dummy"] = DummyLayer

def test_build_model_with_one_layer():
    config = [
        {"type": "dummy", "units": 10},
        {"type": "ReLU"}
    ]

    model = ModelBuilder().build(config)

    assert isinstance(model, Model)
    assert len(model.layers) == 2
    assert isinstance(model.layers[0], DummyLayer)
    assert model.layers[0].units == 10
    assert model.layers[1].name == "ReLU"

def test_build_model_with_empty_config():
    builder = ModelBuilder()

    config = []
    model = builder.build(config)

    assert isinstance(model, Model)
    assert len(model.layers) == 0
