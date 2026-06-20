from typing import List, Dict, Any

from epicon.utils import LAYER_REGISTERY
from epicon.layers import Layer

class ModelBuilder:
    """
    ModelBuilder is responsible for dynamically constructing a model
    from a configuration dictionary. It leverages a layer registry to
    resolve layer classes by name and initialize them with parameters.
    """

    def __init__(self):
        """
        Initializes the ModelBuilder instance with an empty list of layers.
        """
        self.layers: List[Layer] = []

    def build(self, config: List[Dict[str, Any]]):
        """
        Builds a model from a given configuration list.

        Args:
            config (List[Dict[str, Any]]): A list of layer configurations.
                Each dict must have a 'type' key indicating the layer name
                and other keys as parameters for that layer.

        Returns:
            Model: A model instance composed of the specified layers.
        """
        self.layers = []

        for layer_config in config:
            layer_type = layer_config.pop("type")
            LayerClass = LAYER_REGISTERY[layer_type]
            layer_instance = LayerClass(**layer_config)
            self.layers.append(layer_instance)

        return self.__build_model(self.layers)
    
    def __build_model(self, layers: List[Layer]):
        """
        Internal method to construct a Model object from a list of layers.

        Args:
            layers (List[Layer]): List of initialized layer instances.

        Returns:
            Model: A Model composed of the provided layers.
        """
        from epicon.models import Model
        return Model(*layers)
