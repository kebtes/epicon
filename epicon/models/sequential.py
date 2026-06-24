"""
---------------------------------------------------------
SEQUENTIAL MODEL — HIGH-LEVEL NEURAL NETWORK API
---------------------------------------------------------

A simplified Keras-style wrapper around the Model class
that automatically inserts activation layers when activations
are specified as strings or passed alongside Dense layers.
"""

from epicon.activations import LeakyReLU, ReLU, Sigmoid, Softmax, Tanh
from epicon.layers import Dense
from epicon.models import Model

# Activation string to class mapping
_ACTIVATION_MAP = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "softmax": Softmax,
    "tanh": Tanh,
    "leaky_relu": LeakyReLU,
    "leakyrelu": LeakyReLU,
}


class Sequential(Model):
    """
    A Keras-like sequential model that provides a simplified API
    for building neural networks.

    Layers can be added either as a list at construction time or
    via the .add() method. Activation functions can be specified
    as strings (e.g., 'relu', 'sigmoid') and will be automatically
    instantiated.

    Parameters:
        layers (list or None): List of layers to include in the model.
        name (str or None): Optional model name.

    Examples:
        >>> from epicon import Sequential
        >>> from epicon.losses import CategoricalCrossEntropy
        >>> from epicon.optimizers import Adam
        >>> model = Sequential([
        ...     Dense(784, 128, activation='relu'),
        ...     Dense(128, 64, activation='relu'),
        ...     Dense(64, 10, activation='softmax'),
        ... ])
        >>> model.compile(loss=CategoricalCrossEntropy(), optimizer=Adam())
        >>> # model.fit(X, y, epochs=10)
    """

    def __init__(self, layers=None, name=None):
        if layers is None:
            processed = []
        else:
            processed = self._process_layers(layers)

        super().__init__(*processed, name=name)

    def add(self, layer):
        """
        Add a layer to the model.

        Parameters:
            layer (Layer or list): A layer instance, or a list of layers.
        """
        if isinstance(layer, list):
            processed = self._process_layers(layer)
        else:
            processed = self._process_layers([layer])

        self.layers.extend(processed)

    def _process_layers(self, layers):
        """
        Process a list of layers, automatically inserting activations
        when a Dense layer specifies an activation string.

        Parameters:
            layers (list): List of layer instances.

        Returns:
            list: Processed list of layers with activations inserted.
        """
        processed = []
        for layer in layers:
            if isinstance(layer, Dense):
                # Extract activation from Dense if specified as string
                if hasattr(layer, "_activation") and layer._activation:
                    processed.append(layer)
                    activation_cls = _ACTIVATION_MAP.get(layer._activation.lower())
                    if activation_cls:
                        processed.append(activation_cls())
                    delattr(layer, "_activation")
                else:
                    processed.append(layer)
            elif isinstance(layer, str):
                # Support string-based activation specification
                activation_cls = _ACTIVATION_MAP.get(layer.lower())
                if activation_cls:
                    processed.append(activation_cls())
                else:
                    raise ValueError(f"Unknown activation function: '{layer}'")
            else:
                processed.append(layer)
        return processed

    def compile(self, loss, optimizer):
        """
        Configure the model for training.

        Parameters:
            loss (Loss): Loss function instance.
            optimizer (Optimizer): Optimizer instance.
        """
        self.set(loss=loss, optimizer=optimizer)

    def fit(self, X, y, epochs=1, batch_size=None, shuffle=True, validation_split=0.0, callbacks=None, scheduler=None):
        """
        Train the model on data.

        Parameters:
            X (np.ndarray): Input data.
            y (np.ndarray): Target labels.
            epochs (int): Number of epochs. Defaults to 1.
            batch_size (int or None): Batch size. Defaults to None (full batch).
            shuffle (bool): Whether to shuffle data each epoch. Defaults to True.
            validation_split (float): Fraction of training data to use as
                                      validation set. Defaults to 0.0.
            callbacks (list or None): List of callback objects. Defaults to None.
            scheduler: Optional learning rate scheduler. Defaults to None.
        """
        self.shuffle = shuffle
        self.train(X, y, epochs=epochs, batch_size=batch_size,
                   validation_split=validation_split, callbacks=callbacks,
                   scheduler=scheduler)
