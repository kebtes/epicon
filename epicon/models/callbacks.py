import copy

import numpy as np


class EarlyStopping:
    """
    Early stopping callback for Model training.

    Stops training when validation loss stops improving for a given
    number of epochs. Optionally restores the best model weights.

    Parameters:
        patience (int): Number of epochs to wait after last improvement.
                        Defaults to 10.
        min_delta (float): Minimum change in loss to qualify as improvement.
                           Defaults to 1e-4.
        restore_best_weights (bool): Whether to restore model weights from
                                     the epoch with the best loss.
                                     Defaults to True.

    Examples:
        >>> early_stop = EarlyStopping(patience=5, restore_best_weights=True)
        >>> model.train(X_train, y_train, epochs=100, callbacks=[early_stop])
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = np.inf
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = []

    def _save_weights(self, layers: list) -> list:
        """Deep-copy trainable weights from all layers."""
        return [
            {
                "weights": copy.deepcopy(layer.weights),
                "biases": copy.deepcopy(layer.biases),
            }
            for layer in layers
            if layer.trainable
        ]

    def _restore_weights(self, layers: list, weights: list):
        """Restore saved weights to trainable layers."""
        idx = 0
        for layer in layers:
            if layer.trainable:
                layer.weights = weights[idx]["weights"]
                layer.biases = weights[idx]["biases"]
                idx += 1

    def __call__(self, model, val_loss: float, epoch: int) -> bool:
        """
        Check early stopping condition after an epoch.

        Parameters:
            model: The Model being trained.
            val_loss (float): Validation loss for this epoch.
            epoch (int): Current epoch number (1-indexed).

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self._save_weights(model.layers)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights:
                    self._restore_weights(model.layers, self.best_weights)
                return True

        return False
