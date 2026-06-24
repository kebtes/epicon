import numpy as np

from epicon.models.callbacks import EarlyStopping


class DummyModel:
    class Layer:
        def __init__(self):
            self.trainable = True
            self.weights = None
            self.biases = None

    def __init__(self):
        self.layers = [self.Layer()]
        self.layers[0].weights = np.array([1.0, 2.0])
        self.layers[0].biases = np.array([0.0])


class TestEarlyStopping:
    def test_does_not_stop_on_improvement(self):
        early_stop = EarlyStopping(patience=3, min_delta=0.1)
        model = DummyModel()
        assert not early_stop(model, 1.0, 1)
        assert not early_stop(model, 0.8, 2)

    def test_stops_after_patience(self):
        early_stop = EarlyStopping(patience=2, min_delta=0.1)
        model = DummyModel()
        early_stop(model, 1.0, 1)
        early_stop(model, 0.95, 2)  # not improved (< 1.0-0.1)
        result = early_stop(model, 0.95, 3)  # not improved, wait=2 >= patience
        assert result

    def test_restores_best_weights(self):
        model = DummyModel()
        original_weights = model.layers[0].weights.copy()
        early_stop = EarlyStopping(patience=2, min_delta=0.0, restore_best_weights=True)
        early_stop(model, 1.0, 1)
        early_stop(model, 1.0, 2)
        early_stop(model, 1.0, 3)
        np.testing.assert_array_equal(model.layers[0].weights, original_weights)

    def test_stopped_epoch_is_recorded(self):
        early_stop = EarlyStopping(patience=2, min_delta=0.0)
        model = DummyModel()
        early_stop(model, 1.0, 1)
        early_stop(model, 1.0, 2)
        early_stop(model, 1.0, 3)
        assert early_stop.stopped_epoch == 3

    def test_best_loss_tracked(self):
        early_stop = EarlyStopping(patience=3)
        model = DummyModel()
        early_stop(model, 1.0, 1)
        early_stop(model, 0.5, 2)
        early_stop(model, 0.6, 3)
        assert early_stop.best_loss == 0.5
        assert early_stop.best_epoch == 2
