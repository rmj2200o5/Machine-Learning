from __future__ import annotations

import numpy as np

from ..core.base import BaseModel
from ..core.layer import DenseLayer
from ..core.neuron import Neuron


class _Conv1DLayer:
    """1D convolution-like layer with each filter as a shared neuron."""

    def __init__(self, kernel_size: int, n_filters: int, activation: str = "relu"):
        self.kernel_size = kernel_size
        self.filters = [Neuron(n_inputs=kernel_size, activation=activation) for _ in range(n_filters)]
        self._windows: np.ndarray | None = None

    def forward(self, sequence: np.ndarray) -> np.ndarray:
        seq = np.asarray(sequence, dtype=float)
        windows = np.asarray([seq[i : i + self.kernel_size] for i in range(len(seq) - self.kernel_size + 1)], dtype=float)
        self._windows = windows
        feature_maps = []
        for neuron in self.filters:
            feature_maps.append(np.asarray([neuron.forward(window) for window in windows], dtype=float))
        return np.asarray(feature_maps, dtype=float)

    def backward(self, grad_feature_maps: np.ndarray, learning_rate: float) -> np.ndarray:
        grad_sequence = np.zeros(self._windows.shape[0] + self.kernel_size - 1, dtype=float)
        for filter_idx, neuron in enumerate(self.filters):
            for pos, window in enumerate(self._windows):
                neuron.forward(window)
                grad_window = neuron.backward(float(grad_feature_maps[filter_idx, pos]), learning_rate)
                grad_sequence[pos : pos + self.kernel_size] += grad_window
        return grad_sequence


class SimpleCNNRegressor(BaseModel):
    def __init__(self, input_length: int, kernel_size: int = 3, n_filters: int = 2, learning_rate: float = 0.01, epochs: int = 200):
        if kernel_size > input_length:
            raise ValueError("kernel_size must not exceed input_length")
        self.input_length = input_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.conv = _Conv1DLayer(kernel_size=kernel_size, n_filters=n_filters, activation="relu")
        self.output = DenseLayer(n_filters, 1, activation="identity")
        self.losses: list[float] = []

    def _forward(self, x: np.ndarray):
        feature_maps = self.conv.forward(x)
        pooled = feature_maps.mean(axis=1)
        pred = self.output.forward(pooled)
        return pred, feature_maps

    def fit(self, X, y):
        X = self.to_numpy(X)
        y = self.to_numpy(y).reshape(-1)
        self.losses = []

        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xi, yi in zip(X, y):
                pred, feature_maps = self._forward(xi)
                error = pred[0] - yi
                epoch_loss += float(error**2)

                grad_pred = np.asarray([2.0 * error], dtype=float)
                grad_pooled = self.output.backward(grad_pred, self.learning_rate)

                grad_feature_maps = np.zeros_like(feature_maps)
                for f_idx, grad_val in enumerate(grad_pooled):
                    grad_feature_maps[f_idx, :] = grad_val / feature_maps.shape[1]
                self.conv.backward(grad_feature_maps, self.learning_rate)
            self.losses.append(epoch_loss / len(X))
        return self

    def predict(self, X):
        X = self.to_numpy(X)
        return np.asarray([self._forward(x)[0][0] for x in X], dtype=float)
