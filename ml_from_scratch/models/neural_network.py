from __future__ import annotations

import numpy as np

from ..core.base import BaseModel
from ..core.layer import DenseLayer


class NeuralNetworkRegressor(BaseModel):
    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.01, epochs: int = 200):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include at least input and output sizes")
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = [
            DenseLayer(layer_sizes[i], layer_sizes[i + 1], activation="relu" if i < len(layer_sizes) - 2 else "identity")
            for i in range(len(layer_sizes) - 1)
        ]
        self.losses: list[float] = []

    def _forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def fit(self, X, y):
        X = self.to_numpy(X)
        y = self.to_numpy(y).reshape(-1, self.layer_sizes[-1])
        self.losses = []

        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xi, yi in zip(X, y):
                pred = self._forward(xi)
                error = pred - yi
                epoch_loss += float(np.mean(error**2))

                grad = 2.0 * error / len(error)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, self.learning_rate)
            self.losses.append(epoch_loss / len(X))
        return self

    def predict(self, X):
        X = self.to_numpy(X)
        preds = np.asarray([self._forward(x) for x in X], dtype=float)
        if preds.shape[1] == 1:
            return preds.reshape(-1)
        return preds
