from __future__ import annotations

import numpy as np

from ..core.base import BaseModel


class LinearRegression(BaseModel):
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.losses: list[float] = []

    def fit(self, X, y):
        X = self.to_numpy(X)
        y = self.to_numpy(y).reshape(-1)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.losses = []

        for _ in range(self.epochs):
            preds = X @ self.weights + self.bias
            errors = preds - y

            dw = (2 / n_samples) * (X.T @ errors)
            db = (2 / n_samples) * np.sum(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.losses.append(float(np.mean(errors**2)))
        return self

    def predict(self, X):
        X = self.to_numpy(X)
        return X @ self.weights + self.bias
