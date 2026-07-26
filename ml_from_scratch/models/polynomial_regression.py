from __future__ import annotations

import numpy as np

from .linear_regression import LinearRegression


class PolynomialRegression:
    def __init__(self, degree: int = 2, learning_rate: float = 0.01, epochs: int = 1000):
        self.degree = degree
        self.linear = LinearRegression(learning_rate=learning_rate, epochs=epochs)

    def _expand(self, X):
        X = np.asarray(X, dtype=float)
        features = [X]
        for power in range(2, self.degree + 1):
            features.append(X**power)
        return np.concatenate(features, axis=1)

    def fit(self, X, y):
        self.linear.fit(self._expand(X), y)
        return self

    def predict(self, X):
        return self.linear.predict(self._expand(X))

    @property
    def losses(self):
        return self.linear.losses
