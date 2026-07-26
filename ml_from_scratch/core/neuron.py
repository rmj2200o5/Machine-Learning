from __future__ import annotations

import numpy as np


class Neuron:
    """Single trainable neuron unit used by NN and CNN."""

    def __init__(self, n_inputs: int, activation: str = "identity", seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.1, size=n_inputs)
        self.bias = 0.0
        self.activation = activation
        self._last_input = None
        self._last_z = None

    def _activate(self, z: np.ndarray | float) -> np.ndarray | float:
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        return z

    def _activation_grad(self, z: np.ndarray | float) -> np.ndarray | float:
        if self.activation == "relu":
            return float(z > 0)
        if self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        return np.ones_like(z, dtype=float)

    def forward(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        self._last_input = x
        self._last_z = float(np.dot(self.weights, x) + self.bias)
        return float(self._activate(self._last_z))

    def backward(self, grad_output: float, learning_rate: float) -> np.ndarray:
        grad_z = float(grad_output * self._activation_grad(self._last_z))
        grad_weights = grad_z * self._last_input
        grad_bias = grad_z
        old_weights = self.weights.copy()

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_z * old_weights
