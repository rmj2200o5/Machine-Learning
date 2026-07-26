from __future__ import annotations

import numpy as np

from .neuron import Neuron


class DenseLayer:
    """Dense layer implemented as a list of neurons."""

    def __init__(self, n_inputs: int, n_neurons: int, activation: str = "identity"):
        self.neurons = [Neuron(n_inputs=n_inputs, activation=activation) for _ in range(n_neurons)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([neuron.forward(x) for neuron in self.neurons], dtype=float)

    def backward(self, grad_outputs: np.ndarray, learning_rate: float) -> np.ndarray:
        grad_inputs = np.zeros_like(self.neurons[0].weights, dtype=float)
        for neuron, grad_out in zip(self.neurons, grad_outputs):
            grad_inputs += neuron.backward(float(grad_out), learning_rate)
        return grad_inputs
