import unittest

import numpy as np

from ml_from_scratch import (
    DecisionTreeRegressor,
    LinearRegression,
    NeuralNetworkRegressor,
    Neuron,
    PolynomialRegression,
    RandomForestRegressor,
    SimpleCNNRegressor,
)
from ml_from_scratch.core.layer import DenseLayer


class TestCoreAndModels(unittest.TestCase):
    def test_neuron_and_layer_forward_backward(self):
        neuron = Neuron(n_inputs=3, activation="relu", seed=0)
        out = neuron.forward(np.array([1.0, -1.0, 0.5]))
        self.assertIsInstance(out, float)
        grad_input = neuron.backward(grad_output=1.0, learning_rate=0.01)
        self.assertEqual(grad_input.shape, (3,))

        layer = DenseLayer(n_inputs=3, n_neurons=2, activation="identity")
        layer_out = layer.forward(np.array([0.1, 0.2, 0.3]))
        self.assertEqual(layer_out.shape, (2,))

    def test_regression_models_smoke(self):
        X = np.linspace(0, 1, 10, dtype=float).reshape(-1, 1)
        y = 2 * X.reshape(-1) + 1

        linear = LinearRegression(learning_rate=0.01, epochs=400)
        linear.fit(X, y)
        self.assertEqual(linear.predict(X).shape, (10,))

        poly = PolynomialRegression(degree=2, learning_rate=0.01, epochs=300)
        poly.fit(X, y)
        self.assertEqual(poly.predict(X).shape, (10,))

        tree = DecisionTreeRegressor(max_depth=3)
        tree.fit(X, y)
        self.assertEqual(tree.predict(X).shape, (10,))

        forest = RandomForestRegressor(n_trees=3, max_depth=3)
        forest.fit(X, y)
        self.assertEqual(forest.predict(X).shape, (10,))

    def test_nn_and_cnn_use_neuron_building_block(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

        nn = NeuralNetworkRegressor(layer_sizes=[1, 3, 1], learning_rate=0.01, epochs=25)
        nn.fit(X, y)
        self.assertEqual(nn.predict(X).shape, (4,))
        self.assertTrue(all(hasattr(layer.neurons[0], "weights") for layer in nn.layers))

        X_seq = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ])
        y_seq = np.array([1.0, 2.0, 3.0, 4.0])

        cnn = SimpleCNNRegressor(input_length=4, kernel_size=2, n_filters=2, learning_rate=0.01, epochs=20)
        cnn.fit(X_seq, y_seq)
        self.assertEqual(cnn.predict(X_seq).shape, (4,))
        self.assertEqual(len(cnn.conv.filters), 2)
        self.assertTrue(all(isinstance(filter_neuron, Neuron) for filter_neuron in cnn.conv.filters))


if __name__ == "__main__":
    unittest.main()
