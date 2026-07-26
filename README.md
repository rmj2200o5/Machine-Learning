# Machine-Learning

Starter framework for building machine-learning models from scratch in Python.

## Included model families

- Linear Regression
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Neural Network Regressor (built from shared `Neuron` blocks)
- Simple CNN Regressor (built from shared `Neuron` blocks)

## Project layout

- `ml_from_scratch/core`
  - `neuron.py`: foundational neuron building block
  - `layer.py`: dense layer made of neurons
  - `base.py`: shared model interface and data conversion
- `ml_from_scratch/models`
  - starter implementations for the model families above
- `ml_from_scratch/visualization/plotting.py`
  - matplotlib helpers for training curves and prediction plots
- `tests/`
  - smoke tests for shared framework and model wiring

## Example usage

```python
import numpy as np
from ml_from_scratch import LinearRegression, NeuralNetworkRegressor
from ml_from_scratch.visualization.plotting import plot_training_curve

X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([3.0, 5.0, 7.0, 9.0])

model = LinearRegression(learning_rate=0.01, epochs=300)
model.fit(X, y)
preds = model.predict(X)

nn = NeuralNetworkRegressor(layer_sizes=[1, 4, 1], learning_rate=0.01, epochs=100)
n.fit(X, y)
plot_training_curve(nn.losses)
```
