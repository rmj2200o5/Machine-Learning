from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(losses, title: str = "Training Loss"):
    losses = np.asarray(losses, dtype=float)
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_predictions(y_true, y_pred, title: str = "Predictions vs Actual"):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.7)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.3)
    return fig, ax
