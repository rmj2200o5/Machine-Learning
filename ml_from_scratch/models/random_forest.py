from __future__ import annotations

import numpy as np

from ..core.base import BaseModel
from .decision_tree import DecisionTreeRegressor


class RandomForestRegressor(BaseModel):
    def __init__(self, n_trees: int = 5, max_depth: int = 4, min_samples_split: int = 2, seed: int = 42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.trees: list[DecisionTreeRegressor] = []

    def fit(self, X, y):
        X = self.to_numpy(X)
        y = self.to_numpy(y).reshape(-1)
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.seed)
        self.trees = []

        for _ in range(self.n_trees):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = self.to_numpy(X)
        tree_preds = np.asarray([tree.predict(X) for tree in self.trees], dtype=float)
        return np.mean(tree_preds, axis=0)
