from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.base import BaseModel


@dataclass
class _Node:
    feature_idx: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    value: float | None = None


class DecisionTreeRegressor(BaseModel):
    def __init__(self, max_depth: int = 4, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: _Node | None = None

    def fit(self, X, y):
        X = self.to_numpy(X)
        y = self.to_numpy(y).reshape(-1)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_feature, best_threshold, best_loss = None, None, float("inf")
        n_samples, n_features = X.shape
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_var = np.var(y[left_mask]) * left_mask.sum()
                right_var = np.var(y[right_mask]) * right_mask.sum()
                loss = (left_var + right_var) / n_samples
                if loss < best_loss:
                    best_feature, best_threshold, best_loss = feature, float(threshold), float(loss)
        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.allclose(y, y[0]):
            return _Node(value=float(np.mean(y)))

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return _Node(value=float(np.mean(y)))

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature_idx=feature, threshold=threshold, left=left, right=right)

    def _predict_one(self, x: np.ndarray, node: _Node) -> float:
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        X = self.to_numpy(X)
        return np.asarray([self._predict_one(x, self.root) for x in X], dtype=float)
