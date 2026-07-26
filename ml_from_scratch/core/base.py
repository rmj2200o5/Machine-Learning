from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Shared model interface and data conversion helpers."""

    @staticmethod
    def to_numpy(data: Iterable) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return data.astype(float)
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.to_numpy(dtype=float)
        return np.asarray(data, dtype=float)

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError
