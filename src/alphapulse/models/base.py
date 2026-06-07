from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_PROTECTED_COLS: frozenset[str] = frozenset({"era", "id", "data_type"})


def _numeric(X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in X.columns if c not in _PROTECTED_COLS]
    return X[cols].select_dtypes(include=[np.number])


class BaseModel(ABC):
    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self.model: Any = None
        self.is_trained = False

    def _require_trained(self) -> None:
        if not self.is_trained or self.model is None:
            raise RuntimeError(f"{self.name} is not trained.")

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> "BaseModel": ...
