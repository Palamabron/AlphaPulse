from abc import ABC, abstractmethod
from typing import Protocol, Self, runtime_checkable

import pandas as pd

from ..constants import _PROTECTED_COLS


@runtime_checkable
class TrainEvalPreprocessor(Protocol):
    def train(self) -> Self: ...

    def eval(self) -> Self: ...


class BasePreprocessor(ABC):
    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self.is_fitted = False

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "BasePreprocessor": ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def __repr__(self) -> str:
        return f"{self.name}(fitted={self.is_fitted})"
