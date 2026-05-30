from typing import Self

import numpy as np
import pandas as pd

from .base import BasePreprocessor


class GaussianNoiseInjector(BasePreprocessor):
    def __init__(
        self, sigma: float = 0.01, seed: int = 42, name: str | None = None
    ) -> None:
        super().__init__(name=name)
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.sigma = sigma
        self.seed = seed
        self._training = False
        self._feature_std: np.ndarray | None = None
        self._numeric_cols: list[str] = []
        self._rng = np.random.RandomState(seed)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self._numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not self._numeric_cols:
            raise ValueError("GaussianNoiseInjector: no numeric columns found.")
        self._feature_std = np.asarray(
            X[self._numeric_cols].std(axis=0).values, dtype=np.float64
        )
        self._feature_std = np.where(self._feature_std < 1e-12, 1.0, self._feature_std)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted!")
        if not self._training or self.sigma == 0.0:
            return X
        assert self._feature_std is not None
        out = X.copy()
        noise = (
            self._rng.randn(len(X), len(self._numeric_cols))
            * self.sigma
            * self._feature_std
        )
        out[self._numeric_cols] = (
            out[self._numeric_cols].to_numpy(dtype=np.float64) + noise
        )
        return out

    def train(self) -> Self:
        self._training = True
        return self

    def eval(self) -> Self:
        self._training = False
        return self
