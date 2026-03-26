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
        self._training = True
        self._feature_std: np.ndarray | None = None
        self._rng = np.random.RandomState(seed)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self._feature_std = np.asarray(X.std(axis=0).values, dtype=np.float64)
        self._feature_std = np.where(self._feature_std < 1e-12, 1.0, self._feature_std)
        self._training = True
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted!")
        if not self._training or self.sigma == 0.0:
            return X
        assert self._feature_std is not None
        noise = self._rng.randn(X.shape[0], X.shape[1]) * self.sigma * self._feature_std
        return pd.DataFrame(X.values + noise, columns=X.columns, index=X.index)

    def train(self) -> Self:
        self._training = True
        return self

    def eval(self) -> Self:
        self._training = False
        return self
