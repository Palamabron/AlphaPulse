from typing import Self

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

from .base import _PROTECTED_COLS, BasePreprocessor


class PCAPreprocessor(BasePreprocessor):
    def __init__(
        self,
        n_components: int | float | str | None = None,
        random_state: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.n_components = n_components
        self.random_state = random_state
        self._numeric_cols: list[str] = []
        self._pca = PCA(n_components=n_components, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        feat_cols = [c for c in X.columns if c not in _PROTECTED_COLS]
        self._numeric_cols = list(
            X[feat_cols].select_dtypes(include=[np.number]).columns
        )
        if not self._numeric_cols:
            raise ValueError("PCAPreprocessor: no numeric columns found.")
        self._pca.fit(X[self._numeric_cols].values)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted!")
        out = self._pca.transform(X[self._numeric_cols].values)
        cols = [f"pca_{i}" for i in range(out.shape[1])]
        return pd.DataFrame(out, columns=cols, index=X.index)


class TruncatedSVDPreprocessor(BasePreprocessor):
    def __init__(
        self,
        n_components: int = 10,
        random_state: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.n_components = n_components
        self.random_state = random_state
        self._numeric_cols: list[str] = []
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        feat_cols = [c for c in X.columns if c not in _PROTECTED_COLS]
        self._numeric_cols = list(
            X[feat_cols].select_dtypes(include=[np.number]).columns
        )
        if not self._numeric_cols:
            raise ValueError("TruncatedSVDPreprocessor: no numeric columns found.")
        self._svd.fit(X[self._numeric_cols].values)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted!")
        out = self._svd.transform(X[self._numeric_cols].values)
        cols = [f"svd_{i}" for i in range(out.shape[1])]
        return pd.DataFrame(
            np.asarray(out, dtype=np.float64), columns=cols, index=X.index
        )
