import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from .base import BasePreprocessor, _PROTECTED_COLS


class StandardScalerPreprocessor(BasePreprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.scaler = StandardScaler()
        self._numeric_cols: list[str] = []

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "StandardScalerPreprocessor":
        feat_cols = [c for c in X.columns if c not in _PROTECTED_COLS]
        self._numeric_cols = list(X[feat_cols].select_dtypes(include=[np.number]).columns)
        if not self._numeric_cols:
            raise ValueError("StandardScalerPreprocessor: no numeric columns to scale.")

        self.scaler.fit(X[self._numeric_cols])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted!")

        out = X.copy()
        scaled = self.scaler.transform(X[self._numeric_cols])
        out[self._numeric_cols] = scaled.astype(np.float32, copy=False)
        return out


class RobustScalerPreprocessor(BasePreprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.scaler = RobustScaler()
        self._numeric_cols: list[str] = []

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "RobustScalerPreprocessor":
        feat_cols = [c for c in X.columns if c not in _PROTECTED_COLS]
        self._numeric_cols = list(X[feat_cols].select_dtypes(include=[np.number]).columns)
        if not self._numeric_cols:
            raise ValueError("RobustScalerPreprocessor: no numeric columns to scale.")

        self.scaler.fit(X[self._numeric_cols])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted!")

        out = X.copy()
        scaled = self.scaler.transform(X[self._numeric_cols])
        out[self._numeric_cols] = scaled.astype(np.float32, copy=False)
        return out
