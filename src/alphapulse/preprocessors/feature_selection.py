from typing import Self

import numpy as np
import pandas as pd

from .base import _PROTECTED_COLS, BasePreprocessor


class VarianceFeatureSelector(BasePreprocessor):
    def __init__(
        self,
        keep_fraction: float = 1.0,
        threshold: float = 0.0,
        mode: str = "quantile",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
        self.keep_fraction = keep_fraction
        self.threshold = threshold
        self.mode = mode
        self.selected_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        feat_cols = [c for c in X.columns if c not in _PROTECTED_COLS]
        variances = X[feat_cols].var(axis=0)

        if self.mode == "threshold":
            mask = variances > self.threshold
            self.selected_columns_ = list(X.columns[mask])
        else:
            n_keep = max(1, int(len(X.columns) * self.keep_fraction))
            ranked = variances.sort_values(ascending=False)
            self.selected_columns_ = list(ranked.index[:n_keep])

        if not self.selected_columns_:
            self.selected_columns_ = list(X.columns[:1])

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted!")
        cols = [c for c in self.selected_columns_ if c in X.columns]
        return X[cols].copy()


class LGBMImportanceSelector(BasePreprocessor):
    def __init__(
        self,
        keep_fraction: float = 0.75,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        random_state: int = 42,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
        self.keep_fraction = keep_fraction
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.selected_columns_: list[str] = []
        self.importances_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        if y is None:
            raise ValueError("LGBMImportanceSelector requires y for fit().")

        import lightgbm as lgb

        feat_X = X[[c for c in X.columns if c not in _PROTECTED_COLS]]
        model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=1,
        )
        model.fit(feat_X, y)

        importances = np.asarray(model.feature_importances_, dtype=np.float64)
        self.importances_ = importances

        n_keep = max(1, int(len(feat_X.columns) * self.keep_fraction))
        top_indices = np.argsort(importances)[::-1][:n_keep]
        self.selected_columns_ = [str(feat_X.columns[i]) for i in top_indices]

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted!")
        cols = [c for c in self.selected_columns_ if c in X.columns]
        return X[cols].copy()
