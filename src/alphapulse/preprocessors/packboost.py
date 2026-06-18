from typing import Any, Self

import pandas as pd

from ..models.packboost_model import PackboostModel
from .base import BasePreprocessor


class PackboostPreprocessor(BasePreprocessor):
    def __init__(
        self,
        era_column: str = "era",
        output_column: str = "packboost_pred",
        base_params: dict[str, Any] | None = None,
        boost_params: dict[str, Any] | None = None,
        n_worst_eras: int = 5,
        boost_weight: float = 0.3,
        n_rounds_base: int = 500,
        early_stopping_rounds_base: int = 50,
        n_rounds_boost: int = 200,
        early_stopping_rounds_boost: int = 30,
        device: str = "cuda",
        max_depth: int = 7,
        nfolds: int = 8,
        lr: float = 0.07,
        l2: float = 100_000.0,
        nfeatsets: int = 32,
        seed: int = 42,
        name: str = "PackboostPreprocessor",
    ) -> None:
        super().__init__(name)
        self.era_column = era_column
        self.output_column = output_column
        self._model = PackboostModel(
            base_params=base_params,
            boost_params=boost_params,
            era_column=era_column,
            n_worst_eras=n_worst_eras,
            boost_weight=boost_weight,
            n_rounds_base=n_rounds_base,
            early_stopping_rounds_base=early_stopping_rounds_base,
            n_rounds_boost=n_rounds_boost,
            early_stopping_rounds_boost=early_stopping_rounds_boost,
            device=device,
            max_depth=max_depth,
            nfolds=nfolds,
            lr=lr,
            l2=l2,
            nfeatsets=nfeatsets,
            seed=seed,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        if y is None:
            raise ValueError("PackboostPreprocessor requires target y in fit().")
        if self.era_column not in X.columns:
            raise ValueError(
                f"PackboostPreprocessor requires column '{self.era_column}' in X."
            )
        self._model.train(X, y)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("PackboostPreprocessor is not fitted.")
        pred = self._model.predict(X)
        out = X.copy()
        out[self.output_column] = pred
        return out
