from collections.abc import Mapping
from typing import Any

import pandas as pd
import xgboost as xgb

from .model_abstract import ModelAbstract


class ModelXgboost(ModelAbstract):
    def __init__(self) -> None:
        self.model: xgb.Booster | None = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Mapping[str, Any],
        num_boost_round: int = 10,
        **kwargs: Any,
    ) -> xgb.Booster:
        dtrain = xgb.DMatrix(X, label=y)

        self.model = xgb.train(
            params=params, dtrain=dtrain, num_boost_round=num_boost_round, **kwargs
        )
        return self.model

    def finetune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Mapping[str, Any],
        num_boost_round: int = 10,
        **kwargs: Any,
    ) -> xgb.Booster:
        if self.model is None:
            raise RuntimeError("Train initial model")

        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            xgb_model=self.model,
            **kwargs,
        )
        return self.model

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Train a model first")

        dtest = xgb.DMatrix(X)

        preds = self.model.predict(dtest, **kwargs)
        return pd.Series(preds, index=X.index, name="prediction")
