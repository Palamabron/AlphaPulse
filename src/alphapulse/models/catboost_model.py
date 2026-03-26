from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseModel


class CatBoostModel(BaseModel):
    def __init__(
        self,
        params: dict[str, Any] | None = None,
        iterations: int = 2000,
        early_stopping_rounds: int = 100,
        name: str | None = "CatBoost",
    ) -> None:
        super().__init__(name)
        self.params: dict[str, Any] = params or {
            "loss_function": "RMSE",
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,
            "min_data_in_leaf": 200,
            "random_strength": 1.0,
            "bagging_temperature": 0.5,
            "colsample_bylevel": 0.3,
            "verbose": 0,
            "thread_count": -1,
            "allow_writing_files": False,
        }
        self.iterations = int(iterations)
        self.early_stopping_rounds = int(early_stopping_rounds)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        n_rounds: int | None = None,
        early_stopping_rounds: int | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        from catboost import CatBoostRegressor, Pool

        iters = int(n_rounds if n_rounds is not None else self.iterations)
        es = int(
            early_stopping_rounds
            if early_stopping_rounds is not None
            else self.early_stopping_rounds
        )

        full_params = {**self.params, "iterations": iters}
        cb_model = CatBoostRegressor(**full_params)

        feat_train = X_train.select_dtypes(include=[np.number])
        if feat_train.shape[1] == 0:
            raise ValueError("CatBoostModel: no numeric feature columns found.")

        train_pool = Pool(feat_train, label=y_train)
        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            feat_val = X_val.select_dtypes(include=[np.number])
            eval_pool = Pool(feat_val, label=y_val)
            fit_kwargs["eval_set"] = eval_pool
            fit_kwargs["early_stopping_rounds"] = es

        cb_model.fit(train_pool, **fit_kwargs)

        self.model = cb_model
        self.is_trained = True

        metrics: dict[str, float] = {}
        best_score = cb_model.get_best_score()
        for ds_name, ds_metrics in best_score.items():
            for metric_name, value in ds_metrics.items():
                metrics[f"{ds_name}_{metric_name}"] = float(value)
        metrics["best_iteration"] = float(cb_model.get_best_iteration() or iters)
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained!")
        feat = X.select_dtypes(include=[np.number])
        return np.asarray(self.model.predict(feat), dtype=np.float64)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: Path) -> "CatBoostModel":
        from catboost import CatBoostRegressor

        self.model = CatBoostRegressor()
        self.model.load_model(str(path))
        self.is_trained = True
        return self
