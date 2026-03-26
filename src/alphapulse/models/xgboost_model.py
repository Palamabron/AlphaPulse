from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseModel


def _make_ray_callbacks() -> list[Any]:
    try:
        from ray.tune import session as ray_session

        if ray_session.get_session() is None:
            return []

        def _report_cb(env: Any) -> None:
            eval_rmse: float | None = None
            for item in getattr(env, "evaluation_result_list", []) or []:
                if not isinstance(item, tuple) or len(item) < 3:
                    continue
                dataset, metric_name, value = item[0], item[1], item[2]
                if dataset == "eval" and metric_name == "rmse":
                    eval_rmse = float(value)
                    break
            if eval_rmse is not None:
                ray_session.report({"eval_rmse": eval_rmse})

        return [_report_cb]
    except ImportError:
        return []


class XGBoostModel(BaseModel):
    def __init__(
        self, params: dict[str, Any] | None = None, name: str | None = "XGBoost"
    ) -> None:
        super().__init__(name)
        self.params: dict[str, Any] = params or {
            "max_depth": 5,
            "learning_rate": 0.01,
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        n_rounds: int = 1500,
        early_stopping_rounds: int = 100,
        **kwargs: Any,
    ) -> dict[str, float]:
        feat_train = X_train.select_dtypes(include=[np.number])
        if feat_train.shape[1] == 0:
            raise ValueError("XGBoostModel: no numeric feature columns found.")

        dtrain = xgb.DMatrix(feat_train, label=y_train)

        eval_set: list[tuple[xgb.DMatrix, str]] = []
        if X_val is not None and y_val is not None:
            feat_val = X_val.select_dtypes(include=[np.number])
            dval = xgb.DMatrix(feat_val, label=y_val)
            eval_set = [(dtrain, "train"), (dval, "eval")]

        evals_result: dict[str, Any] = {}
        callbacks: list[Any] = _make_ray_callbacks()

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=n_rounds,
            evals=eval_set,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            verbose_eval=False,
            callbacks=callbacks,
        )
        self.is_trained = True

        metrics: dict[str, float] = {}
        for key, values in evals_result.items():
            first_metric = list(values.keys())[0]
            metrics[f"{key}_final"] = float(values[first_metric][-1])
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained!")

        feat = X.select_dtypes(include=[np.number])
        dtest = xgb.DMatrix(feat)
        return np.asarray(self.model.predict(dtest), dtype=np.float64)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: Path) -> "XGBoostModel":
        booster = xgb.Booster()
        booster.load_model(str(path))
        self.model = booster
        self.is_trained = True
        return self
