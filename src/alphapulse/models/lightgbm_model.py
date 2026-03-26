from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseModel


class LightGBMModel(BaseModel):
    def __init__(
        self,
        params: dict[str, Any] | None = None,
        n_estimators: int = 2000,
        early_stopping_rounds: int = 100,
        name: str | None = "LightGBM",
    ) -> None:
        super().__init__(name)
        self.params: dict[str, Any] = params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "max_depth": 5,
            "learning_rate": 0.01,
            "num_leaves": 31,
            "min_child_samples": 200,
            "colsample_bytree": 0.3,
            "subsample": 0.7,
            "subsample_freq": 1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "n_jobs": -1,
        }
        self.n_estimators = int(n_estimators)
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
        import lightgbm as lgb

        n_est = int(n_rounds if n_rounds is not None else self.n_estimators)
        es = int(
            early_stopping_rounds
            if early_stopping_rounds is not None
            else self.early_stopping_rounds
        )

        feat_train = X_train.select_dtypes(include=[np.number])
        if feat_train.shape[1] == 0:
            raise ValueError("LightGBMModel: no numeric feature columns found.")

        dtrain = lgb.Dataset(feat_train, label=y_train, free_raw_data=False)

        evals_result: dict[str, Any] = {}
        callbacks: list[Any] = [
            lgb.log_evaluation(period=0),
            lgb.record_evaluation(evals_result),
        ]
        valid_sets = [dtrain]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            feat_val = X_val.select_dtypes(include=[np.number])
            dval = lgb.Dataset(
                feat_val, label=y_val, reference=dtrain, free_raw_data=False
            )
            valid_sets.append(dval)
            valid_names.append("eval")
            callbacks.append(lgb.early_stopping(stopping_rounds=es, verbose=False))

        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=n_est,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.is_trained = True

        metrics: dict[str, float] = {}
        for ds_name, ds_metrics in evals_result.items():
            for metric_name, values in ds_metrics.items():
                if values:
                    metrics[f"{ds_name}_{metric_name}"] = float(values[-1])
        metrics["best_iteration"] = float(self.model.best_iteration)
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained!")

        feat = X.select_dtypes(include=[np.number])
        return np.asarray(
            self.model.predict(feat, num_iteration=self.model.best_iteration),
            dtype=np.float64,
        )

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: Path) -> "LightGBMModel":
        import lightgbm as lgb

        self.model = lgb.Booster(model_file=str(path))
        self.is_trained = True
        return self
