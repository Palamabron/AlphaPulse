from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from ..evaluation.metrics import per_era_correlation
from .base import BaseModel, _numeric
from .xgboost_model import _make_ray_callbacks


class PackboostModel(BaseModel):
    def __init__(
        self,
        base_params: dict[str, Any] | None = None,
        boost_params: dict[str, Any] | None = None,
        era_column: str = "era",
        n_worst_eras: int = 5,
        boost_weight: float = 0.3,
        n_rounds_base: int = 500,
        early_stopping_rounds_base: int = 50,
        n_rounds_boost: int = 200,
        early_stopping_rounds_boost: int = 30,
        name: str = "Packboost",
    ) -> None:
        super().__init__(name)
        self.base_params = base_params or {
            "max_depth": 5,
            "learning_rate": 0.01,
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }
        self.boost_params = boost_params or {
            "max_depth": 3,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }
        self.era_column = era_column
        self.n_worst_eras = int(n_worst_eras)
        self.boost_weight = float(boost_weight)
        self.n_rounds_base = int(n_rounds_base)
        self.early_stopping_rounds_base = int(early_stopping_rounds_base)
        self.n_rounds_boost = int(n_rounds_boost)
        self.early_stopping_rounds_boost = int(early_stopping_rounds_boost)

        self._base_model: xgb.Booster | None = None
        self._era_models: dict[Any, xgb.Booster] = {}
        self._feature_columns: list[str] | None = None
        self._worst_eras: list[Any] = []

    def _feature_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._feature_columns is not None:
            return X[self._feature_columns]
        return _numeric(X)

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
        era_train = kwargs.get("era_train")
        if era_train is None and self.era_column in X_train.columns:
            era_train = X_train[self.era_column]
        if era_train is None:
            raise ValueError(
                f"Era required for PackboostModel. Pass era_train in kwargs "
                f"or include '{self.era_column}' in X_train."
            )

        feat = self._feature_matrix(X_train)
        self._feature_columns = list(feat.columns)
        dtrain = xgb.DMatrix(feat, label=y_train)

        eval_set: list[tuple[xgb.DMatrix, str]] = []
        evals_result: dict[str, Any] = {}
        n_rounds_use = n_rounds if n_rounds is not None else self.n_rounds_base
        early_use = (
            early_stopping_rounds
            if early_stopping_rounds is not None
            else self.early_stopping_rounds_base
        )

        feat_val: pd.DataFrame | None = None
        era_val_series: pd.Series | None = None
        if X_val is not None and y_val is not None:
            feat_val = self._feature_matrix(X_val)
            dval = xgb.DMatrix(feat_val, label=y_val)
            eval_set = [(dtrain, "train"), (dval, "eval")]
            if self.era_column in X_val.columns:
                era_val_series = X_val[self.era_column]

        callbacks: list[Any] = _make_ray_callbacks()

        self._base_model = xgb.train(
            self.base_params,
            dtrain,
            num_boost_round=n_rounds_use,
            evals=eval_set,
            evals_result=evals_result,
            early_stopping_rounds=early_use if eval_set else None,
            verbose_eval=False,
            callbacks=callbacks,
        )

        base_pred = self._base_model.predict(xgb.DMatrix(feat))
        per_era_corr = per_era_correlation(y_train, base_pred, era_train)
        valid_corr = per_era_corr.dropna()
        if len(valid_corr) == 0:
            self._worst_eras = []
            self.is_trained = True
            return {
                "base_final_rmse": float(
                    evals_result.get("train", {}).get("rmse", [0])[-1]
                )
            }

        ascending_order = per_era_corr.sort_values(ascending=True)
        self._worst_eras = ascending_order.index[: self.n_worst_eras].tolist()

        MIN_ERA_ROWS = 10
        self._era_models = {}
        for era_id in self._worst_eras:
            mask = era_train == era_id
            if int(mask.sum()) < MIN_ERA_ROWS:
                continue
            X_era = feat.loc[mask]
            y_era = y_train.loc[mask]
            eval_set_era: list[tuple[xgb.DMatrix, str]] = []

            if feat_val is not None and era_val_series is not None:
                mask_val = era_val_series == era_id
                if int(mask_val.sum()) >= MIN_ERA_ROWS:
                    X_era_val = feat_val.loc[mask_val]
                    y_era_val = y_val.loc[mask_val] if y_val is not None else None
                    if y_era_val is not None and len(X_era_val) == len(y_era_val):
                        d_era_fit = xgb.DMatrix(X_era, label=y_era)
                        d_era_val = xgb.DMatrix(X_era_val, label=y_era_val)
                        eval_set_era = [(d_era_fit, "train"), (d_era_val, "eval")]

            if not eval_set_era:
                n = len(X_era)
                if n <= 2:
                    d_era_fit = xgb.DMatrix(X_era, label=y_era)
                    booster = xgb.train(
                        self.boost_params,
                        d_era_fit,
                        num_boost_round=self.n_rounds_boost,
                        verbose_eval=False,
                    )
                    self._era_models[era_id] = booster
                    continue

                n_val_era = max(1, int(n * 0.1))
                n_val_era = min(n_val_era, n - 1)
                X_era_fit = X_era.iloc[: n - n_val_era]
                y_era_fit = y_era.iloc[: n - n_val_era]
                X_era_eval = X_era.iloc[n - n_val_era :]
                y_era_eval = y_era.iloc[n - n_val_era :]

                d_era_fit = xgb.DMatrix(X_era_fit, label=y_era_fit)
                d_era_eval = xgb.DMatrix(X_era_eval, label=y_era_eval)
                eval_set_era = [(d_era_fit, "train"), (d_era_eval, "eval")]

            booster = xgb.train(
                self.boost_params,
                eval_set_era[0][0],
                num_boost_round=self.n_rounds_boost,
                evals=eval_set_era,
                early_stopping_rounds=self.early_stopping_rounds_boost
                if eval_set_era
                else None,
                verbose_eval=False,
            )
            self._era_models[era_id] = booster

        self.is_trained = True
        self.model = {
            "base": self._base_model,
            "era_models": self._era_models,
            "feature_columns": self._feature_columns,
            "worst_eras": self._worst_eras,
        }

        metrics: dict[str, float] = {}
        for k, v in evals_result.items():
            for metric_name, values in v.items():
                if values:
                    metrics[f"{k}_{metric_name}"] = float(values[-1])
        metrics["n_boost_eras"] = float(len(self._era_models))
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self._base_model is None:
            raise ValueError("PackboostModel is not trained.")

        feat = self._feature_matrix(X)
        base_pred = self._base_model.predict(xgb.DMatrix(feat))
        out = np.array(base_pred, dtype=np.float64)

        if self.era_column not in X.columns or not self._era_models:
            return out

        era_values = X[self.era_column]
        for era_id, booster in self._era_models.items():
            mask = (era_values == era_id).values
            if not mask.any():
                continue
            era_feat = feat.loc[mask]
            boost_pred = booster.predict(xgb.DMatrix(era_feat))
            out[mask] += self.boost_weight * boost_pred

        return out

    def save(self, path: Path) -> None:
        import cloudpickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "base": self._base_model,
            "era_models": self._era_models,
            "feature_columns": self._feature_columns,
            "worst_eras": self._worst_eras,
            "era_column": self.era_column,
            "boost_weight": self.boost_weight,
        }
        with open(path, "wb") as f:
            cloudpickle.dump(state, f)

    def load(self, path: Path) -> "PackboostModel":
        import cloudpickle

        with open(path, "rb") as f:
            state = cloudpickle.load(f)
        self._base_model = state["base"]
        self._era_models = state["era_models"]
        self._feature_columns = state["feature_columns"]
        self._worst_eras = state["worst_eras"]
        self.era_column = state.get("era_column", "era")
        self.boost_weight = state.get("boost_weight", 0.3)
        self.model = state
        self.is_trained = True
        return self
