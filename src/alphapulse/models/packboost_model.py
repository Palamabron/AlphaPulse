from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.metrics import per_era_correlation
from .base import BaseModel, _numeric
from .packboost_backend import PackBoostTrainer, require_packboost_cuda

_MIN_TRAIN_ROWS = 10
_MIN_VAL_ROWS = 10


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
        device: str = "cuda",
        max_depth: int = 7,
        nfolds: int = 8,
        lr: float = 0.07,
        l2: float = 100_000.0,
        nfeatsets: int = 32,
        seed: int = 42,
        name: str = "Packboost",
    ) -> None:
        super().__init__(name)
        self.base_params = base_params or {}
        self.boost_params = boost_params or {"max_depth": 3}
        self.era_column = era_column
        self.n_worst_eras = int(n_worst_eras)
        self.boost_weight = float(boost_weight)
        self.n_rounds_base = int(n_rounds_base)
        self.early_stopping_rounds_base = int(early_stopping_rounds_base)
        self.n_rounds_boost = int(n_rounds_boost)
        self.early_stopping_rounds_boost = int(early_stopping_rounds_boost)
        self.device = device
        self.max_depth = int(max_depth)
        self.nfolds = int(nfolds)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.nfeatsets = int(nfeatsets)
        self.seed = int(seed)

        self._base_trainer: PackBoostTrainer | None = None
        self._era_trainers: dict[Any, PackBoostTrainer] = {}
        self._feature_columns: list[str] | None = None
        self._worst_eras: list[Any] = []

    def _feature_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._feature_columns is not None:
            return X[self._feature_columns]
        return _numeric(X)

    def _make_trainer(self, *, max_depth: int | None = None) -> PackBoostTrainer:
        return PackBoostTrainer(
            device=self.device,
            max_depth=max_depth if max_depth is not None else self.max_depth,
            nfolds=self.nfolds,
            lr=self.lr,
            l2=self.l2,
            nfeatsets=self.nfeatsets,
            seed=self.seed,
        )

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
        require_packboost_cuda(device=self.device)

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

        n_rounds_use = n_rounds if n_rounds is not None else self.n_rounds_base
        feat_val: pd.DataFrame | None = None
        era_val_series: pd.Series | None = None
        if X_val is not None and y_val is not None:
            feat_val = self._feature_matrix(X_val)
            if self.era_column in X_val.columns:
                era_val_series = X_val[self.era_column]

        self._base_trainer = self._make_trainer()
        self._base_trainer.fit(
            feat,
            y_train,
            era=era_train,
            val_features=feat_val,
            val_target=y_val,
            rounds=n_rounds_use,
        )
        base_pred = self._base_trainer.predict(feat)
        per_era_corr = per_era_correlation(y_train, base_pred, era_train)
        valid_corr = per_era_corr.dropna()
        if len(valid_corr) == 0:
            self._worst_eras = []
            self.is_trained = True
            return {"n_boost_eras": 0.0}

        ascending_order = per_era_corr.sort_values(ascending=True)
        self._worst_eras = ascending_order.index[: self.n_worst_eras].tolist()

        self._era_trainers = {}
        boost_depth = int(self.boost_params.get("max_depth", 3))
        for era_id in self._worst_eras:
            mask = era_train == era_id
            if int(mask.sum()) < _MIN_TRAIN_ROWS:
                continue
            x_era = feat.loc[mask]
            y_era = y_train.loc[mask]
            x_era_fit = x_era
            y_era_fit = y_era
            x_era_val_local: pd.DataFrame | None = None
            y_era_val_local: pd.Series | None = None

            if feat_val is not None and era_val_series is not None:
                mask_val = era_val_series == era_id
                if int(mask_val.sum()) >= _MIN_VAL_ROWS:
                    x_era_val_local = feat_val.loc[mask_val]
                    y_era_val_local = y_val.loc[mask_val] if y_val is not None else None

            if x_era_val_local is None:
                n = len(x_era)
                if n <= 2:
                    trainer = self._make_trainer(max_depth=boost_depth)
                    trainer.fit(x_era, y_era, rounds=self.n_rounds_boost)
                    self._era_trainers[era_id] = trainer
                    continue
                n_val_era = max(1, min(int(n * 0.1), n - 1))
                x_era_fit = x_era.iloc[: n - n_val_era]
                y_era_fit = y_era.iloc[: n - n_val_era]
                x_era_val_local = x_era.iloc[n - n_val_era :]
                y_era_val_local = y_era.iloc[n - n_val_era :]

            trainer = self._make_trainer(max_depth=boost_depth)
            trainer.fit(
                x_era_fit,
                y_era_fit,
                val_features=x_era_val_local,
                val_target=y_era_val_local,
                rounds=self.n_rounds_boost,
            )
            self._era_trainers[era_id] = trainer

        self.is_trained = True
        self.model = {
            "base_trainer": self._base_trainer,
            "era_trainers": self._era_trainers,
            "feature_columns": self._feature_columns,
            "worst_eras": self._worst_eras,
        }
        return {"n_boost_eras": float(len(self._era_trainers))}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self._base_trainer is None:
            raise ValueError("PackboostModel is not trained.")

        feat = self._feature_matrix(X)
        out = self._base_trainer.predict(feat)
        if self.era_column in X.columns and self._era_trainers:
            era_values = X[self.era_column]
            for era_id, trainer in self._era_trainers.items():
                mask = (era_values == era_id).values
                if not mask.any():
                    continue
                boost_pred = trainer.predict(feat.loc[mask])
                out[mask] += self.boost_weight * boost_pred
        return out

    def save(self, path: Path) -> None:
        import cloudpickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "base_trainer": self._base_trainer,
            "era_trainers": self._era_trainers,
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
        if "base" in state or "era_models" in state:
            raise ValueError(
                "Legacy XGBoost Packboost checkpoint is not supported. "
                "Retrain with PackBoost CUDA."
            )
        self._base_trainer = state["base_trainer"]
        self._era_trainers = state["era_trainers"]
        self._feature_columns = state["feature_columns"]
        self._worst_eras = state["worst_eras"]
        self.era_column = state.get("era_column", "era")
        self.boost_weight = state.get("boost_weight", 0.3)
        self.model = state
        self.is_trained = True
        return self
