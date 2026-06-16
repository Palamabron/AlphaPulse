import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge

from .base import BaseModel


class EraEnsembleModel(BaseModel):
    """Era-partitioned ensemble model (V3X-style).

    Partitions training eras into n_subs groups, trains one sub-model per
    group on its era partition, collects each sub-model's predictions on the
    *full* training set (creating temporal diversity), then fits a Ridge
    meta-learner to combine them.

    Falls back to single-model training when era data is unavailable, so
    existing tests that don't pass era columns continue to work.
    """

    def __init__(
        self,
        base_model_factory: Callable[[], BaseModel],
        n_subs: int = 10,
        era_column: str = "era",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "EraEnsemble")
        self.base_model_factory = base_model_factory
        self.n_subs = n_subs
        self.era_column = era_column
        self._sub_models: list[BaseModel] = []
        self._meta_model: Ridge | None = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        era_train: pd.Series | None = kwargs.pop("era_train", None)
        if era_train is None and self.era_column in X_train.columns:
            era_train = X_train[self.era_column]

        if era_train is None:
            warnings.warn(
                f"EraEnsembleModel: era column {self.era_column!r} not found after "
                "preprocessing; falling back to single-model training.",
                UserWarning,
                stacklevel=2,
            )
            logger.info(
                "{}: no era column, training single base model",
                self.name,
            )
            model = self.base_model_factory()
            metrics = model.train(X_train, y_train, X_val=X_val, y_val=y_val, **kwargs)
            self._sub_models = [model]
            self._meta_model = None
            self.is_trained = True
            return metrics

        era_val: pd.Series | None = None
        if X_val is not None and self.era_column in X_val.columns:
            era_val = X_val[self.era_column]

        unique_eras = np.sort(era_train.unique())
        n_parts = min(self.n_subs, len(unique_eras))
        era_partitions = np.array_split(unique_eras, n_parts)
        logger.info(
            "{}: training {} era partitions ({} unique eras)",
            self.name,
            n_parts,
            len(unique_eras),
        )

        sub_preds: list[np.ndarray] = []
        self._sub_models = []

        for i, era_group in enumerate(era_partitions):
            if len(era_group) == 0:
                continue
            mask = era_train.isin(era_group)
            X_sub = X_train[mask]
            y_sub = y_train[mask]
            logger.info(
                "{} sub {}/{}: rows={} eras={}",
                self.name,
                i + 1,
                n_parts,
                len(X_sub),
                len(era_group),
            )

            X_sub_val: pd.DataFrame | None = None
            y_sub_val: pd.Series | None = None
            if X_val is not None and era_val is not None:
                val_mask = era_val.isin(era_group)
                if val_mask.any():
                    X_sub_val = X_val[val_mask]
                    y_sub_val = y_val[val_mask] if y_val is not None else None

            model = self.base_model_factory()
            model.name = f"{self.name}_sub{i}"
            model.train(
                X_sub,
                y_sub,
                X_val=X_sub_val,
                y_val=y_sub_val,
                **kwargs,
            )
            self._sub_models.append(model)

            sub_preds.append(model.predict(X_train))

        logger.info(
            "{}: fitting Ridge meta-learner on {} sub-models",
            self.name,
            len(self._sub_models),
        )
        X_meta = np.column_stack(sub_preds)
        self._meta_model = Ridge(alpha=100.0).fit(X_meta, y_train)
        self.is_trained = True
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        preds = [m.predict(X) for m in self._sub_models]
        if self._meta_model is None:
            return preds[0]
        result: np.ndarray = self._meta_model.predict(np.column_stack(preds))
        return result

    def _require_trained(self) -> None:
        if not self.is_trained or not self._sub_models:
            raise RuntimeError(f"{self.name} is not trained.")

    def save(self, path: Path) -> None:
        import cloudpickle

        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

    def load(self, path: Path) -> "EraEnsembleModel":
        import cloudpickle

        with open(path, "rb") as f:
            loaded = cloudpickle.load(f)
        if not isinstance(loaded, EraEnsembleModel):
            raise TypeError(
                f"Expected an EraEnsembleModel object, got {type(loaded).__name__}"
            )
        self.__dict__.update(loaded.__dict__)
        return self
