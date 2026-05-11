from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
            model = self.base_model_factory()
            metrics = model.train(X_train, y_train, X_val=X_val, y_val=y_val, **kwargs)
            self._sub_models = [model]
            self._meta_model = None
            self.is_trained = True
            return metrics

        unique_eras = era_train.unique()
        n_parts = min(self.n_subs, len(unique_eras))
        era_partitions = np.array_split(unique_eras, n_parts)

        sub_preds: list[np.ndarray] = []
        self._sub_models = []

        for i, era_group in enumerate(era_partitions):
            if len(era_group) == 0:
                continue
            mask = era_train.isin(era_group)
            X_sub = X_train[mask]
            y_sub = y_train[mask]

            model = self.base_model_factory()
            model.name = f"{self.name}_sub{i}"
            model.train(X_sub, y_sub, **kwargs)
            self._sub_models.append(model)

            sub_preds.append(model.predict(X_train))

        X_meta = np.column_stack(sub_preds)
        self._meta_model = Ridge(alpha=100.0).fit(X_meta, y_train)
        self.is_trained = True
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = [m.predict(X) for m in self._sub_models]
        if self._meta_model is None:
            return preds[0]
        result: np.ndarray = self._meta_model.predict(np.column_stack(preds))
        return result

    def save(self, path: Path) -> None:
        import cloudpickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

    def load(self, path: Path) -> "EraEnsembleModel":
        import cloudpickle

        with open(path, "rb") as f:
            return cloudpickle.load(f)  # type: ignore[no-any-return]
