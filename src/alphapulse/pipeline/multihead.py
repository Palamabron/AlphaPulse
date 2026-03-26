from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd

from ..evaluation.metrics import rank_normalize
from ..models.base import BaseModel
from ..preprocessors.base import BasePreprocessor
from .ensemble import EnsembleStrategy


class HeadSpec:
    def __init__(
        self,
        model: BaseModel,
        input_columns: list[str] | None,
        input_group: str | None,
        local_preprocessors: list[BasePreprocessor],
        feature_groups: dict[str, list[str]],
    ) -> None:
        self.model = model
        self.input_columns = input_columns
        self.input_group = input_group
        self.local_preprocessors = local_preprocessors
        self.feature_groups = feature_groups

    def resolved_columns(self, X: pd.DataFrame) -> list[str]:
        if self.input_columns is not None:
            return list(self.input_columns)
        if self.input_group is not None:
            cols = self.feature_groups.get(self.input_group, [])
            found = [c for c in cols if c in X.columns]
            if not found:
                raise ValueError(
                    f"No columns from input_group {self.input_group!r} present in data"
                )
            return found
        return list(X.columns)


class MultiHeadPipeline:
    def __init__(
        self,
        global_preprocessors: list[BasePreprocessor],
        heads: list[HeadSpec],
        feature_columns: list[str] | None = None,
        ensemble_method: Literal["single", "weighted", "stacking"] = "single",
        ensemble_params: dict[str, Any] | None = None,
        benchmark_blend_weight: float = 0.0,
    ) -> None:
        if not heads:
            raise ValueError("heads must be non-empty")
        self.global_preprocessors = global_preprocessors
        self.heads = heads
        self.feature_columns = feature_columns
        self.benchmark_blend_weight = benchmark_blend_weight
        self._ensemble = EnsembleStrategy(
            method=ensemble_method, params=ensemble_params or {}
        )

    def _transform_global(self, X: pd.DataFrame) -> pd.DataFrame:
        for pp in self.global_preprocessors:
            X = pp.transform(X)
        return X

    def _head_matrix(
        self, X_global: pd.DataFrame, head: HeadSpec, fit: bool, y: pd.Series | None
    ) -> pd.DataFrame:
        cols = head.resolved_columns(X_global)
        missing = [c for c in cols if c not in X_global.columns]
        if missing:
            raise ValueError(
                f"Missing columns for head {head.model.name}: {missing[:10]}..."
            )
        X_h = X_global[cols].copy()
        for pp in head.local_preprocessors:
            if fit:
                pp.fit(X_h, y)
            X_h = pp.transform(X_h)
        return X_h

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **model_train_kwargs: Any,
    ) -> dict[str, float]:
        if self.feature_columns is None:
            self.feature_columns = list(X.columns)

        X_fit = X
        for pp in self.global_preprocessors:
            pp.fit(X_fit, y)
            X_fit = pp.transform(X_fit)

        X_val_g: pd.DataFrame | None = None
        if X_val is not None:
            X_val_g = self._transform_global(X_val)

        all_metrics: dict[str, float] = {}
        for head in self.heads:
            X_tr = self._head_matrix(X_fit, head, fit=True, y=y)
            X_va_h = (
                self._head_matrix(X_val_g, head, fit=False, y=None)
                if X_val_g is not None
                else None
            )
            metrics = head.model.train(
                X_tr, y, X_val=X_va_h, y_val=y_val, **model_train_kwargs
            )
            for k, v in metrics.items():
                all_metrics[f"{head.model.name}_{k}"] = v

        if len(self.heads) > 1:

            def get_val_preds() -> np.ndarray:
                return np.column_stack(
                    [
                        h.model.predict(
                            self._head_matrix(X_val_g, h, fit=False, y=None)
                        )
                        for h in self.heads
                    ]
                )

            self._ensemble.fit(
                n_models=len(self.heads),
                get_val_predictions=get_val_preds if X_val_g is not None else None,
                y_val=y_val,
            )

        return all_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_g = self._transform_global(X)
        if len(self.heads) == 1:
            X_h = self._head_matrix(X_g, self.heads[0], fit=False, y=None)
            return self.heads[0].model.predict(X_h)

        preds = np.column_stack(
            [
                h.model.predict(self._head_matrix(X_g, h, fit=False, y=None))
                for h in self.heads
            ]
        )
        return self._ensemble.combine(preds)

    def to_numerai_predict(
        self,
        benchmark_col: str | None = None,
    ) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        pipeline = self
        feature_columns = self.feature_columns or []
        blend_weight = self.benchmark_blend_weight
        bench_col = benchmark_col

        def predict(
            live_features: pd.DataFrame,
            live_benchmark_models: pd.DataFrame,
        ) -> pd.DataFrame:
            X = live_features[feature_columns]
            raw_preds = pipeline.predict(X)
            ranked = rank_normalize(raw_preds)

            if (
                blend_weight > 0.0
                and bench_col
                and bench_col in live_benchmark_models.columns
            ):
                bench = live_benchmark_models[bench_col].values
                ranked = (1.0 - blend_weight) * ranked + blend_weight * bench

            return pd.DataFrame({"prediction": ranked}, index=live_features.index)

        return predict

    def save_pipeline(self, path: Path) -> None:
        import cloudpickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load_pipeline(cls, path: Path) -> Self:
        import cloudpickle

        with open(path, "rb") as f:
            loaded = cloudpickle.load(f)
            if not isinstance(loaded, cls):
                raise TypeError(
                    f"Expected a {cls.__name__} object, got {type(loaded).__name__}"
                )
            return loaded
