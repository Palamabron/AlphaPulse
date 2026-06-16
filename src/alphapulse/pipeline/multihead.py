from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd

from ..evaluation.metrics import rank_normalize
from ..models.base import BaseModel
from ..preprocessors.base import BasePreprocessor, TrainEvalPreprocessor
from .ensemble import EnsembleStrategy
from .row_utils import (
    filter_invalid_rows,
    filter_nan_rows,
    invalid_row_mask,
    protected_metadata_frame,
    reattach_protected_columns,
    safe_median,
)


class HeadSpec:
    def __init__(
        self,
        model: BaseModel,
        input_columns: list[str] | None,
        input_group: str | None,
        local_preprocessors: list[BasePreprocessor],
        feature_groups: dict[str, list[str]],
        input_groups: list[str] | None = None,
    ) -> None:
        self.model = model
        self.input_columns = input_columns
        self.input_group = input_group
        self.input_groups = input_groups
        self.local_preprocessors = local_preprocessors
        self.feature_groups = feature_groups

    def resolved_columns(self, X: pd.DataFrame) -> list[str]:
        if self.input_columns is not None:
            missing = [c for c in self.input_columns if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Model {self.model.name!r} input_columns references columns not "
                    f"present in data: {missing[:10]}"
                    + (" (and more)" if len(missing) > 10 else "")
                )
            return list(self.input_columns)
        if self.input_groups:
            seen: set[str] = set()
            out: list[str] = []
            for group in self.input_groups:
                if group not in self.feature_groups:
                    available = sorted(self.feature_groups.keys())
                    raise ValueError(
                        f"Model {self.model.name!r} references input_groups containing "
                        f"{group!r}, which is not defined in feature_groups. "
                        f"Available groups: {available}"
                    )
                for col in self.feature_groups[group]:
                    if col in X.columns and col not in seen:
                        seen.add(col)
                        out.append(col)
            if not out:
                raise ValueError(
                    f"Model {self.model.name!r} input_groups={self.input_groups!r} "
                    "resolved to zero columns present in data"
                )
            return out
        if self.input_group is not None:
            if self.input_group not in self.feature_groups:
                available = sorted(self.feature_groups.keys())
                raise ValueError(
                    f"Model {self.model.name!r} references input_group="
                    f"{self.input_group!r}, which is not defined in feature_groups. "
                    f"Available groups: {available}"
                )
            cols = self.feature_groups[self.input_group]
            found = [c for c in cols if c in X.columns]
            if not found:
                raise ValueError(
                    f"Model {self.model.name!r} input_group={self.input_group!r} "
                    f"defines {len(cols)} column(s) but none are present in data. "
                    f"First few expected: {cols[:5]}"
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

    def _transform_global(
        self, X: pd.DataFrame, *, fit: bool, y: pd.Series | None
    ) -> pd.DataFrame:
        era_meta = protected_metadata_frame(X)
        for pp in self.global_preprocessors:
            if isinstance(pp, TrainEvalPreprocessor):
                pp.train() if fit else pp.eval()
            if fit:
                pp.fit(X, y)
            X = pp.transform(X)
            X = reattach_protected_columns(X, era_meta)
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
        era_meta = protected_metadata_frame(X_global)
        for pp in head.local_preprocessors:
            if isinstance(pp, TrainEvalPreprocessor):
                pp.train() if fit else pp.eval()
            if fit:
                pp.fit(X_h, y)
            X_h = pp.transform(X_h)
            X_h = reattach_protected_columns(X_h, era_meta)
        return reattach_protected_columns(X_h, era_meta)

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

        X, y = filter_invalid_rows(X, y)
        if len(X) == 0:
            raise ValueError("No valid training rows after filtering invalid values")

        X_fit = self._transform_global(X, fit=True, y=y)
        X_fit, y = filter_nan_rows(X_fit, y)
        if len(X_fit) == 0:
            raise ValueError("No valid training rows after global preprocessing")

        X_val_g: pd.DataFrame | None = None
        if X_val is not None:
            X_val, y_val = filter_invalid_rows(X_val, y_val)
            if len(X_val) > 0:
                X_val_g = self._transform_global(X_val, fit=False, y=None)
                X_val_g, y_val = filter_nan_rows(X_val_g, y_val)
                if len(X_val_g) == 0:
                    X_val_g = None
                    y_val = None
            else:
                y_val = None

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
        input_invalid_mask = invalid_row_mask(X)

        if input_invalid_mask.any():
            valid_mask = ~input_invalid_mask
            X_valid = X[valid_mask]
            if len(X_valid) == 0:
                return np.full(len(X), 0.0)

            X_g = self._transform_global(X_valid, fit=False, y=None)
            if X_g.isna().any().any():
                post_nan_mask = ~X_g.isna().any(axis=1)
                X_g = X_g[post_nan_mask]
                combined_valid_mask = valid_mask.to_numpy(dtype=bool).copy()
                valid_positions = np.flatnonzero(combined_valid_mask)
                post_arr = np.asarray(post_nan_mask, dtype=bool)
                combined_valid_mask[valid_positions[~post_arr]] = False
            else:
                combined_valid_mask = valid_mask.to_numpy(dtype=bool)

            if len(X_g) == 0:
                return np.full(len(X), 0.0)

            if len(self.heads) == 1:
                valid_preds = self.heads[0].model.predict(
                    self._head_matrix(X_g, self.heads[0], fit=False, y=None)
                )
            else:
                preds = np.column_stack(
                    [
                        h.model.predict(self._head_matrix(X_g, h, fit=False, y=None))
                        for h in self.heads
                    ]
                )
                valid_preds = self._ensemble.combine(preds)

            raw_preds = np.full(len(X), safe_median(valid_preds))
            raw_preds[combined_valid_mask] = valid_preds
            return raw_preds

        X_g = self._transform_global(X, fit=False, y=None)
        if X_g.isna().any().any():
            nan_mask = ~X_g.isna().any(axis=1)
            X_g_valid = X_g[nan_mask]
            if len(X_g_valid) == 0:
                return np.full(len(X), 0.0)

            if len(self.heads) == 1:
                valid_preds = self.heads[0].model.predict(
                    self._head_matrix(X_g_valid, self.heads[0], fit=False, y=None)
                )
            else:
                preds = np.column_stack(
                    [
                        h.model.predict(
                            self._head_matrix(X_g_valid, h, fit=False, y=None)
                        )
                        for h in self.heads
                    ]
                )
                valid_preds = self._ensemble.combine(preds)

            raw_preds = np.full(len(X), safe_median(valid_preds))
            raw_preds[nan_mask.to_numpy(dtype=bool)] = valid_preds
            return raw_preds

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
            X = live_features.reindex(columns=feature_columns, fill_value=0.0)
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
