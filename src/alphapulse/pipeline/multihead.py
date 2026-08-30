from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd

from ..evaluation.metrics import rank_normalize
from ..experiments.data import require_meta_model_from_benchmarks
from ..models.base import BaseModel
from ..preprocessors.base import BasePreprocessor, TrainEvalPreprocessor
from ..preprocessors.era_stable import EraStableFeatureSelector
from .ensemble import EnsembleStrategy
from .neutralizer import (
    FeatureNeutralizer,
    MetaModelNeutralizer,
    apply_prediction_neutralization,
)
from .row_utils import (
    blend_with_benchmark,
    filter_invalid_rows,
    filter_nan_rows,
    invalid_row_mask,
    protected_metadata_frame,
    reattach_protected_columns,
    safe_median,
    select_required_features,
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
        neutralize_proportion: float = 0.0,
        neutralize_features: list[str] | None = None,
        meta_neutralize_proportion: float = 0.0,
    ) -> None:
        if not heads:
            raise ValueError("heads must be non-empty")
        self.global_preprocessors = global_preprocessors
        self.heads = heads
        self.feature_columns = feature_columns
        self.benchmark_blend_weight = benchmark_blend_weight
        self.neutralize_proportion = float(neutralize_proportion)
        self.neutralize_features = neutralize_features
        self._neutralizer = (
            FeatureNeutralizer(proportion=self.neutralize_proportion)
            if self.neutralize_proportion > 0.0
            else None
        )
        self.meta_neutralize_proportion = float(meta_neutralize_proportion)
        self._meta_neutralizer = (
            MetaModelNeutralizer(proportion=self.meta_neutralize_proportion)
            if self.meta_neutralize_proportion > 0.0
            else None
        )
        self._ensemble = EnsembleStrategy(
            method=ensemble_method, params=ensemble_params or {}
        )

    @property
    def ensemble_method(self) -> str:
        return self._ensemble.method

    @property
    def ensemble_weights(self) -> list[float] | None:
        weights = self._ensemble.weights
        if weights is None:
            return None
        return [float(weight) for weight in weights]

    def _transform_global(
        self, X: pd.DataFrame, *, fit: bool, y: pd.Series | None
    ) -> pd.DataFrame:
        era_meta = protected_metadata_frame(X)
        for pp in self.global_preprocessors:
            if isinstance(pp, TrainEvalPreprocessor):
                pp.train() if fit else pp.eval()
            if fit:
                if isinstance(pp, EraStableFeatureSelector) and era_meta is not None:
                    pp.fit(X, y, eras=era_meta["era"])
                else:
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
                if isinstance(pp, EraStableFeatureSelector) and era_meta is not None:
                    pp.fit(X_h, y, eras=era_meta["era"])
                else:
                    pp.fit(X_h, y)
            X_h = pp.transform(X_h)
            X_h = reattach_protected_columns(X_h, era_meta)
        return reattach_protected_columns(X_h, era_meta)

    def _with_era_column(
        self,
        frame: pd.DataFrame,
        era: pd.Series | None,
    ) -> pd.DataFrame:
        if era is None or "era" in frame.columns:
            return frame
        out = frame.copy()
        out["era"] = era.loc[frame.index]
        return out

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        ensemble_meta_preds: pd.Series | np.ndarray | None = None,
        **model_train_kwargs: Any,
    ) -> dict[str, float]:
        era_train = model_train_kwargs.pop("era_train", None)
        era_val = model_train_kwargs.pop("era_val", None)
        per_model_kwargs = model_train_kwargs.pop("model_train_kwargs_by_index", None)
        X = self._with_era_column(X, era_train)
        if X_val is not None:
            X_val = self._with_era_column(X_val, era_val)

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
        ensemble_meta_series: pd.Series | None = None
        if X_val is not None:
            if ensemble_meta_preds is not None:
                if isinstance(ensemble_meta_preds, pd.Series):
                    if ensemble_meta_preds.index.has_duplicates:
                        raise ValueError(
                            "ensemble_meta_preds index must not contain duplicates"
                        )
                    missing = X_val.index.difference(ensemble_meta_preds.index)
                    extra = ensemble_meta_preds.index.difference(X_val.index)
                    if len(missing) > 0 or len(extra) > 0:
                        raise ValueError(
                            "ensemble_meta_preds row IDs must exactly match X_val"
                        )
                    ensemble_meta_series = ensemble_meta_preds.reindex(X_val.index)
                else:
                    meta_array = np.asarray(ensemble_meta_preds, dtype=np.float64)
                    if len(meta_array) != len(X_val):
                        raise ValueError(
                            "ensemble_meta_preds length must match X_val length"
                        )
                    ensemble_meta_series = pd.Series(meta_array, index=X_val.index)
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
        for head_index, head in enumerate(self.heads):
            X_tr = self._head_matrix(X_fit, head, fit=True, y=y)
            X_va_h = (
                self._head_matrix(X_val_g, head, fit=False, y=None)
                if X_val_g is not None
                else None
            )
            train_kwargs = dict(model_train_kwargs)
            if per_model_kwargs and head_index < len(per_model_kwargs):
                train_kwargs.update(per_model_kwargs[head_index])
            metrics = head.model.train(
                X_tr, y, X_val=X_va_h, y_val=y_val, **train_kwargs
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
                eras_val=(
                    era_val.loc[X_val_g.index]
                    if era_val is not None and X_val_g is not None
                    else None
                ),
                meta_model_preds=(
                    ensemble_meta_series.loc[X_val_g.index].to_numpy(dtype=np.float64)
                    if ensemble_meta_series is not None and X_val_g is not None
                    else None
                ),
            )

        return all_metrics

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
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

    def predict(
        self,
        X: pd.DataFrame,
        eras: pd.Series | None = None,
        meta_model: np.ndarray | pd.Series | None = None,
    ) -> np.ndarray:
        raw_preds = self._predict_raw(X)
        return apply_prediction_neutralization(
            raw_preds,
            X,
            eras=eras,
            feature_columns=self.feature_columns,
            neutralize_features=self.neutralize_features,
            feature_neutralizer=self._neutralizer,
            meta_model=meta_model,
            meta_neutralizer=self._meta_neutralizer,
        )

    def to_numerai_predict(
        self,
        benchmark_col: str | None = None,
    ) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        pipeline = self
        feature_columns = self.feature_columns or []
        blend_weight = self.benchmark_blend_weight
        bench_col = benchmark_col
        use_neutralization = pipeline._neutralizer is not None
        use_meta_neutralization = pipeline._meta_neutralizer is not None

        def predict(
            live_features: pd.DataFrame,
            live_benchmark_models: pd.DataFrame,
        ) -> pd.DataFrame:
            X = select_required_features(live_features, feature_columns)
            eras = (
                live_features["era"]
                if (use_neutralization or use_meta_neutralization)
                and "era" in live_features.columns
                else None
            )
            meta_model = (
                require_meta_model_from_benchmarks(
                    live_benchmark_models, live_features.index
                )
                if use_meta_neutralization
                else None
            )
            raw_preds = pipeline.predict(X, eras=eras, meta_model=meta_model)
            ranked = rank_normalize(raw_preds)

            ranked = blend_with_benchmark(
                ranked,
                live_benchmark_models,
                live_features.index,
                benchmark_column=bench_col,
                weight=blend_weight,
            )

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
