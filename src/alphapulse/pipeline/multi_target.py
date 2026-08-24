from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

from ..evaluation.metrics import era_sharpe, rank_normalize
from ..experiments.data import require_meta_model_from_benchmarks
from ..models.base import BaseModel
from ..preprocessors.base import BasePreprocessor, TrainEvalPreprocessor
from ..preprocessors.era_stable import EraStableFeatureSelector
from .neutralizer import (
    FeatureNeutralizer,
    MetaModelNeutralizer,
    apply_prediction_neutralization,
)
from .row_utils import (
    blend_with_benchmark,
    filter_invalid_rows,
    filter_nan_rows,
    protected_metadata_frame,
    reattach_protected_columns,
    select_required_features,
)

_MIN_TRAIN_ROWS = 10
_MIN_VAL_ROWS = 2


def _align_targets(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    if frame.index.has_duplicates:
        raise ValueError(f"{name} feature index must not contain duplicates")
    if targets.index.has_duplicates:
        raise ValueError(f"{name} target index must not contain duplicates")
    missing = frame.index.difference(targets.index)
    extra = targets.index.difference(frame.index)
    if len(missing) > 0 or len(extra) > 0:
        raise ValueError(f"{name} feature and target row IDs must exactly match")
    return targets.reindex(frame.index)


class MultiTargetPipeline:
    def __init__(
        self,
        preprocessors: list[BasePreprocessor],
        model_factory: Callable[[], BaseModel],
        target_columns: list[str],
        primary_target: str = "target",
        blend_method: str = "equal",
        benchmark_blend_weight: float = 0.0,
        neutralize_proportion: float = 0.0,
        neutralize_features: list[str] | None = None,
        meta_neutralize_proportion: float = 0.0,
    ) -> None:
        if not target_columns:
            raise ValueError("target_columns must be non-empty")
        if primary_target not in target_columns:
            target_columns = [primary_target] + list(target_columns)
        self.preprocessors = preprocessors
        self.model_factory = model_factory
        self.target_columns = target_columns
        self.primary_target = primary_target
        self.blend_method = blend_method
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

        self._models: dict[str, BaseModel] = {}
        self._weights: np.ndarray | None = None
        self.feature_columns: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        targets: pd.DataFrame,
        *,
        X_val: pd.DataFrame | None = None,
        targets_val: pd.DataFrame | None = None,
        era_train: pd.Series | None = None,
        era_val: pd.Series | None = None,
        **model_train_kwargs: Any,
    ) -> dict[str, float]:
        self._models = {}
        self._weights = None
        self.feature_columns = list(X.columns)
        targets = _align_targets(X, targets, name="training")
        if X_val is None and targets_val is not None:
            raise ValueError("targets_val requires X_val")
        if X_val is not None and targets_val is not None:
            targets_val = _align_targets(X_val, targets_val, name="validation")
        if era_train is not None and "era" not in X.columns:
            X = X.copy()
            X["era"] = era_train.reindex(X.index)
        if X_val is not None and era_val is not None and "era" not in X_val.columns:
            X_val = X_val.copy()
            X_val["era"] = era_val.reindex(X_val.index)

        X, _ = filter_invalid_rows(X)
        targets = targets.loc[X.index]
        if era_train is not None:
            era_train = era_train.loc[X.index]

        y_primary = (
            targets[self.primary_target]
            if self.primary_target in targets.columns
            else None
        )

        train_metadata = protected_metadata_frame(X)
        X_fit = X
        for pp in self.preprocessors:
            if isinstance(pp, TrainEvalPreprocessor):
                pp.train()
            if isinstance(pp, EraStableFeatureSelector) and era_train is not None:
                pp.fit(X_fit, y_primary, eras=era_train)
            else:
                pp.fit(X_fit, y_primary)
            X_fit = pp.transform(X_fit)
            X_fit = reattach_protected_columns(X_fit, train_metadata)

        X_fit, _ = filter_nan_rows(X_fit)
        targets = targets.loc[X_fit.index]
        if era_train is not None:
            era_train = era_train.loc[X_fit.index]

        X_val_t: pd.DataFrame | None = None
        if X_val is not None:
            if era_val is None and "era" in X_val.columns:
                era_val = X_val["era"]
            X_val, _ = filter_invalid_rows(X_val)
            if targets_val is not None:
                targets_val = targets_val.loc[X_val.index]
            if era_val is not None:
                era_val = era_val.loc[X_val.index]
            validation_metadata = protected_metadata_frame(X_val)
            X_val_t = X_val
            for pp in self.preprocessors:
                if isinstance(pp, TrainEvalPreprocessor):
                    pp.eval()
                X_val_t = pp.transform(X_val_t)
                X_val_t = reattach_protected_columns(X_val_t, validation_metadata)
            X_val_t, _ = filter_nan_rows(X_val_t)
            if targets_val is not None:
                targets_val = targets_val.loc[X_val_t.index]
            if era_val is not None:
                era_val = era_val.loc[X_val_t.index]

        all_metrics: dict[str, float] = {}
        available_targets = [t for t in self.target_columns if t in targets.columns]
        if not available_targets:
            raise ValueError(
                f"None of {self.target_columns} found in targets DataFrame"
            )

        for target_col in available_targets:
            y = targets[target_col]
            valid_mask = y.notna()
            if valid_mask.sum() < _MIN_TRAIN_ROWS:
                continue

            y_val = (
                targets_val[target_col]
                if targets_val is not None and target_col in targets_val.columns
                else None
            )

            X_fit_masked = X_fit[valid_mask]
            y_masked = y[valid_mask]

            X_val_masked: Any = X_val_t
            y_val_masked: Any = y_val
            if y_val is not None:
                val_valid = y_val.notna()
                if val_valid.sum() >= _MIN_VAL_ROWS:
                    valid_index = y_val.index[val_valid]
                    X_val_masked = (
                        X_val_t.loc[valid_index] if X_val_t is not None else None
                    )
                    y_val_masked = y_val.loc[valid_index]
                else:
                    X_val_masked = None
                    y_val_masked = None

            model = self.model_factory()
            train_kw = dict(model_train_kwargs)
            if era_train is not None:
                train_kw["era_train"] = era_train.loc[X_fit_masked.index]
            if era_val is not None and X_val_masked is not None:
                train_kw["era_val"] = era_val.loc[X_val_masked.index]
            metrics = model.train(
                X_fit_masked,
                y_masked,
                X_val=X_val_masked,
                y_val=y_val_masked,
                **train_kw,
            )
            self._models[target_col] = model
            for k, v in metrics.items():
                all_metrics[f"{target_col}_{k}"] = v

        fitted_targets = [t for t in available_targets if t in self._models]
        if not fitted_targets:
            raise ValueError(
                "No target has enough valid training rows to fit a model; "
                f"at least {_MIN_TRAIN_ROWS} rows are required"
            )
        self._compute_weights(
            X_fit, targets, era_train, fitted_targets, X_val_t, targets_val, era_val
        )
        return all_metrics

    def _compute_weights(
        self,
        X_fit: pd.DataFrame,
        targets: pd.DataFrame,
        era_train: pd.Series | None,
        fitted_targets: list[str],
        X_val_t: pd.DataFrame | None = None,
        targets_val: pd.DataFrame | None = None,
        era_val: pd.Series | None = None,
    ) -> None:
        n = len(fitted_targets)
        if n <= 1 or self.blend_method == "equal":
            self._weights = np.ones(n) / n
            return

        if (
            self.blend_method == "sharpe"
            and X_val_t is not None
            and targets_val is not None
            and era_val is not None
            and self.primary_target in targets_val.columns
        ):
            y_prim = targets_val[self.primary_target]
            valid = y_prim.notna()
            y_prim = y_prim[valid]
            X_eval: pd.DataFrame = X_val_t.loc[y_prim.index]
            era_eval: pd.Series = era_val.loc[y_prim.index]
        elif (
            self.blend_method == "sharpe"
            and era_train is not None
            and self.primary_target in targets.columns
        ):
            y_prim = targets[self.primary_target]
            X_eval = X_fit
            era_eval = era_train
        else:
            self._weights = np.ones(n) / n
            return

        sharpes = []
        for t in fitted_targets:
            pred = self._models[t].predict(X_eval)
            s = era_sharpe(y_prim, pred, era_eval)
            sharpes.append(max(s, 0.0) if np.isfinite(s) else 0.0)

        total = sum(sharpes)
        if total > 0:
            self._weights = np.array(sharpes, dtype=np.float64) / total
        else:
            self._weights = np.ones(n) / n

    @property
    def ensemble_weights(self) -> list[float] | None:
        if self._weights is None:
            return None
        return [float(weight) for weight in self._weights]

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        X_t = X
        for pp in self.preprocessors:
            if isinstance(pp, TrainEvalPreprocessor):
                pp.eval()
            X_t = pp.transform(X_t)

        available = [t for t in self.target_columns if t in self._models]
        if not available:
            raise ValueError("No fitted models found")

        if len(available) == 1:
            return self._models[available[0]].predict(X_t)

        preds = np.column_stack([self._models[t].predict(X_t) for t in available])
        w = (
            self._weights
            if self._weights is not None
            else np.ones(len(available)) / len(available)
        )
        return np.asarray(preds @ w, dtype=np.float64)

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
