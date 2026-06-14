from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.metrics import era_sharpe, rank_normalize
from ..models.base import BaseModel
from ..preprocessors.base import BasePreprocessor
from .row_utils import filter_invalid_rows, filter_nan_rows

_MIN_TRAIN_ROWS = 10
_MIN_VAL_ROWS = 2


class MultiTargetPipeline:
    def __init__(
        self,
        preprocessors: list[BasePreprocessor],
        model_factory: Callable[[], BaseModel],
        target_columns: list[str],
        primary_target: str = "target",
        blend_method: str = "equal",
        benchmark_blend_weight: float = 0.0,
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
        **model_train_kwargs: Any,
    ) -> dict[str, float]:
        self.feature_columns = list(X.columns)

        X, _ = filter_invalid_rows(X)
        targets = targets.loc[X.index]
        if era_train is not None:
            era_train = era_train.loc[X.index]

        y_primary = (
            targets[self.primary_target]
            if self.primary_target in targets.columns
            else None
        )

        X_fit = X
        for pp in self.preprocessors:
            pp.fit(X_fit, y_primary)
            X_fit = pp.transform(X_fit)

        X_fit, _ = filter_nan_rows(X_fit)
        targets = targets.loc[X_fit.index]
        if era_train is not None:
            era_train = era_train.loc[X_fit.index]

        era_val: pd.Series | None = None
        X_val_t: pd.DataFrame | None = None
        if X_val is not None:
            era_val = X_val["era"] if "era" in X_val.columns else None
            X_val_t = X_val
            for pp in self.preprocessors:
                X_val_t = pp.transform(X_val_t)

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
                    X_val_masked = X_val_t[val_valid] if X_val_t is not None else None
                    y_val_masked = y_val[val_valid]
                else:
                    X_val_masked = None
                    y_val_masked = None

            model = self.model_factory()
            metrics = model.train(
                X_fit_masked,
                y_masked,
                X_val=X_val_masked,
                y_val=y_val_masked,
                **model_train_kwargs,
            )
            self._models[target_col] = model
            for k, v in metrics.items():
                all_metrics[f"{target_col}_{k}"] = v

        fitted_targets = [t for t in available_targets if t in self._models]
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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_t = X
        for pp in self.preprocessors:
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
