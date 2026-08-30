from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd

from .ensemble_optimizer import (
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MIN_WEIGHT,
    EnsembleOptimizer,
)


def needs_internal_val_for_ensemble(pipeline_cfg: dict[str, Any]) -> bool:
    models = pipeline_cfg.get("models") or []
    if len(models) <= 1:
        return False
    method = pipeline_cfg.get("ensemble_method", "single")
    if method == "stacking":
        return True
    if method == "weighted":
        params = pipeline_cfg.get("ensemble_params") or {}
        return bool(params.get("optimize_weights"))
    return False


class EnsembleStrategy:
    def __init__(
        self,
        method: Literal["single", "weighted", "stacking"] = "single",
        params: dict[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.params = params or {}
        self._weights: np.ndarray | None = None
        self._meta_learner: Any = None
        self._meta_learner_type: str | None = None

    @property
    def weights(self) -> np.ndarray | None:
        return None if self._weights is None else self._weights.copy()

    def set_weights(self, weights: np.ndarray | list[float]) -> None:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1:
            raise ValueError("weights must be a 1-D array")
        total = float(w.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError(
                f"weights must sum to a positive finite value, got {total}"
            )
        self._weights = w / total

    def fit(
        self,
        n_models: int,
        get_val_predictions: Callable[[], np.ndarray] | None = None,
        y_val: pd.Series | None = None,
        eras_val: pd.Series | None = None,
        meta_model_preds: np.ndarray | None = None,
    ) -> None:
        if n_models <= 1:
            return

        if self.method == "weighted":
            weights = self.params.get("weights")
            optimize_weights = bool(self.params.get("optimize_weights"))
            if optimize_weights and weights is not None:
                raise ValueError(
                    "ensemble_params cannot set both optimize_weights and fixed weights"
                )
            if (
                optimize_weights
                and get_val_predictions is not None
                and y_val is not None
            ):
                stack_X = get_val_predictions()
                if stack_X.shape[1] != n_models:
                    raise ValueError(
                        f"validation predictions have {stack_X.shape[1]} columns "
                        f"but expected {n_models}"
                    )
                finite_mask = np.isfinite(stack_X).all(axis=1) & np.isfinite(
                    np.asarray(y_val, dtype=np.float64)
                )
                if eras_val is not None:
                    finite_mask &= eras_val.notna().to_numpy()
                if not finite_mask.all():
                    stack_X = stack_X[finite_mask]
                    y_val_arr = np.asarray(y_val, dtype=np.float64)[finite_mask]
                    eras_fit = (
                        eras_val.iloc[finite_mask]
                        if eras_val is not None
                        else pd.Series(np.zeros(len(y_val_arr)))
                    )
                    meta_fit = (
                        np.asarray(meta_model_preds, dtype=np.float64)[finite_mask]
                        if meta_model_preds is not None
                        else None
                    )
                else:
                    y_val_arr = np.asarray(y_val, dtype=np.float64)
                    eras_fit = (
                        eras_val
                        if eras_val is not None
                        else pd.Series(np.zeros(len(y_val_arr)))
                    )
                    meta_fit = meta_model_preds

                objective = self.params.get("objective", "corr_sharpe")
                if objective not in ("corr_sharpe", "payout_score"):
                    objective = "corr_sharpe"
                optimizer = EnsembleOptimizer(
                    objective=objective,
                    corr_weight=float(self.params.get("corr_weight", 0.75)),
                    mmc_weight=float(self.params.get("mmc_weight", 2.25)),
                    min_weight=float(self.params.get("min_weight", DEFAULT_MIN_WEIGHT)),
                    max_weight=float(self.params.get("max_weight", DEFAULT_MAX_WEIGHT)),
                    seed=int(self.params.get("seed", 42)),
                )
                min_w = self.params.get("min_weights")
                max_w = self.params.get("max_weights")
                optimizer.fit(
                    stack_X,
                    y_val_arr,
                    eras_fit,
                    meta_model_preds=meta_fit,
                    min_weights=list(min_w) if min_w is not None else None,
                    max_weights=list(max_w) if max_w is not None else None,
                )
                self._weights = optimizer.weights_
                return

            if weights is not None:
                w = np.asarray(weights, dtype=np.float64)
                if len(w) != n_models:
                    raise ValueError(
                        f"weights length ({len(w)}) does not match "
                        f"n_models ({n_models})"
                    )
                w_sum = float(w.sum())
                if w_sum <= 0.0 or not np.isfinite(w_sum):
                    raise ValueError(
                        f"weights must sum to a positive finite value, got sum={w_sum}"
                    )
                self._weights = w / w_sum
            else:
                self._weights = np.ones(n_models) / n_models

        elif self.method == "stacking":
            if get_val_predictions is None or y_val is None:
                raise ValueError(
                    "Stacking requires validation predictions and targets. "
                    "Provide X_val and y_val when fitting a pipeline with "
                    "ensemble_method='stacking'."
                )

            stack_X = get_val_predictions()
            if stack_X.shape[1] != n_models:
                raise ValueError(
                    f"Stacking validation predictions have {stack_X.shape[1]} "
                    f"columns but expected {n_models} (one per model)"
                )

            finite_mask = np.isfinite(stack_X).all(axis=1)
            if not finite_mask.all():
                stack_X = stack_X[finite_mask]
                y_val = np.asarray(y_val, dtype=np.float64)[finite_mask]

            meta = self.params.get("meta_learner", "ridge")
            meta_params = self.params.get("meta_params") or {}

            if meta == "ridge":
                from sklearn.linear_model import Ridge

                self._meta_learner = Ridge(**meta_params).fit(stack_X, y_val)
                self._meta_learner_type = "ridge"
            elif meta == "xgboost":
                import xgboost as xgb

                dtrain = xgb.DMatrix(stack_X, label=y_val)
                self._meta_learner = xgb.train(
                    {
                        "max_depth": meta_params.get("max_depth", 3),
                        "objective": "reg:squarederror",
                        **meta_params,
                    },
                    dtrain,
                    num_boost_round=meta_params.get("num_boost_round", 50),
                    verbose_eval=False,
                )
                self._meta_learner_type = "xgboost"
            else:
                raise ValueError(f"Unknown meta_learner: {meta}")

    def combine(self, pred_matrix: np.ndarray) -> np.ndarray:
        n_cols = pred_matrix.shape[1] if pred_matrix.ndim == 2 else 1
        if n_cols == 1:
            return pred_matrix.ravel()

        if self.method == "weighted":
            w = self._weights if self._weights is not None else np.ones(n_cols) / n_cols
            return pred_matrix @ w

        if self.method == "stacking":
            if self._meta_learner is None:
                raise RuntimeError(
                    "Stacking meta-learner is not fitted. "
                    "Call fit() with validation predictions and targets first."
                )
            if self._meta_learner_type == "xgboost":
                import xgboost as xgb

                return np.asarray(
                    self._meta_learner.predict(xgb.DMatrix(pred_matrix)),
                    dtype=np.float64,
                )
            return np.asarray(self._meta_learner.predict(pred_matrix), dtype=np.float64)

        return np.asarray(pred_matrix.mean(axis=1), dtype=np.float64)
