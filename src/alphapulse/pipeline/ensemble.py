from typing import Any, Literal

import numpy as np
import pandas as pd


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

    def fit(
        self,
        n_models: int,
        get_val_predictions: Any = None,
        y_val: pd.Series | None = None,
    ) -> None:
        if n_models <= 1:
            return

        if self.method == "weighted":
            weights = self.params.get("weights")
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

        if self.method == "stacking" and self._meta_learner is not None:
            if self._meta_learner_type == "xgboost":
                import xgboost as xgb

                return np.asarray(
                    self._meta_learner.predict(xgb.DMatrix(pred_matrix)),
                    dtype=np.float64,
                )
            return np.asarray(self._meta_learner.predict(pred_matrix), dtype=np.float64)

        return np.asarray(pred_matrix.mean(axis=1), dtype=np.float64)
