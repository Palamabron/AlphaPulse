from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseModel, _numeric


def _require_xgboost() -> Any:
    try:
        import xgboost
    except ImportError as exc:
        raise ImportError("XGBoostModel requires xgboost.") from exc
    return xgboost


def _make_progress_callbacks(model_name: str, log_every: int = 10) -> list[Any]:
    xgb = _require_xgboost()
    ray_callbacks = _make_ray_callbacks()
    if ray_callbacks:
        return ray_callbacks

    from ..logging_.wandb_logging import (
        log_boosting_round_metrics,
        parse_xgb_evals_log,
        wandb_run_active,
    )

    class _LogProgressCallback(xgb.callback.TrainingCallback):  # type: ignore[name-defined]
        def after_iteration(
            self,
            model: Any,
            epoch: int,
            evals_log: dict[str, dict[str, list[float] | list[tuple[float, float]]]],
        ) -> bool:
            if epoch != 0 and (epoch + 1) % log_every != 0:
                return False
            parsed = parse_xgb_evals_log(evals_log)
            if parsed:
                parts = [f"{k}={v:.6f}" for k, v in parsed.items()]
                logger.info("{} round {}: {}", model_name, epoch + 1, " ".join(parts))
                if wandb_run_active():
                    log_boosting_round_metrics(
                        model_name=model_name,
                        round_num=epoch + 1,
                        metrics=parsed,
                    )
            return False

    return [_LogProgressCallback()]


def _make_ray_callbacks() -> list[Any]:
    try:
        xgb = _require_xgboost()
        from ray import tune as ray_tune

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trial_id = ray_tune.get_context().get_trial_id()
        if trial_id is None:
            return []

        class _RayReportCallback(xgb.callback.TrainingCallback):  # type: ignore[name-defined]
            def after_iteration(
                self,
                model: Any,
                _epoch: int,
                evals_log: dict[
                    str, dict[str, list[float] | list[tuple[float, float]]]
                ],
            ) -> bool:
                eval_rmse: float | None = None
                for dataset, metrics in (evals_log or {}).items():
                    if dataset == "eval" and "rmse" in metrics:
                        values = metrics["rmse"]
                        if values:
                            last = values[-1]
                            eval_rmse = float(
                                last[0] if isinstance(last, tuple) else last
                            )
                            break
                if eval_rmse is None:
                    for item in getattr(model, "evaluation_result_list", []) or []:
                        if not isinstance(item, tuple) or len(item) < 3:
                            continue
                        dataset, metric_name, value = item[0], item[1], item[2]
                        if dataset == "eval" and metric_name == "rmse":
                            eval_rmse = float(value)
                            break
                if eval_rmse is not None:
                    ray_tune.report({"eval_rmse": eval_rmse})
                return False

        return [_RayReportCallback()]
    except ImportError:
        return []


class XGBoostModel(BaseModel):
    def __init__(
        self, params: dict[str, Any] | None = None, name: str | None = "XGBoost"
    ) -> None:
        super().__init__(name)
        self.params: dict[str, Any] = params or {
            "max_depth": 5,
            "learning_rate": 0.01,
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        n_rounds: int = 1500,
        early_stopping_rounds: int = 100,
        **kwargs: Any,
    ) -> dict[str, float]:
        xgb = _require_xgboost()
        feat_train = _numeric(X_train)
        if feat_train.shape[1] == 0:
            raise ValueError(f"{self.name}: no numeric feature columns found.")

        dtrain = xgb.DMatrix(feat_train, label=y_train)

        eval_set: list[tuple[Any, str]] = []
        if X_val is not None and y_val is not None:
            feat_val = _numeric(X_val)
            dval = xgb.DMatrix(feat_val, label=y_val)
            eval_set = [(dtrain, "train"), (dval, "eval")]

        evals_result: dict[str, Any] = {}
        callbacks: list[Any] = _make_progress_callbacks(self.name)

        logger.info(
            "{}: starting XGBoost train rows={} features={} rounds={} early_stop={}",
            self.name,
            len(feat_train),
            feat_train.shape[1],
            n_rounds,
            early_stopping_rounds if eval_set else None,
        )

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=n_rounds,
            evals=eval_set,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            verbose_eval=False,
            callbacks=callbacks,
        )
        self.is_trained = True
        best_iter = getattr(self.model, "best_iteration", n_rounds - 1)
        logger.info("{}: finished at iteration {}", self.name, best_iter + 1)

        metrics: dict[str, float] = {}
        for key, values in evals_result.items():
            first_metric = list(values.keys())[0]
            metrics[f"{key}_final"] = float(values[first_metric][-1])
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        xgb = _require_xgboost()
        self._require_trained()
        feat = _numeric(X)
        dtest = xgb.DMatrix(feat)
        return np.asarray(self.model.predict(dtest), dtype=np.float64)

    def save(self, path: Path) -> None:
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: Path) -> XGBoostModel:
        xgb = _require_xgboost()
        booster = xgb.Booster()
        booster.load_model(str(path))
        self.model = booster
        self.is_trained = True
        return self
