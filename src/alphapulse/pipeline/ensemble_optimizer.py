from typing import Literal, Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..evaluation.metrics import era_sharpe, payout_score

DEFAULT_MIN_WEIGHT = 0.05
DEFAULT_MAX_WEIGHT = 0.90


def validate_weight_bounds_list(
    min_weights: list[float], max_weights: list[float]
) -> None:
    k = len(min_weights)
    if k != len(max_weights):
        raise ValueError("min_weights and max_weights must have the same length")
    if k < 1:
        raise ValueError("k must be >= 1")
    for i, (lo, hi) in enumerate(zip(min_weights, max_weights, strict=True)):
        if lo < 0.0 or hi > 1.0 or lo > hi:
            raise ValueError(f"invalid weight bounds at index {i}: min={lo}, max={hi}")
    if sum(min_weights) > 1.0 + 1e-9:
        raise ValueError(f"infeasible: sum(min_weights) > 1 ({sum(min_weights)})")
    if sum(max_weights) < 1.0 - 1e-9:
        raise ValueError(f"infeasible: sum(max_weights) < 1 ({sum(max_weights)})")


def validate_weight_bounds(k: int, min_weight: float, max_weight: float) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if min_weight < 0.0 or max_weight > 1.0 or min_weight > max_weight:
        raise ValueError(
            f"invalid weight bounds: min_weight={min_weight}, max_weight={max_weight}"
        )
    if k * min_weight > 1.0 + 1e-9:
        raise ValueError(f"infeasible: k * min_weight > 1 ({k} * {min_weight})")
    if k * max_weight < 1.0 - 1e-9:
        raise ValueError(f"infeasible: k * max_weight < 1 ({k} * {max_weight})")


def project_weights_to_bounds_list(
    weights: np.ndarray,
    min_weights: list[float],
    max_weights: list[float],
) -> np.ndarray:
    k = len(weights)
    lo = [float(v) for v in min_weights]
    hi = [float(v) for v in max_weights]
    validate_weight_bounds_list(lo, hi)
    w = np.clip(np.asarray(weights, dtype=np.float64), lo, hi)
    for _ in range(64):
        total = float(w.sum())
        if abs(total - 1.0) < 1e-10:
            break
        slack_hi = np.asarray(hi, dtype=np.float64) - w
        slack_lo = w - np.asarray(lo, dtype=np.float64)
        delta = 1.0 - total
        if delta > 0:
            room = slack_hi.sum()
            if room > 1e-12:
                w += delta * (slack_hi / room)
        else:
            room = slack_lo.sum()
            if room > 1e-12:
                w += delta * (slack_lo / room)
        w = np.clip(w, lo, hi)
    if abs(float(w.sum()) - 1.0) > 1e-6:
        w = project_weights_to_bounds(np.ones(k) / k, lo[0], hi[0])
        w = project_weights_to_bounds_list(w, lo, hi)
    return w


def feasible_weight_starts_list(
    min_weights: list[float],
    max_weights: list[float],
    rng: np.random.RandomState,
) -> list[np.ndarray]:
    lo = [float(v) for v in min_weights]
    hi = [float(v) for v in max_weights]
    validate_weight_bounds_list(lo, hi)
    k = len(lo)
    starts = [
        project_weights_to_bounds_list(np.ones(k) / k, lo, hi),
    ]
    for _ in range(2):
        raw = rng.dirichlet(np.ones(k))
        starts.append(project_weights_to_bounds_list(raw, lo, hi))
    return starts


def project_weights_to_bounds(
    weights: np.ndarray,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    k = len(weights)
    validate_weight_bounds(k, min_weight, max_weight)
    w = np.clip(np.asarray(weights, dtype=np.float64), min_weight, max_weight)
    for _ in range(64):
        total = float(w.sum())
        if abs(total - 1.0) < 1e-10:
            break
        delta = (1.0 - total) / k
        w = np.clip(w + delta, min_weight, max_weight)
    if abs(float(w.sum()) - 1.0) > 1e-6:
        w = np.ones(k, dtype=np.float64) / k
        w = np.clip(w, min_weight, max_weight)
        w = project_weights_to_bounds(w, min_weight, max_weight)
    return w


def feasible_weight_starts(
    k: int,
    min_weight: float,
    max_weight: float,
    rng: np.random.RandomState,
) -> list[np.ndarray]:
    validate_weight_bounds(k, min_weight, max_weight)
    starts = [project_weights_to_bounds(np.ones(k) / k, min_weight, max_weight)]
    for _ in range(2):
        raw = rng.dirichlet(np.ones(k))
        starts.append(project_weights_to_bounds(raw, min_weight, max_weight))
    return starts


class EnsembleOptimizer:
    def __init__(
        self,
        method: str = "SLSQP",
        max_iter: int = 500,
        seed: int = 42,
        objective: Literal["corr_sharpe", "payout_score"] = "corr_sharpe",
        corr_weight: float = 0.75,
        mmc_weight: float = 2.25,
        min_weight: float = DEFAULT_MIN_WEIGHT,
        max_weight: float = DEFAULT_MAX_WEIGHT,
    ) -> None:
        self.method = method
        self.max_iter = max_iter
        self.seed = seed
        self.objective = objective
        self.corr_weight = corr_weight
        self.mmc_weight = mmc_weight
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.weights_: np.ndarray | None = None
        self.sharpe_: float = float("-inf")

    def fit(
        self,
        oof_matrix: np.ndarray,
        y_oof: np.ndarray,
        eras_oof: pd.Series,
        *,
        meta_model_preds: np.ndarray | None = None,
        min_weights: list[float] | None = None,
        max_weights: list[float] | None = None,
    ) -> Self:
        """Optimise ensemble weights on OOF predictions.

        Args:
            oof_matrix: Shape (n_samples, n_models) OOF prediction matrix.
            y_oof: True target values aligned with oof_matrix.
            eras_oof: Era labels aligned with oof_matrix.
            meta_model_preds: Optional Numerai meta model predictions. Required
                when ``objective="payout_score"``.
        """
        k = oof_matrix.shape[1]
        if k < 1:
            raise ValueError("Need at least 1 column in oof_matrix")
        if self.objective == "payout_score" and meta_model_preds is None:
            raise ValueError(
                "payout_score weight optimization requires aligned meta-model "
                "predictions"
            )

        y_series = pd.Series(y_oof)
        use_payout = self.objective == "payout_score" and meta_model_preds is not None
        meta_arr = (
            np.asarray(meta_model_preds, dtype=np.float64) if use_payout else None
        )

        if k == 1:
            self.weights_ = np.array([1.0])
            self.sharpe_ = era_sharpe(y_series, oof_matrix[:, 0], eras_oof)
            return self

        lo = (
            [float(v) for v in min_weights]
            if min_weights is not None
            else [self.min_weight] * k
        )
        hi = (
            [float(v) for v in max_weights]
            if max_weights is not None
            else [self.max_weight] * k
        )
        validate_weight_bounds_list(lo, hi)
        bounds = list(zip(lo, hi, strict=True))

        def neg_objective(w: np.ndarray) -> float:
            blend = oof_matrix @ w
            if use_payout and meta_arr is not None:
                score = payout_score(
                    y_series,
                    blend,
                    meta_arr,
                    eras_oof,
                    corr_weight=self.corr_weight,
                    mmc_weight=self.mmc_weight,
                )
            else:
                score = era_sharpe(y_series, blend, eras_oof)
            return -score if np.isfinite(score) else 1e6

        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

        rng = np.random.RandomState(self.seed)
        best_result = None
        best_val = float("inf")

        for w0 in feasible_weight_starts_list(lo, hi, rng):
            res = minimize(
                neg_objective,
                w0,
                method=self.method,
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": self.max_iter, "ftol": 1e-9},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res

        assert best_result is not None
        self.weights_ = project_weights_to_bounds_list(
            np.asarray(best_result.x, dtype=np.float64),
            lo,
            hi,
        )
        self.sharpe_ = -best_val
        return self

    def predict(self, pred_matrix: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Call fit() first.")
        return np.asarray(pred_matrix @ self.weights_, dtype=np.float64)


class GreedyEnsembleSelector:
    def __init__(
        self,
        max_models: int = 20,
        metric: Literal["corr_sharpe", "payout_score"] = "payout_score",
        corr_weight: float = 0.75,
        mmc_weight: float = 2.25,
    ) -> None:
        self.max_models = max_models
        self.metric = metric
        self.corr_weight = corr_weight
        self.mmc_weight = mmc_weight
        self.selected_indices_: list[int] = []
        self.best_score_: float = float("-inf")
        self.model_names_: list[str] | None = None

    def _score(
        self,
        blend: np.ndarray,
        y: pd.Series,
        eras: pd.Series,
        meta_arr: np.ndarray | None,
    ) -> float:
        if self.metric == "payout_score" and meta_arr is not None:
            return payout_score(
                y, blend, meta_arr, eras, self.corr_weight, self.mmc_weight
            )
        return era_sharpe(y, blend, eras)

    def fit(
        self,
        oof_matrix: np.ndarray,
        y_oof: np.ndarray,
        eras_oof: pd.Series,
        *,
        meta_model_preds: np.ndarray | None = None,
        model_names: list[str] | None = None,
    ) -> Self:
        n_models = oof_matrix.shape[1]
        y_series = pd.Series(y_oof)
        meta_arr = (
            np.asarray(meta_model_preds, dtype=np.float64)
            if meta_model_preds is not None
            else None
        )

        selected: list[int] = []
        current_blend = np.zeros(len(y_oof), dtype=np.float64)
        best_score = float("-inf")

        for _ in range(min(self.max_models, n_models)):
            best_i = -1
            best_step_score = float("-inf")
            for i in range(n_models):
                if i in selected:
                    continue
                n_sel = len(selected)
                candidate = (current_blend * n_sel + oof_matrix[:, i]) / (n_sel + 1)
                s = self._score(candidate, y_series, eras_oof, meta_arr)
                if s > best_step_score:
                    best_step_score = s
                    best_i = i

            if best_i == -1 or best_step_score <= best_score:
                break

            selected.append(best_i)
            n_sel = len(selected)
            current_blend = (
                current_blend * (n_sel - 1) + oof_matrix[:, best_i]
            ) / n_sel
            best_score = best_step_score

        self.selected_indices_ = selected
        self.best_score_ = best_score
        self.model_names_ = model_names
        return self

    @property
    def selected_names(self) -> list[str] | list[int]:
        if self.model_names_ is not None:
            return [self.model_names_[i] for i in self.selected_indices_]
        return self.selected_indices_

    def predict(self, pred_matrix: np.ndarray) -> np.ndarray:
        if not self.selected_indices_:
            raise RuntimeError("Call fit() first or no models were selected.")
        subset = pred_matrix[:, self.selected_indices_]
        return np.asarray(subset.mean(axis=1), dtype=np.float64)
