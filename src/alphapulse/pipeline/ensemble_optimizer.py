from typing import Literal, Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..evaluation.metrics import era_sharpe, payout_score

MAX_RESTARTS = 3


class EnsembleOptimizer:
    def __init__(
        self,
        method: str = "SLSQP",
        max_iter: int = 500,
        seed: int = 42,
        objective: Literal["corr_sharpe", "payout_score"] = "corr_sharpe",
        corr_weight: float = 0.75,
        mmc_weight: float = 2.25,
    ) -> None:
        self.method = method
        self.max_iter = max_iter
        self.seed = seed
        self.objective = objective
        self.corr_weight = corr_weight
        self.mmc_weight = mmc_weight
        self.weights_: np.ndarray | None = None
        self.sharpe_: float = float("-inf")

    def fit(
        self,
        oof_matrix: np.ndarray,
        y_oof: np.ndarray,
        eras_oof: pd.Series,
        *,
        meta_model_preds: np.ndarray | None = None,
    ) -> Self:
        """Optimise ensemble weights on OOF predictions.

        Args:
            oof_matrix: Shape (n_samples, n_models) OOF prediction matrix.
            y_oof: True target values aligned with oof_matrix.
            eras_oof: Era labels aligned with oof_matrix.
            meta_model_preds: Optional Numerai meta model predictions. Required
                when ``objective="payout_score"``; falls back to corr_sharpe if None.
        """
        k = oof_matrix.shape[1]
        if k < 1:
            raise ValueError("Need at least 1 column in oof_matrix")

        y_series = pd.Series(y_oof)
        use_payout = self.objective == "payout_score" and meta_model_preds is not None
        meta_arr = (
            np.asarray(meta_model_preds, dtype=np.float64) if use_payout else None
        )

        if k == 1:
            self.weights_ = np.array([1.0])
            self.sharpe_ = era_sharpe(y_series, oof_matrix[:, 0], eras_oof)
            return self

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
        bounds = [(0.0, 1.0)] * k

        rng = np.random.RandomState(self.seed)
        best_result = None
        best_val = float("inf")

        starts = [
            np.ones(k) / k,
            rng.dirichlet(np.ones(k)),
            rng.dirichlet(np.ones(k) * 0.1),
        ]

        for w0 in starts:
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
        self.weights_ = np.clip(np.asarray(best_result.x, dtype=np.float64), 0.0, None)
        self.weights_ /= self.weights_.sum()
        self.sharpe_ = -best_val
        return self

    def predict(self, pred_matrix: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Call fit() first.")
        return np.asarray(pred_matrix @ self.weights_, dtype=np.float64)


class GreedyEnsembleSelector:
    """Greedily select the best subset of models from OOF predictions.

    Starting from an empty ensemble, iteratively adds the model whose OOF
    predictions most improve the objective (equal-weight average) when included.
    Continues until no improvement or ``max_models`` is reached.

    This is faster than exhaustive search and produces sparse, high-quality
    ensembles. Typical usage: run after collecting OOF predictions from many
    HPO trials to find the best k of n configs.

    Args:
        max_models: Maximum number of models to include.
        metric: Objective to maximise. ``"payout_score"`` requires meta model.
        corr_weight: CORR weight in payout formula. Default 0.75.
        mmc_weight: MMC weight in payout formula. Default 2.25.
    """

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
        """Select the best subset of models from OOF predictions.

        Args:
            oof_matrix: Shape (n_samples, n_models) OOF prediction matrix.
            y_oof: True target values.
            eras_oof: Era labels.
            meta_model_preds: Optional Numerai meta model predictions.
                Required when ``metric="payout_score"``.
            model_names: Optional display names for each column (for inspection).
        """
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
                break  # no more improvement

            selected.append(best_i)
            n_sel = len(selected)
            current_blend = (
                current_blend * (n_sel - 1) + oof_matrix[:, best_i]
            ) / n_sel
            best_score = best_step_score

        self.selected_indices_ = selected
        self.best_score_ = best_score
        return self

    @property
    def selected_names(self) -> list[int]:
        return self.selected_indices_

    def predict(self, pred_matrix: np.ndarray) -> np.ndarray:
        if not self.selected_indices_:
            raise RuntimeError("Call fit() first or no models were selected.")
        subset = pred_matrix[:, self.selected_indices_]
        return np.asarray(subset.mean(axis=1), dtype=np.float64)
