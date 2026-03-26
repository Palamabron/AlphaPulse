from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..evaluation.metrics import era_sharpe

MAX_RESTARTS = 3


class EnsembleOptimizer:
    def __init__(
        self, method: str = "SLSQP", max_iter: int = 500, seed: int = 42
    ) -> None:
        self.method = method
        self.max_iter = max_iter
        self.seed = seed
        self.weights_: np.ndarray | None = None
        self.sharpe_: float = float("-inf")

    def fit(
        self, oof_matrix: np.ndarray, y_oof: np.ndarray, eras_oof: pd.Series
    ) -> Self:
        k = oof_matrix.shape[1]
        if k < 1:
            raise ValueError("Need at least 1 column in oof_matrix")

        if k == 1:
            self.weights_ = np.array([1.0])
            self.sharpe_ = era_sharpe(pd.Series(y_oof), oof_matrix[:, 0], eras_oof)
            return self

        y_series = pd.Series(y_oof)

        def neg_sharpe(w: np.ndarray) -> float:
            blend = oof_matrix @ w
            s = era_sharpe(y_series, blend, eras_oof)
            return -s if np.isfinite(s) else 1e6

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
                neg_sharpe,
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
