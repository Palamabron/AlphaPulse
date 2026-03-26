from typing import Protocol

import numpy as np
import pandas as pd

from .metrics import calculate_metrics


class PredictorProtocol(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class Backtester:
    """Evaluate a fitted predictor on an (X, y, era) validation split.

    Args:
        predictor: Any object implementing ``predict(X) -> np.ndarray``.
        feature_columns: Subset of columns to pass to the predictor.
            If *None*, the full DataFrame is used.
    """

    def __init__(
        self,
        predictor: PredictorProtocol,
        *,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.predictor = predictor
        self.feature_columns = feature_columns

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series,
    ) -> dict[str, float]:
        """Run predictions and compute era-level backtest metrics.

        Args:
            X: Validation features.
            y: Validation targets.
            era: Era labels aligned with *X* and *y*.

        Returns:
            Dictionary with keys ``mean_per_era_correlation``,
            ``std_per_era_correlation``, ``sharpe``, and ``correlation``.
        """
        X_use = X[self.feature_columns] if self.feature_columns is not None else X
        preds = self.predictor.predict(X_use)
        return calculate_metrics(y, preds, era)
