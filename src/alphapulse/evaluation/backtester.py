from typing import Protocol

import numpy as np
import pandas as pd

from .metrics import calculate_metrics, mmc_score


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
        *,
        meta_model_preds: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Run predictions and compute era-level backtest metrics.

        Args:
            X: Validation features.
            y: Validation targets.
            era: Era labels aligned with *X* and *y*.
            meta_model_preds: Optional Numerai meta model predictions for the
                same rows. When provided, ``mmc`` (Meta Model Contribution) is
                included in the returned metrics.

        Returns:
            Dictionary with keys ``mean_per_era_correlation``,
            ``std_per_era_correlation``, ``corr_sharpe``, ``sharpe``,
            ``correlation``, and optionally ``mmc``.
        """
        X_use = X[self.feature_columns] if self.feature_columns is not None else X
        preds = self.predictor.predict(X_use)
        metrics = calculate_metrics(y, preds, era)

        if meta_model_preds is not None:
            metrics["mmc"] = mmc_score(y, preds, np.asarray(meta_model_preds), era)

        return metrics
