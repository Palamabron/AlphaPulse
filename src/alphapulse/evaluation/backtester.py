from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

from .metrics import calculate_metrics, era_sharpe_of_fnc, mmc_score

if TYPE_CHECKING:
    from ..pipeline.neutralizer import FeatureNeutralizer


class PredictorProtocol(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class Backtester:
    """Evaluate a fitted predictor on an (X, y, era) validation split.

    Args:
        predictor: Any object implementing ``predict(X) -> np.ndarray``.
        feature_columns: Subset of columns to pass to the predictor.
            If *None*, the full DataFrame is used.
        neutralizer: Optional ``FeatureNeutralizer`` applied to raw predictions
            before metric computation. Rewards genuinely novel alpha over
            crowded factor exposure.
    """

    def __init__(
        self,
        predictor: PredictorProtocol,
        *,
        feature_columns: list[str] | None = None,
        neutralizer: "FeatureNeutralizer | None" = None,
    ) -> None:
        self.predictor = predictor
        self.feature_columns = feature_columns
        self.neutralizer = neutralizer

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series,
        *,
        meta_model_preds: np.ndarray | None = None,
        compute_fnc: bool = False,
        corr_weight: float = 0.75,
        mmc_weight: float = 2.25,
    ) -> dict[str, float]:
        """Run predictions and compute era-level backtest metrics.

        Args:
            X: Validation features.
            y: Validation targets.
            era: Era labels aligned with *X* and *y*.
            meta_model_preds: Optional Numerai meta model predictions for the
                same rows. When provided, ``mmc``, ``mmc_sharpe``, and
                ``payout_score`` are included in the returned metrics.
            compute_fnc: When True, compute Feature Neutral Correlation (FNC)
                using the feature columns in X. Can be slow for large feature sets.
            corr_weight: Weight for CORR Sharpe in payout formula. Default 0.75.
            mmc_weight: Weight for MMC Sharpe in payout formula. Default 2.25.

        Returns:
            Dictionary with ``mean_per_era_correlation``, ``std_per_era_correlation``,
            ``corr_sharpe``, ``max_drawdown``,
            ``pct_positive_eras``, ``n_valid_eras``, and optionally ``mmc``,
            ``mmc_sharpe``, ``payout_score``, ``fnc_sharpe``.
        """
        X_use = X[self.feature_columns] if self.feature_columns is not None else X
        preds = self.predictor.predict(X_use)

        if self.neutralizer is not None:
            preds = self.neutralizer.neutralize(preds, X_use, era)

        meta_arr = (
            np.asarray(meta_model_preds, dtype=np.float64)
            if meta_model_preds is not None
            else None
        )
        metrics = calculate_metrics(
            y,
            preds,
            era,
            meta_model_preds=meta_arr,
            corr_weight=corr_weight,
            mmc_weight=mmc_weight,
        )

        if meta_arr is not None:
            metrics["mmc"] = mmc_score(y, preds, meta_arr, era)

        if compute_fnc:
            feature_cols = self.feature_columns or list(X.columns)
            features_df = X[feature_cols]
            metrics["fnc_sharpe"] = era_sharpe_of_fnc(y, preds, features_df, era)

        return metrics
