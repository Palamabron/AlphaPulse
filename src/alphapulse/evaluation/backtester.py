from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

from ..utils.alignment import align_series_to_frame
from .metrics import calculate_metrics, era_sharpe_of_fnc, mmc_score

if TYPE_CHECKING:
    from ..pipeline.neutralizer import FeatureNeutralizer


class PredictorProtocol(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def predict_with_optional_eras(
    predictor: PredictorProtocol,
    X: pd.DataFrame,
    era: pd.Series,
    meta_model_preds: np.ndarray | None = None,
) -> np.ndarray:
    uses_meta_neutralization = (
        float(getattr(predictor, "meta_neutralize_proportion", 0.0)) > 0.0
    )
    if uses_meta_neutralization and meta_model_preds is None:
        raise ValueError(
            "Meta-model neutralization is configured, but aligned meta-model "
            "predictions were not provided"
        )
    meta_kw = (
        {"meta_model": meta_model_preds}
        if meta_model_preds is not None and uses_meta_neutralization
        else {}
    )
    uses_feature_neutralization = (
        float(getattr(predictor, "neutralize_proportion", 0.0)) > 0.0
    )
    if uses_feature_neutralization or meta_kw:
        return np.asarray(
            predictor.predict(X, eras=era, **meta_kw),  # type: ignore[call-arg]
            dtype=np.float64,
        )
    return np.asarray(predictor.predict(X), dtype=np.float64)


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
        meta_model_preds: np.ndarray | pd.Series | None = None,
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
            corr_weight: Weight for legacy correlation Sharpe. Default 0.75.
            mmc_weight: Weight for legacy MMC Sharpe. Default 2.25.

        Returns:
            Dictionary with ``mean_per_era_correlation``, ``std_per_era_correlation``,
            ``corr_sharpe``, ``max_drawdown``,
            ``pct_positive_eras``, ``n_valid_eras``, and optionally ``mmc``,
            ``mmc_sharpe``, ``payout_score``, ``fnc_sharpe``.
        """
        y = align_series_to_frame(X, y, name="target")
        era = align_series_to_frame(X, era, name="era")
        if isinstance(meta_model_preds, pd.Series):
            meta_model_preds = align_series_to_frame(
                X,
                meta_model_preds,
                name="meta_model_preds",
            )
        if meta_model_preds is not None:
            meta_array = np.asarray(meta_model_preds, dtype=np.float64).reshape(-1)
            if len(meta_array) != len(X):
                raise ValueError(
                    "meta_model_preds length must match the validation frame"
                )
            if not np.isfinite(meta_array).all():
                raise ValueError(
                    "meta_model_preds contains missing or non-finite predictions"
                )
        else:
            meta_array = None

        X_use = X[self.feature_columns] if self.feature_columns is not None else X
        preds = predict_with_optional_eras(
            self.predictor, X_use, era, meta_model_preds=meta_array
        )

        if self.neutralizer is not None:
            preds = self.neutralizer.neutralize(preds, X_use, era)

        metrics = calculate_metrics(
            y,
            preds,
            era,
            meta_model_preds=meta_array,
            corr_weight=corr_weight,
            mmc_weight=mmc_weight,
        )

        if meta_array is not None:
            metrics["mmc"] = mmc_score(y, preds, meta_array, era)

        if compute_fnc:
            feature_cols = self.feature_columns or list(X.columns)
            features_df = X[feature_cols]
            metrics["fnc_sharpe"] = era_sharpe_of_fnc(y, preds, features_df, era)

        return metrics
