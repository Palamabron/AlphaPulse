"""Ensemble diversity diagnostics for Numerai pipelines.

A good Numerai ensemble combines models that are as decorrelated from each
other as possible while still predicting the target well. High inter-model
correlation means the ensemble gains little over a single model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def correlation_matrix(
    oof_predictions: dict[str, np.ndarray],
    eras: pd.Series,
) -> pd.DataFrame:
    """Compute per-era-averaged Spearman correlation between model predictions.

    For each pair of models, the per-era Spearman correlation is averaged.
    This is more meaningful than a raw correlation because it respects
    Numerai's era structure (predictions are rank-normalized within each era).

    Args:
        oof_predictions: Mapping of model name → 1-D array of OOF predictions.
        eras: Era labels aligned with the prediction arrays.

    Returns:
        Square DataFrame with model names as index and columns.
    """
    names = list(oof_predictions.keys())
    n = len(names)
    e_arr = np.asarray(eras.to_numpy())
    unique_eras = sorted(pd.unique(e_arr), key=str)

    era_corr_sums = np.zeros((n, n), dtype=np.float64)
    era_count = 0

    for era in unique_eras:
        mask = e_arr == era
        if mask.sum() < 2:
            continue
        era_preds = np.stack(
            [rankdata(oof_predictions[name][mask], method="average") for name in names],
            axis=1,
        )
        corr = np.corrcoef(era_preds.T)
        if np.all(np.isfinite(corr)):
            era_corr_sums += corr
            era_count += 1

    if era_count == 0:
        corr_mean = np.eye(n)
    else:
        corr_mean = era_corr_sums / era_count

    return pd.DataFrame(corr_mean, index=names, columns=names)


def effective_model_count(
    weights: np.ndarray,
    corr_matrix_df: pd.DataFrame,
) -> float:
    """Estimate effective number of independent models in the ensemble.

    Uses the portfolio-theory idea: N_eff = 1 / sum(w_i * w_j * rho_ij).
    Returns a value in [1, n_models]; higher = more diverse ensemble.

    Args:
        weights: 1-D array of model weights (must sum to 1).
        corr_matrix_df: Square correlation matrix from ``correlation_matrix()``.

    Returns:
        Effective model count (float).
    """
    w = np.asarray(weights, dtype=np.float64)
    C = corr_matrix_df.to_numpy(dtype=np.float64)
    portfolio_variance = float(w @ C @ w)
    if portfolio_variance <= 0.0:
        return float(len(w))
    return 1.0 / portfolio_variance


def compute_ensemble_diagnostics(
    oof_predictions: dict[str, np.ndarray],
    y_oof: np.ndarray,
    eras_oof: pd.Series,
    *,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute ensemble diversity diagnostics.

    Args:
        oof_predictions: Mapping of model name → 1-D array of OOF predictions.
        y_oof: True target values aligned with OOF predictions.
        eras_oof: Era labels aligned with OOF predictions.
        weights: Optional 1-D weight array (must sum to 1). When None, equal
            weights are used for the effective model count.

    Returns:
        Dict with:
          - ``correlation_matrix``: nested dict (model → model → mean_corr).
          - ``effective_model_count``: float — higher is more diverse.
          - ``mean_pairwise_correlation``: float — lower is more diverse.
          - ``model_names``: list of model names.
    """
    names = list(oof_predictions.keys())
    n = len(names)

    corr_df = correlation_matrix(oof_predictions, eras_oof)

    if weights is None:
        w = np.ones(n, dtype=np.float64) / n
    else:
        w = np.asarray(weights, dtype=np.float64)
        if not np.isclose(w.sum(), 1.0):
            w = w / w.sum()

    eff_count = effective_model_count(w, corr_df)

    mask_off_diag = ~np.eye(n, dtype=bool)
    off_diag_vals = corr_df.to_numpy()[mask_off_diag]
    mean_pairwise = (
        float(np.nanmean(off_diag_vals)) if len(off_diag_vals) > 0 else float("nan")
    )

    return {
        "correlation_matrix": corr_df.round(4).to_dict(),
        "effective_model_count": round(eff_count, 4),
        "mean_pairwise_correlation": round(mean_pairwise, 4),
        "model_names": names,
    }
