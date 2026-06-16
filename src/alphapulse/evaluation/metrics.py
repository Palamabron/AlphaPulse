from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rank_normalize(predictions: np.ndarray) -> np.ndarray:
    """Map ranks to [0, 1] interval (Numerai-style post-processing)."""
    x = np.asarray(predictions, dtype=np.float64)
    if x.ndim != 1:
        x = x.reshape(-1)

    out = np.full_like(x, fill_value=np.nan, dtype=np.float64)
    mask = np.isfinite(x)
    if not mask.any():
        return out

    vals = x[mask]
    if vals.size == 1:
        out[mask] = 0.5
        return out

    ranks = rankdata(vals, method="average")
    out[mask] = (ranks - 1.0) / (vals.size - 1.0)
    return out


def rank_normalize_per_era(
    predictions: np.ndarray,
    eras: pd.Series,
) -> np.ndarray:
    """Rank-normalize predictions independently within each era.

    Applies ``rank_normalize`` per era so that the output matches the
    cross-sectional [0, 1] uniform distribution Numerai uses before scoring.
    Rows whose era produces fewer than 2 finite predictions are left as NaN.

    Args:
        predictions: Raw model predictions aligned with *eras*.
        eras: Era labels with the same length as *predictions*.

    Returns:
        Array of rank-normalized predictions, same shape as *predictions*.
    """
    p = np.asarray(predictions, dtype=np.float64)
    e = np.asarray(eras.to_numpy())
    if len(p) != len(e):
        raise ValueError("predictions and eras must have the same length.")
    out = np.full_like(p, fill_value=np.nan)
    for era in pd.unique(e):
        mask = e == era
        out[mask] = rank_normalize(p[mask])
    return out


def _corr_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x, method="average").astype(np.float64)
    ry = rankdata(y, method="average").astype(np.float64)
    return _corr_pearson(rx, ry)


def per_era_correlation(
    y_true: pd.Series,
    y_pred: np.ndarray,
    eras: pd.Series,
    *,
    method: Literal["pearson", "spearman"] = "spearman",
) -> pd.Series:
    """Compute per-era correlation between predictions and target.

    Numerai's primary metric (CORR) is Spearman rank correlation, so
    the default method is "spearman".
    """
    y_arr = np.asarray(y_true.to_numpy(dtype=np.float64), dtype=np.float64)
    p_arr = np.asarray(y_pred, dtype=np.float64)
    e_arr = np.asarray(eras.to_numpy())

    if not (len(y_arr) == len(p_arr) == len(e_arr)):
        raise ValueError("y_true, y_pred, and eras must have the same length.")

    if len(y_arr) == 0:
        return pd.Series(dtype=np.float64)

    eras_sorted = sorted(pd.unique(e_arr), key=str)
    out: dict[Any, float] = {}

    for era in eras_sorted:
        mask = e_arr == era
        yt = y_arr[mask]
        yp = p_arr[mask]
        finite = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[finite]
        yp = yp[finite]

        if yt.size < 2:
            out[era] = float("nan")
            continue

        if method == "pearson":
            out[era] = _corr_pearson(yt, yp)
        else:
            out[era] = _corr_spearman(yt, yp)

    return pd.Series(out, dtype=np.float64)


def per_era_spearman(
    y_true: pd.Series, y_pred: np.ndarray, eras: pd.Series
) -> pd.Series:
    return per_era_correlation(y_true, y_pred, eras, method="spearman")


def era_sharpe(y_true: pd.Series, y_pred: np.ndarray, eras: pd.Series) -> float:
    """Sharpe ratio of per-era Spearman correlations."""
    per_era = per_era_correlation(y_true, y_pred, eras, method="spearman")
    valid = per_era.dropna()
    if len(valid) == 0:
        return float("-inf")

    std = float(valid.std(ddof=0))
    if std == 0.0 or not np.isfinite(std):
        return float("-inf")

    sharpe = float(valid.mean() / std)
    if not np.isfinite(sharpe):
        return float("-inf")
    return sharpe


def _neutralize_vec(p_r: np.ndarray, m_r: np.ndarray) -> np.ndarray:
    """Remove the component of p_r in the direction of m_r (OLS residual)."""
    denom = float(m_r @ m_r)
    if denom > 0.0:
        return p_r - (float(p_r @ m_r) / denom) * m_r
    return p_r.copy()


def mmc_score(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta_model: np.ndarray,
    eras: pd.Series,
) -> float:
    """Meta Model Contribution: correlation of neutralized predictions with target.

    Measures how much your model contributes beyond the Numerai meta model.
    Computed per era by:
      1. Rank-normalizing both y_pred and meta_model within the era
      2. Centering both arrays
      3. Residualizing y_pred against meta_model (removing meta_model component)
      4. Computing Spearman correlation of residual with target
    Returns the mean over all valid eras.
    """
    valid = per_era_mmc(y_true, y_pred, meta_model, eras).dropna()
    return float(valid.mean()) if len(valid) > 0 else float("nan")


def per_era_mmc(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta_model: np.ndarray,
    eras: pd.Series,
) -> pd.Series:
    """Compute per-era MMC (Meta Model Contribution) values.

    Returns a Series indexed by era with per-era MMC correlation scores.
    """
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    meta_arr = np.asarray(meta_model, dtype=np.float64)
    y_arr = np.asarray(y_true.to_numpy(dtype=np.float64), dtype=np.float64)
    e_arr = np.asarray(eras.to_numpy())

    if not (len(pred_arr) == len(meta_arr) == len(y_arr) == len(e_arr)):
        raise ValueError(
            "y_true, y_pred, meta_model, and eras must have the same length."
        )

    eras_sorted = sorted(pd.unique(e_arr), key=str)
    out: dict[Any, float] = {}

    for era in eras_sorted:
        mask = e_arr == era
        if mask.sum() < 2:
            out[era] = float("nan")
            continue

        p = pred_arr[mask]
        m = meta_arr[mask]
        t = y_arr[mask]

        finite = np.isfinite(p) & np.isfinite(m) & np.isfinite(t)
        if finite.sum() < 2:
            out[era] = float("nan")
            continue

        p = p[finite]
        m = m[finite]
        t = t[finite]

        p_r = rankdata(p, method="average").astype(np.float64)
        m_r = rankdata(m, method="average").astype(np.float64)
        p_r -= p_r.mean()
        m_r -= m_r.mean()

        p_neutral = _neutralize_vec(p_r, m_r)
        t_r = rankdata(t, method="average").astype(np.float64)
        score = _corr_pearson(p_neutral, t_r)
        out[era] = score if np.isfinite(score) else float("nan")

    return pd.Series(out, dtype=np.float64)


def era_sharpe_of_mmc(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta_model: np.ndarray,
    eras: pd.Series,
) -> float:
    """Sharpe ratio of per-era MMC values (mean / std)."""
    per_era = per_era_mmc(y_true, y_pred, meta_model, eras)
    valid = per_era.dropna()
    if len(valid) == 0:
        return float("-inf")
    std = float(valid.std(ddof=0))
    if std == 0.0 or not np.isfinite(std):
        return float("-inf")
    sharpe = float(valid.mean() / std)
    return sharpe if np.isfinite(sharpe) else float("-inf")


def fnc_score(
    y_true: pd.Series,
    y_pred: np.ndarray,
    features_df: pd.DataFrame,
    eras: pd.Series,
) -> float:
    """FNC: mean per-era CORR after neutralizing predictions against features.

    Per era:
      1. Rank-normalize predictions and each feature column.
      2. Regress predictions on features (OLS) and take residuals.
      3. Compute Spearman correlation of residuals with target.
    Returns the mean over valid eras.

    Note: Expensive for large feature sets. Call selectively.
    """
    valid = per_era_fnc(y_true, y_pred, features_df, eras).dropna()
    return float(valid.mean()) if len(valid) > 0 else float("nan")


def per_era_fnc(
    y_true: pd.Series,
    y_pred: np.ndarray,
    features_df: pd.DataFrame,
    eras: pd.Series,
) -> pd.Series:
    """Compute per-era Feature Neutral Correlation values."""
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    y_arr = np.asarray(y_true.to_numpy(dtype=np.float64), dtype=np.float64)
    e_arr = np.asarray(eras.to_numpy())
    feat_arr = np.asarray(features_df.to_numpy(dtype=np.float64))

    if not (len(pred_arr) == len(y_arr) == len(e_arr) == len(feat_arr)):
        raise ValueError(
            "y_true, y_pred, features_df, and eras must have the same length."
        )

    eras_sorted = sorted(pd.unique(e_arr), key=str)
    out: dict[Any, float] = {}

    for era in eras_sorted:
        mask = e_arr == era
        if mask.sum() < 2:
            out[era] = float("nan")
            continue

        p = pred_arr[mask]
        t = y_arr[mask]
        F = feat_arr[mask]

        valid_rows = np.isfinite(p) & np.isfinite(t) & np.all(np.isfinite(F), axis=1)
        if valid_rows.sum() < 2:
            out[era] = float("nan")
            continue

        p = p[valid_rows]
        t = t[valid_rows]
        F = F[valid_rows]

        p_r = rankdata(p, method="average").astype(np.float64)
        F_r = np.column_stack(
            [
                rankdata(F[:, j], method="average").astype(np.float64)
                for j in range(F.shape[1])
            ]
        )
        p_r -= p_r.mean()
        F_r -= F_r.mean(axis=0)

        beta, _, _, _ = np.linalg.lstsq(F_r, p_r, rcond=None)
        residual = p_r - F_r @ beta

        t_r = rankdata(t, method="average").astype(np.float64)
        score = _corr_pearson(residual, t_r)
        out[era] = score if np.isfinite(score) else float("nan")

    return pd.Series(out, dtype=np.float64)


def era_sharpe_of_fnc(
    y_true: pd.Series,
    y_pred: np.ndarray,
    features_df: pd.DataFrame,
    eras: pd.Series,
) -> float:
    """Sharpe ratio of per-era FNC values (mean / std)."""
    per_era = per_era_fnc(y_true, y_pred, features_df, eras)
    valid = per_era.dropna()
    if len(valid) == 0:
        return float("-inf")
    std = float(valid.std(ddof=0))
    if std == 0.0 or not np.isfinite(std):
        return float("-inf")
    sharpe = float(valid.mean() / std)
    return sharpe if np.isfinite(sharpe) else float("-inf")


def payout_score(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta_model_preds: np.ndarray,
    eras: pd.Series,
    corr_weight: float = 0.75,
    mmc_weight: float = 2.25,
) -> float:
    """Numerai payout formula: corr_weight * corr_sharpe + mmc_weight * mmc_sharpe.

    Default weights reflect the 0.75*CORR20v2 + 2.25*MMC tournament formula.
    Both components use Sharpe ratios (mean/std) of per-era scores.
    """
    cs = era_sharpe(y_true, y_pred, eras)
    ms = era_sharpe_of_mmc(y_true, y_pred, meta_model_preds, eras)
    cs_val = cs if np.isfinite(cs) else 0.0
    ms_val = ms if np.isfinite(ms) else 0.0
    return corr_weight * cs_val + mmc_weight * ms_val


def era_correlation_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    eras: pd.Series,
) -> dict[str, float]:
    per_era = per_era_correlation(y_true, y_pred, eras, method="spearman")
    valid = per_era.dropna()
    n_valid_eras = int(len(valid))

    if n_valid_eras == 0:
        return {
            "mean_era_corr": float("nan"),
            "std_era_corr": float("nan"),
            "corr_sharpe": float("-inf"),
            "max_drawdown": 0.0,
            "pct_positive_eras": 0.0,
            "n_valid_eras": 0.0,
        }

    mean_era_corr = float(valid.mean())
    std_era_corr = float(valid.std(ddof=0)) if n_valid_eras > 1 else float("nan")
    corr_sharpe = era_sharpe(y_true, y_pred, eras)

    pct_positive_eras = float((valid > 0.0).mean())

    cum = valid.cumsum().to_numpy(dtype=np.float64)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_drawdown = float(np.max(dd)) if dd.size else 0.0

    return {
        "mean_era_corr": mean_era_corr,
        "std_era_corr": std_era_corr,
        "corr_sharpe": corr_sharpe,
        "max_drawdown": max_drawdown,
        "pct_positive_eras": pct_positive_eras,
        "n_valid_eras": float(n_valid_eras),
    }


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    eras: pd.Series,
    *,
    meta_model_preds: np.ndarray | None = None,
    corr_weight: float = 0.75,
    mmc_weight: float = 2.25,
) -> dict[str, float]:
    """Compute the canonical backtest metric dict.

    Args:
        y_true: True target values.
        y_pred: Model predictions.
        eras: Era labels aligned with y_true and y_pred.
        meta_model_preds: Optional Numerai meta model predictions for the same rows.
            When provided, ``mmc_sharpe`` and ``payout_score`` are included.
        corr_weight: Weight for CORR Sharpe in payout formula. Default 0.75.
        mmc_weight: Weight for MMC Sharpe in payout formula. Default 2.25.

    Returns:
        Dict with ``mean_per_era_correlation``, ``std_per_era_correlation``,
        ``corr_sharpe``, ``max_drawdown``, ``pct_positive_eras``,
        ``n_valid_eras``.
        When meta model is provided: also ``mmc_sharpe`` and ``payout_score``.
    """
    scoring = era_correlation_metrics(y_true, y_pred, eras)
    mean_corr = scoring["mean_era_corr"]
    corr_sharpe = scoring["corr_sharpe"]
    result: dict[str, float] = {
        "mean_per_era_correlation": mean_corr,
        "std_per_era_correlation": scoring["std_era_corr"],
        "corr_sharpe": corr_sharpe,
        "max_drawdown": scoring["max_drawdown"],
        "pct_positive_eras": scoring["pct_positive_eras"],
        "n_valid_eras": scoring["n_valid_eras"],
    }
    if meta_model_preds is not None:
        meta_arr = np.asarray(meta_model_preds, dtype=np.float64)
        if np.isfinite(meta_arr).sum() >= 2:
            ms = era_sharpe_of_mmc(y_true, y_pred, meta_arr, eras)
            result["mmc_sharpe"] = ms
            ps = payout_score(
                y_true,
                y_pred,
                meta_arr,
                eras,
                corr_weight=corr_weight,
                mmc_weight=mmc_weight,
            )
            result["payout_score"] = ps
    return result
