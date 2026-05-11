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

    eras_sorted = sorted(pd.unique(e_arr), key=lambda v: str(v))
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
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    meta_arr = np.asarray(meta_model, dtype=np.float64)
    y_arr = np.asarray(y_true.to_numpy(dtype=np.float64), dtype=np.float64)
    e_arr = np.asarray(eras.to_numpy())

    if not (len(pred_arr) == len(meta_arr) == len(y_arr) == len(e_arr)):
        raise ValueError(
            "y_true, y_pred, meta_model, and eras must have the same length."
        )

    eras_sorted = sorted(pd.unique(e_arr), key=lambda v: str(v))
    era_scores: list[float] = []

    for era in eras_sorted:
        mask = e_arr == era
        if mask.sum() < 2:
            continue

        p = pred_arr[mask]
        m = meta_arr[mask]
        t = y_arr[mask]

        finite = np.isfinite(p) & np.isfinite(m) & np.isfinite(t)
        if finite.sum() < 2:
            continue

        p = p[finite]
        m = m[finite]
        t = t[finite]

        p_r = rankdata(p, method="average").astype(np.float64)
        m_r = rankdata(m, method="average").astype(np.float64)
        p_r -= p_r.mean()
        m_r -= m_r.mean()

        denom = float(m_r @ m_r)
        if denom > 0.0:
            p_neutral = p_r - (float(p_r @ m_r) / denom) * m_r
        else:
            p_neutral = p_r

        t_r = rankdata(t, method="average").astype(np.float64)
        score = _corr_pearson(p_neutral, t_r)
        if np.isfinite(score):
            era_scores.append(score)

    return float(np.mean(era_scores)) if era_scores else float("nan")


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

    cum = valid.to_numpy(dtype=np.float64)
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
) -> dict[str, float]:
    """Compute the canonical backtest metric dict.

    Returns:
        Dict with ``mean_per_era_correlation``, ``std_per_era_correlation``,
        ``corr_sharpe``, and the legacy aliases ``sharpe`` and ``correlation``
        (equal to ``corr_sharpe`` and ``mean_per_era_correlation`` respectively).
    """
    scoring = era_correlation_metrics(y_true, y_pred, eras)
    mean_corr = scoring["mean_era_corr"]
    corr_sharpe = scoring["corr_sharpe"]
    return {
        "mean_per_era_correlation": mean_corr,
        "std_per_era_correlation": scoring["std_era_corr"],
        "corr_sharpe": corr_sharpe,
        "sharpe": corr_sharpe,
        "correlation": mean_corr,
    }
