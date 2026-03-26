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
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.Series:
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
    per_era = per_era_correlation(y_true, y_pred, eras, method="pearson")
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


def era_correlation_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    eras: pd.Series,
) -> dict[str, float]:
    per_era = per_era_correlation(y_true, y_pred, eras, method="pearson")
    valid = per_era.dropna()
    n_valid_eras = int(len(valid))

    if n_valid_eras == 0:
        return {
            "mean_era_corr": float("nan"),
            "std_era_corr": float("nan"),
            "sharpe_era_corr": float("-inf"),
            "max_drawdown": 0.0,
            "pct_positive_eras": 0.0,
            "n_valid_eras": 0.0,
        }

    mean_era_corr = float(valid.mean())
    std_era_corr = float(valid.std(ddof=0)) if n_valid_eras > 1 else float("nan")
    sharpe_era_corr = era_sharpe(y_true, y_pred, eras)

    pct_positive_eras = float((valid > 0.0).mean())

    cum = valid.to_numpy(dtype=np.float64)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_drawdown = float(np.max(dd)) if dd.size else 0.0

    return {
        "mean_era_corr": mean_era_corr,
        "std_era_corr": std_era_corr,
        "sharpe_era_corr": sharpe_era_corr,
        "max_drawdown": max_drawdown,
        "pct_positive_eras": pct_positive_eras,
        "n_valid_eras": float(n_valid_eras),
    }


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    eras: pd.Series,
) -> dict[str, float]:
    scoring = era_correlation_metrics(y_true, y_pred, eras)
    return {
        "mean_per_era_correlation": scoring["mean_era_corr"],
        "std_per_era_correlation": scoring["std_era_corr"],
        "sharpe": scoring["sharpe_era_corr"],
        "correlation": scoring["mean_era_corr"],
    }
