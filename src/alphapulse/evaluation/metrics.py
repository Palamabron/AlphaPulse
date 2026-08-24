from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

NUMERAI_TOOLS_REFERENCE_VERSION = "0.6.0"
NUMERAI_MAX_FILTERED_RATIO = 0.2
NUMERAI_SEASON_CORR_WEIGHT = 0.75
NUMERAI_SEASON_MMC_WEIGHT = 2.25


def rank_normalize(predictions: np.ndarray) -> np.ndarray:
    """Map ranks to [0, 1] for AlphaPulse's legacy prediction post-processing.

    This transform is not the tie-kept percentile rank used internally by the
    official Numerai scoring functions.
    """
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


def _target_series(values: pd.Series, name: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series with row IDs as its index.")
    if values.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicate row IDs.")
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty.")
    result = pd.Series(
        values.to_numpy(dtype=np.float64),
        index=values.index,
        name=name,
    )
    if np.isinf(result.to_numpy(dtype=np.float64)).any():
        raise ValueError(f"{name} must not contain infinite values.")
    return result


def _aligned_float_series(
    values: pd.Series | np.ndarray,
    name: str,
    index: pd.Index,
) -> pd.Series:
    if isinstance(values, pd.Series):
        if values.index.has_duplicates:
            raise ValueError(f"{name} index must not contain duplicate row IDs.")
        missing = index.difference(values.index)
        extra = values.index.difference(index)
        if len(missing) > 0 or len(extra) > 0:
            raise ValueError(
                f"{name} row IDs must exactly match y_true; "
                f"missing={len(missing)}, extra={len(extra)}."
            )
        result = pd.Series(
            values.reindex(index).to_numpy(dtype=np.float64),
            index=index,
            name=name,
        )
    else:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if len(array) != len(index):
            raise ValueError(
                f"{name} length {len(array)} does not match y_true length {len(index)}."
            )
        result = pd.Series(array, index=index, name=name)
    if np.isinf(result.to_numpy(dtype=np.float64)).any():
        raise ValueError(f"{name} must not contain infinite values.")
    return result


def _aligned_era_series(eras: pd.Series, index: pd.Index) -> pd.Series:
    if not isinstance(eras, pd.Series):
        raise TypeError("eras must be a pandas Series with row IDs as its index.")
    if eras.index.has_duplicates:
        raise ValueError("eras index must not contain duplicate row IDs.")
    missing = index.difference(eras.index)
    extra = eras.index.difference(index)
    if len(missing) > 0 or len(extra) > 0:
        raise ValueError(
            "eras row IDs must exactly match y_true; "
            f"missing={len(missing)}, extra={len(extra)}."
        )
    result = eras.reindex(index)
    if result.isna().any():
        raise ValueError("eras must not contain missing values.")
    return result


def _validate_official_metric_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eras: np.ndarray,
    meta_model: np.ndarray | None = None,
) -> None:
    lengths = {len(y_true), len(y_pred), len(eras)}
    if meta_model is not None:
        lengths.add(len(meta_model))
    if len(lengths) != 1:
        names = (
            "y_true, y_pred, meta_model, and eras"
            if meta_model is not None
            else "y_true, y_pred, and eras"
        )
        raise ValueError(f"{names} must have the same length.")
    if pd.isna(eras).any():
        raise ValueError("eras must not contain missing values.")


def _filter_numerai_missing_rows(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    if not arrays or len(arrays[0]) == 0:
        return arrays
    missing = np.zeros(len(arrays[0]), dtype=bool)
    for array in arrays:
        missing |= np.asarray(pd.isna(array), dtype=bool)
    retained_ratio = float((~missing).mean())
    if retained_ratio < 1.0 - NUMERAI_MAX_FILTERED_RATIO:
        raise ValueError(
            "Numerai scoring requires at least 80% non-missing aligned rows "
            "within each era."
        )
    return tuple(array[~missing] for array in arrays)


def _tie_kept_rank_gaussian(values: np.ndarray) -> np.ndarray:
    percentile_ranks = (
        rankdata(values, method="average").astype(np.float64) - 0.5
    ) / len(values)
    return np.asarray(norm.ppf(percentile_ranks), dtype=np.float64)


def _signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.asarray(
        np.sign(values) * np.abs(values) ** exponent,
        dtype=np.float64,
    )


def _numerai_corr_for_era(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    centered_target = y_true - float(pd.Series(y_true).mean())
    target, prediction = _filter_numerai_missing_rows(centered_target, y_pred)
    if len(target) < 2:
        return float("nan")
    transformed_prediction = _signed_power(_tie_kept_rank_gaussian(prediction), 1.5)
    transformed_target = _signed_power(target, 1.5)
    return _corr_pearson(transformed_target, transformed_prediction)


def _numerai_mmc_for_era(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta_model: np.ndarray,
) -> float:
    target, prediction, meta = _filter_numerai_missing_rows(y_true, y_pred, meta_model)
    if len(target) == 0:
        return float("nan")
    normalized_prediction = _tie_kept_rank_gaussian(prediction)
    normalized_meta = _tie_kept_rank_gaussian(meta)
    meta_norm = float(normalized_meta @ normalized_meta)
    if meta_norm == 0.0:
        return float("nan")
    projection = float(normalized_prediction @ normalized_meta) / meta_norm
    neutral_prediction = normalized_prediction - normalized_meta * projection
    if bool(np.all((target >= 0.0) & (target <= 1.0))):
        target = target * 4.0
    centered_target = target - target.mean()
    return float(centered_target @ neutral_prediction / len(centered_target))


def per_era_numerai_corr(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    eras: pd.Series,
) -> pd.Series:
    """Compute official Numerai Corr independently within each era.

    The implementation reproduces the numerical transforms in ``numerai_corr``
    from ``numerai-tools==0.6.0``: tie-kept percentile ranking,
    Gaussianization, signed power 1.5, target centering, and Pearson
    correlation. Indexed inputs are aligned by row ID and must contain exactly
    the same ID set. Missing values may filter at most 20% of an era.

    Source: https://pypi.org/project/numerai-tools/0.6.0/
    """
    target_series = _target_series(y_true, "y_true")
    prediction_series = _aligned_float_series(y_pred, "y_pred", target_series.index)
    era_series = _aligned_era_series(eras, target_series.index)
    target = target_series.to_numpy(dtype=np.float64)
    prediction = prediction_series.to_numpy(dtype=np.float64)
    era_values = era_series.to_numpy()
    _validate_official_metric_inputs(target, prediction, era_values)

    scores: dict[Any, float] = {}
    for era in sorted(pd.unique(era_values), key=str):
        mask = era_values == era
        scores[era] = _numerai_corr_for_era(target[mask], prediction[mask])
    return pd.Series(scores, dtype=np.float64)


def per_era_numerai_mmc(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    meta_model: pd.Series | np.ndarray,
    eras: pd.Series,
) -> pd.Series:
    """Compute official Numerai MMC independently within each era.

    The implementation reproduces the numerical transforms in
    ``correlation_contribution`` from ``numerai-tools==0.6.0``. Predictions and
    the meta model are tie-kept ranked and Gaussianized, predictions are
    orthogonalized to the meta model, and the score is covariance with the
    centered target. Indexed inputs are aligned by row ID and must contain
    exactly the same ID set.

    Source: https://pypi.org/project/numerai-tools/0.6.0/
    """
    target_series = _target_series(y_true, "y_true")
    prediction_series = _aligned_float_series(y_pred, "y_pred", target_series.index)
    meta_series = _aligned_float_series(meta_model, "meta_model", target_series.index)
    era_series = _aligned_era_series(eras, target_series.index)
    target = target_series.to_numpy(dtype=np.float64)
    prediction = prediction_series.to_numpy(dtype=np.float64)
    meta = meta_series.to_numpy(dtype=np.float64)
    era_values = era_series.to_numpy()
    _validate_official_metric_inputs(target, prediction, era_values, meta)

    scores: dict[Any, float] = {}
    for era in sorted(pd.unique(era_values), key=str):
        mask = era_values == era
        scores[era] = _numerai_mmc_for_era(target[mask], prediction[mask], meta[mask])
    return pd.Series(scores, dtype=np.float64)


def per_era_weighted_corr_mmc(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    meta_model: pd.Series | np.ndarray,
    eras: pd.Series,
    corr_weight: float = NUMERAI_SEASON_CORR_WEIGHT,
    mmc_weight: float = NUMERAI_SEASON_MMC_WEIGHT,
) -> pd.Series:
    """Build a diagnostic weighted series from Numerai CORR and MMC.

    The defaults reproduce the historical 0.75 CORR plus 2.25 MMC composition.
    This is an official Season score only when the target, horizon, SWMM,
    component versions, and multipliers all match the score configuration for
    the relevant round. Invalid component values propagate as NaN; this helper
    does not infer that a component was intentionally absent.

    Source: https://docs.numer.ai/numerai-tournament/scoring/definitions
    """
    corr = per_era_numerai_corr(y_true, y_pred, eras)
    mmc = per_era_numerai_mmc(y_true, y_pred, meta_model, eras)
    return corr * corr_weight + mmc * mmc_weight


def weighted_corr_mmc_sharpe(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    meta_model: pd.Series | np.ndarray,
    eras: pd.Series,
    corr_weight: float = NUMERAI_SEASON_CORR_WEIGHT,
    mmc_weight: float = NUMERAI_SEASON_MMC_WEIGHT,
) -> float:
    """Compute a diagnostic Sharpe over a weighted CORR-plus-MMC series.

    The arithmetic matches ``sharpe_ratio`` from ``numerai-tools==0.6.0``. It is
    not itself a published tournament payout metric. Any invalid per-era
    component makes the aggregate invalid instead of being silently discarded.

    Source: https://pypi.org/project/numerai-tools/0.6.0/
    """
    scores = per_era_weighted_corr_mmc(
        y_true,
        y_pred,
        meta_model,
        eras,
        corr_weight=corr_weight,
        mmc_weight=mmc_weight,
    )
    if scores.isna().any():
        return float("nan")
    valid = scores.to_numpy(dtype=np.float64)
    if len(valid) == 0:
        return float("nan")
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=0))
    if std == 0.0:
        if mean == 0.0:
            return float("nan")
        return float(np.copysign(np.inf, mean))
    return mean / std


def _score_series_summary(
    scores: pd.Series,
    *,
    prefix: str,
) -> dict[str, float]:
    valid = scores.dropna()
    if len(valid) == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_sharpe": float("-inf"),
            f"{prefix}_max_drawdown": float("nan"),
            f"{prefix}_pct_positive_eras": float("nan"),
        }
    values = valid.to_numpy(dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if std == 0.0:
        sharpe = float("nan") if mean == 0.0 else float(np.copysign(np.inf, mean))
    else:
        sharpe = mean / std
    cumulative = np.cumsum(values)
    drawdown = np.maximum.accumulate(cumulative) - cumulative
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_sharpe": float(sharpe),
        f"{prefix}_max_drawdown": float(np.max(drawdown)),
        f"{prefix}_pct_positive_eras": float(np.mean(values > 0.0)),
    }


def numerai_official_diagnostics(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    eras: pd.Series,
    *,
    meta_model: pd.Series | np.ndarray | None = None,
    corr_weight: float = NUMERAI_SEASON_CORR_WEIGHT,
    mmc_weight: float = NUMERAI_SEASON_MMC_WEIGHT,
) -> dict[str, float]:
    """Summarize frozen Numerai CORR/MMC components under explicit names."""
    corr = per_era_numerai_corr(y_true, y_pred, eras)
    result = _score_series_summary(corr, prefix="numerai_corr")
    if meta_model is None:
        return result
    mmc = per_era_numerai_mmc(y_true, y_pred, meta_model, eras)
    weighted = corr * corr_weight + mmc * mmc_weight
    result.update(_score_series_summary(mmc, prefix="numerai_mmc"))
    result.update(_score_series_summary(weighted, prefix="weighted_corr_mmc"))
    return result


def per_era_correlation(
    y_true: pd.Series,
    y_pred: np.ndarray,
    eras: pd.Series,
    *,
    method: Literal["pearson", "spearman"] = "spearman",
) -> pd.Series:
    """Compute AlphaPulse's legacy per-era Pearson or Spearman correlation.

    The historical default is Spearman and is retained for compatibility with
    existing HPO artifacts. It is not official Numerai Corr; use
    :func:`per_era_numerai_corr` for the frozen official definition.
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
    """Compute the legacy AlphaPulse Spearman correlation per era."""
    return per_era_correlation(y_true, y_pred, eras, method="spearman")


def era_sharpe(y_true: pd.Series, y_pred: np.ndarray, eras: pd.Series) -> float:
    """Compute the legacy Sharpe ratio of per-era Spearman correlations."""
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
    """Compute AlphaPulse's legacy MMC correlation approximation.

    This historical metric averages Pearson correlations between rank-based
    residual predictions and ranked targets. It is retained for compatibility
    and is not official covariance-based Numerai MMC. Use
    :func:`per_era_numerai_mmc` for the frozen official definition.
    """
    valid = per_era_mmc(y_true, y_pred, meta_model, eras).dropna()
    return float(valid.mean()) if len(valid) > 0 else float("nan")


def per_era_mmc(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta_model: np.ndarray,
    eras: pd.Series,
) -> pd.Series:
    """Compute AlphaPulse's legacy per-era MMC correlation approximation.

    The function is retained unchanged for historical HPO compatibility. Its
    values must not be presented as official Numerai MMC.
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
    """Compute Sharpe over legacy AlphaPulse MMC correlation values."""
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
    """Compute AlphaPulse's legacy weighted-Sharpe HPO proxy.

    The formula combines legacy CORR and MMC Sharpes rather than combining
    official per-era scores. It is retained for historical HPO compatibility
    and is neither the official Numerai Season score nor an NMR payout amount.
    Use :func:`per_era_numerai_corr` and :func:`per_era_numerai_mmc` for the
    frozen official component definitions. The diagnostic composition helper
    is :func:`per_era_weighted_corr_mmc`.
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
    """Summarize AlphaPulse's legacy per-era Spearman correlation series."""
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
    """Compute the legacy-compatible AlphaPulse backtest metric dict.

    Existing keys retain their historical definitions so stored HPO results
    remain comparable. Official Numerai components are intentionally available
    through the separately named ``per_era_numerai_*`` functions and do not
    replace the historical HPO objective.

    Args:
        y_true: True target values.
        y_pred: Model predictions.
        eras: Era labels aligned with y_true and y_pred.
        meta_model_preds: Optional Numerai meta model predictions for the same rows.
            When provided, legacy ``mmc_sharpe`` and ``payout_score`` values are
            included.
        corr_weight: Weight for legacy correlation Sharpe. Defaults to 0.75.
        mmc_weight: Weight for legacy MMC Sharpe. Defaults to 2.25.

    Returns:
        Dict with ``mean_per_era_correlation``, ``std_per_era_correlation``,
        ``corr_sharpe``, ``max_drawdown``, ``pct_positive_eras``,
        ``n_valid_eras``.
        When a meta model is provided, legacy ``mmc_sharpe`` and
        ``payout_score`` values are included.
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
    result.update(
        numerai_official_diagnostics(
            y_true,
            y_pred,
            eras,
            meta_model=meta_model_preds,
            corr_weight=corr_weight,
            mmc_weight=mmc_weight,
        )
    )
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
