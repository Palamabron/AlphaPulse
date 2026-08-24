from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.alignment import align_series_to_frame

PROTECTED_METADATA_COLUMNS = frozenset({"era", "id", "data_type"})


def invalid_row_mask(X: pd.DataFrame, y: pd.Series | None = None) -> pd.Series:
    numeric = X.select_dtypes(include=[np.number])
    x_invalid = X.isna().any(axis=1)
    if not numeric.empty:
        x_invalid = x_invalid | np.isinf(numeric).any(axis=1)
    if y is not None:
        y = align_series_to_frame(X, y)
        y_invalid = y.isna() | np.isinf(y)
        return x_invalid | y_invalid
    return x_invalid


def filter_invalid_rows(
    X: pd.DataFrame, y: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series | None]:
    if y is not None:
        y = align_series_to_frame(X, y)
    mask = invalid_row_mask(X, y)
    if not mask.any():
        return X, y
    valid = ~mask
    y_filtered = y[valid] if y is not None else None
    return X[valid], y_filtered


def filter_nan_rows(
    X: pd.DataFrame, y: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series | None]:
    if y is not None:
        y = align_series_to_frame(X, y)
    if not X.isna().any().any():
        return X, y
    nan_mask = ~X.isna().any(axis=1)
    y_filtered = y[nan_mask] if y is not None else None
    return X[nan_mask], y_filtered


def protected_metadata_frame(X: pd.DataFrame) -> pd.DataFrame | None:
    cols = [c for c in PROTECTED_METADATA_COLUMNS if c in X.columns]
    if not cols:
        return None
    return X[cols].copy()


def reattach_protected_columns(
    X: pd.DataFrame, meta: pd.DataFrame | None
) -> pd.DataFrame:
    if meta is None:
        return X
    out = X.copy()
    for col in meta.columns:
        if col not in out.columns:
            out[col] = meta.loc[out.index, col]
    return out


def safe_median(values: np.ndarray, default: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return default
    return float(np.median(finite))


def select_required_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Live input is missing {len(missing)} required feature column(s): "
            f"{missing[:10]}" + (" (and more)" if len(missing) > 10 else "")
        )
    return frame.reindex(columns=feature_columns)


def blend_with_benchmark(
    predictions: np.ndarray,
    benchmark_models: pd.DataFrame,
    index: pd.Index,
    *,
    benchmark_column: str | None,
    weight: float,
) -> np.ndarray:
    if weight <= 0.0:
        return predictions
    if not benchmark_column:
        raise ValueError(
            "Benchmark blending is configured, but no benchmark column was specified"
        )
    if benchmark_column not in benchmark_models.columns:
        raise ValueError(f"Benchmark blending requires column {benchmark_column!r}")
    benchmark = (
        benchmark_models[benchmark_column].reindex(index).to_numpy(dtype=np.float64)
    )
    if not np.isfinite(benchmark).all():
        raise ValueError(
            f"Benchmark column {benchmark_column!r} is missing or non-finite "
            "for one or more live rows"
        )
    return np.asarray(
        (1.0 - weight) * predictions + weight * benchmark,
        dtype=np.float64,
    )
