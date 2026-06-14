from __future__ import annotations

import numpy as np
import pandas as pd

PROTECTED_METADATA_COLUMNS = frozenset({"era", "id", "data_type"})


def invalid_row_mask(X: pd.DataFrame, y: pd.Series | None = None) -> pd.Series:
    numeric = X.select_dtypes(include=[np.number])
    x_invalid = X.isna().any(axis=1)
    if not numeric.empty:
        x_invalid = x_invalid | np.isinf(numeric).any(axis=1)
    if y is not None:
        y_invalid = y.isna() | np.isinf(y)
        return x_invalid | y_invalid
    return x_invalid


def filter_invalid_rows(
    X: pd.DataFrame, y: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series | None]:
    mask = invalid_row_mask(X, y)
    if not mask.any():
        return X, y
    valid = ~mask
    y_filtered = y[valid] if y is not None else None
    return X[valid], y_filtered


def filter_nan_rows(
    X: pd.DataFrame, y: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series | None]:
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
