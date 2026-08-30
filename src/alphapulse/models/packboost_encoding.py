from __future__ import annotations

import numpy as np
import pandas as pd

Q30_SCALE = 1 << 30


def bin_features_for_packboost(features: pd.DataFrame) -> np.ndarray:
    arr = features.to_numpy(copy=False, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0)
    rounded = np.rint(arr)
    if (
        arr.size > 0
        and arr.min() >= 0.0
        and arr.max() <= 4.0
        and np.allclose(arr, rounded, atol=1e-6)
    ):
        return np.asarray(np.clip(rounded, 0, 4), dtype=np.int8)

    n_rows = arr.shape[0]
    if n_rows == 0:
        return np.empty((0, arr.shape[1]), dtype=np.int8)

    out = np.empty(arr.shape, dtype=np.int8)
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty(n_rows, dtype=np.float64)
        ranks[order] = np.arange(n_rows, dtype=np.float64)
        if n_rows == 1:
            out[:, col_idx] = 2
        else:
            out[:, col_idx] = np.floor(ranks / (n_rows - 1) * 4.0).astype(np.int8)
    return out


def encode_era_ids(era: pd.Series) -> np.ndarray:
    ordered = sorted(era.unique(), key=str)
    mapping = {label: idx for idx, label in enumerate(ordered)}
    return np.asarray(era.map(mapping).astype(np.int32).to_numpy())


def sort_rows_by_era(
    features: pd.DataFrame,
    target: pd.Series,
    era: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    era_codes = encode_era_ids(era)
    order = np.argsort(era_codes, kind="stable")
    idx = features.index[order]
    return (
        features.loc[idx],
        target.loc[idx],
        era.loc[idx],
    )


def default_nfeatsets(n_features: int, requested: int = 32) -> int:
    bitplanes = 4 * n_features
    max_sets = max(1, bitplanes // 32)
    return min(requested, max_sets)


def q30_predictions_to_float(predictions_q30: np.ndarray) -> np.ndarray:
    return predictions_q30.astype(np.float64) / float(Q30_SCALE)
