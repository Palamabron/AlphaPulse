from __future__ import annotations

from typing import Any

import pandas as pd

INTERNAL_VAL_THRESHOLD = 5000
INTERNAL_VAL_FRACTION = 0.1
INTERNAL_VAL_ERA_FRACTION = 0.1
MIN_ERAS_FOR_INTERNAL_SPLIT = 2


def internal_val_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    era_train: pd.Series | None = None,
    *,
    force_internal: bool = False,
    purge_eras: int = 0,
) -> tuple[Any, Any, Any, Any]:
    if purge_eras < 0:
        raise ValueError("purge_eras must be >= 0")
    if not y_train.index.equals(X_train.index):
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        if era_train is not None:
            era_train = era_train.reset_index(drop=True)

    use_split = force_internal or len(X_train) > INTERNAL_VAL_THRESHOLD
    if not use_split:
        return X_train, y_train, None, None

    if era_train is not None:
        unique_eras = sorted(era_train.unique(), key=str)
        if len(unique_eras) >= MIN_ERAS_FOR_INTERNAL_SPLIT:
            n_val_eras = max(1, int(len(unique_eras) * INTERNAL_VAL_ERA_FRACTION))
            val_eras = set(unique_eras[-n_val_eras:])
            val_start = len(unique_eras) - n_val_eras
            train_end = max(0, val_start - purge_eras)
            if train_end == 0:
                raise ValueError(
                    "Internal validation split leaves no training eras after purge: "
                    f"eras={len(unique_eras)}, val_eras={n_val_eras}, "
                    f"purge_eras={purge_eras}"
                )
            train_eras = set(unique_eras[:train_end])
            train_mask = era_train.isin(train_eras)
            val_mask = era_train.isin(val_eras)
            return (
                X_train[train_mask],
                y_train[train_mask],
                X_train[val_mask],
                y_train[val_mask],
            )
        return X_train, y_train, None, None

    n_val = max(1, int(len(X_train) * INTERNAL_VAL_FRACTION))
    if n_val >= len(X_train):
        n_val = max(1, len(X_train) // 10)
    sorted_idx = X_train.sort_index().index
    X_sorted = X_train.loc[sorted_idx]
    y_sorted = y_train.loc[sorted_idx]
    return (
        X_sorted.iloc[:-n_val],
        y_sorted.iloc[:-n_val],
        X_sorted.iloc[-n_val:],
        y_sorted.iloc[-n_val:],
    )
