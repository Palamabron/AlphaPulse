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
) -> tuple[Any, Any, Any, Any]:
    use_split = force_internal or len(X_train) > INTERNAL_VAL_THRESHOLD
    if not use_split:
        return X_train, y_train, None, None

    if era_train is not None:
        unique_eras = era_train.unique()
        if len(unique_eras) >= MIN_ERAS_FOR_INTERNAL_SPLIT:
            n_val_eras = max(1, int(len(unique_eras) * INTERNAL_VAL_ERA_FRACTION))
            val_eras = set(unique_eras[-n_val_eras:])
            mask = era_train.isin(val_eras)
            return (
                X_train[~mask],
                y_train[~mask],
                X_train[mask],
                y_train[mask],
            )

    n_val = max(1, int(len(X_train) * INTERNAL_VAL_FRACTION))
    if n_val >= len(X_train):
        n_val = max(1, len(X_train) // 10)
    return (
        X_train.iloc[:-n_val],
        y_train.iloc[:-n_val],
        X_train.tail(n_val),
        y_train.tail(n_val),
    )
