from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class PurgedEraCV:
    """Era-aware cross-validation with purging and embargo for temporal data.

    Splits eras (not rows) into train/test blocks, drops n_purge eras between
    train and test to eliminate look-ahead bias, and optionally drops n_embargo
    eras after the test block.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_purge: int = 2,
        n_embargo: int = 1,
        *,
        max_train_eras: int | None = None,
        min_train_eras: int = 10,
        era_column: str = "era",
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if n_purge < 0:
            raise ValueError(f"n_purge must be >= 0, got {n_purge}")
        if n_embargo < 0:
            raise ValueError(f"n_embargo must be >= 0, got {n_embargo}")
        self.n_splits = n_splits
        self.n_purge = n_purge
        self.n_embargo = n_embargo
        self.max_train_eras = max_train_eras
        self.min_train_eras = min_train_eras
        self.era_column = era_column

    def get_n_splits(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        groups: pd.Series | None = None,
    ) -> int:
        return self.n_splits

    def _resolve_groups(self, X: pd.DataFrame, groups: pd.Series | None) -> pd.Series:
        if groups is not None:
            return groups
        if self.era_column in X.columns:
            return X[self.era_column]
        raise ValueError(
            f"groups not provided and '{self.era_column}' column not found in X. "
            "Pass era labels via the groups parameter."
        )

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups: pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        era_series = self._resolve_groups(X, groups)
        sorted_eras = sorted(era_series.unique(), key=lambda x: str(x))
        n_eras = len(sorted_eras)

        overhead_per_fold = self.n_purge + self.n_embargo
        available = n_eras - overhead_per_fold * self.n_splits
        if available < self.n_splits + self.min_train_eras:
            min_eras_needed = (
                self.n_splits + overhead_per_fold * self.n_splits + self.min_train_eras
            )
            raise ValueError(
                f"Not enough eras ({n_eras}) for {self.n_splits} splits with "
                f"n_purge={self.n_purge}, n_embargo={self.n_embargo}, "
                f"min_train_eras={self.min_train_eras}. "
                f"Need at least {min_eras_needed} eras."
            )

        test_era_count = max(1, (n_eras - self.min_train_eras) // self.n_splits)

        era_to_indices: dict[str, np.ndarray] = {}
        positions = np.arange(len(X))
        for era in sorted_eras:
            mask = era_series.values == era
            era_to_indices[era] = positions[mask]

        folds_yielded = 0
        for fold_i in range(self.n_splits):
            test_start = self.min_train_eras + self.n_purge + fold_i * test_era_count
            test_end = min(test_start + test_era_count, n_eras)
            if test_start >= n_eras:
                break

            test_eras = sorted_eras[test_start:test_end]
            train_end_era_idx = test_start - self.n_purge
            if train_end_era_idx <= 0:
                logger.debug("Fold {}: not enough training eras, skipping", fold_i)
                continue

            if self.max_train_eras is not None:
                train_start_era_idx = max(0, train_end_era_idx - self.max_train_eras)
            else:
                train_start_era_idx = 0

            train_eras = sorted_eras[train_start_era_idx:train_end_era_idx]
            embargo_end = min(test_end + self.n_embargo, n_eras)
            embargo_eras = set(sorted_eras[test_end:embargo_end])
            train_eras = [e for e in train_eras if e not in embargo_eras]

            if len(train_eras) < self.min_train_eras:
                logger.debug(
                    "Fold {}: only {} train eras (need {}), skipping",
                    fold_i,
                    len(train_eras),
                    self.min_train_eras,
                )
                continue

            train_idx = np.concatenate([era_to_indices[e] for e in train_eras])
            test_idx = np.concatenate([era_to_indices[e] for e in test_eras])

            folds_yielded += 1
            yield train_idx, test_idx

        if folds_yielded == 0:
            raise ValueError(
                "No valid folds could be generated. Try reducing n_splits, "
                "n_purge, n_embargo, or min_train_eras."
            )

    def split_eras(
        self, era_series: pd.Series
    ) -> Iterator[tuple[list[str], list[str]]]:
        sorted_eras = sorted(era_series.unique(), key=lambda x: str(x))
        n_eras = len(sorted_eras)
        test_era_count = max(1, (n_eras - self.min_train_eras) // self.n_splits)

        for fold_i in range(self.n_splits):
            test_start = self.min_train_eras + self.n_purge + fold_i * test_era_count
            test_end = min(test_start + test_era_count, n_eras)
            if test_start >= n_eras:
                break

            test_eras = sorted_eras[test_start:test_end]
            train_end_era_idx = test_start - self.n_purge
            if train_end_era_idx <= 0:
                continue

            if self.max_train_eras is not None:
                train_start = max(0, train_end_era_idx - self.max_train_eras)
            else:
                train_start = 0

            train_eras = sorted_eras[train_start:train_end_era_idx]
            embargo_end = min(test_end + self.n_embargo, n_eras)
            embargo_eras = set(sorted_eras[test_end:embargo_end])
            train_eras = [e for e in train_eras if e not in embargo_eras]

            if len(train_eras) < self.min_train_eras:
                continue

            yield train_eras, test_eras

    def summary(self, era_series: pd.Series) -> list[dict[str, Any]]:
        result = []
        for i, (train_eras, test_eras) in enumerate(self.split_eras(era_series)):
            result.append(
                {
                    "fold": i,
                    "n_train_eras": len(train_eras),
                    "n_test_eras": len(test_eras),
                    "train_range": (train_eras[0], train_eras[-1]),
                    "test_range": (test_eras[0], test_eras[-1]),
                    "purge_gap": self.n_purge,
                    "embargo": self.n_embargo,
                }
            )
        return result

    def __repr__(self) -> str:
        return (
            f"PurgedEraCV(n_splits={self.n_splits}, n_purge={self.n_purge}, "
            f"n_embargo={self.n_embargo}, max_train_eras={self.max_train_eras}, "
            f"min_train_eras={self.min_train_eras})"
        )
