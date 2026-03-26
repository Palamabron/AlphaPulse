from typing import Self

import pandas as pd

from .base import BasePreprocessor


class GroupedPreprocessor(BasePreprocessor):
    """Applies independent preprocessor chains per feature group, then concatenates."""

    def __init__(
        self,
        groups: dict[str, list[str]],
        group_preprocessors: dict[str, list[BasePreprocessor]],
        column_prefix: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if set(groups.keys()) != set(group_preprocessors.keys()):
            raise ValueError("groups and group_preprocessors must have the same keys")
        self.groups = {k: list(v) for k, v in groups.items()}
        self.group_preprocessors = group_preprocessors
        self.column_prefix = column_prefix

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        for g_name, cols in self.groups.items():
            missing = [c for c in cols if c not in X.columns]
            if missing:
                raise ValueError(f"Group {g_name}: missing columns {missing[:5]}...")
            X_g = X[cols]
            for pp in self.group_preprocessors[g_name]:
                pp.fit(X_g, y)
                X_g = pp.transform(X_g)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted!")
        parts: list[pd.DataFrame] = []
        for g_name, cols in self.groups.items():
            X_g = X[cols].copy()
            for pp in self.group_preprocessors[g_name]:
                X_g = pp.transform(X_g)
            if self.column_prefix:
                X_g.columns = [f"{g_name}__{c}" for c in X_g.columns]
            parts.append(X_g)
        return pd.concat(parts, axis=1)
