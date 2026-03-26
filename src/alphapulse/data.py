from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .experiments.data import load_feature_names, resolve_feature_columns


@dataclass
class NumeraiDataset:
    """Loaded split of a Numerai dataset with resolved feature metadata.

    Attributes:
        df: The raw DataFrame for the split (all columns).
        feature_columns: Ordered list of feature column names.
        target_col: Name of the primary target column.
        split: Which split this represents (e.g. ``"train"``, ``"validation"``).
    """

    df: pd.DataFrame
    feature_columns: list[str]
    target_col: str = "target"
    split: str = ""

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def X(self) -> pd.DataFrame:
        return self.df[self.feature_columns]

    @property
    def y(self) -> pd.Series:
        return self.df[self.target_col]

    @property
    def era(self) -> pd.Series | None:
        if "era" in self.df.columns:
            return self.df["era"]
        return None


class NumeraiDataLoader:
    """Loads Numerai parquet splits from a versioned data directory.

    Args:
        data_dir: Path to the directory containing parquet files and
            ``features.json`` (e.g. ``data/v5.2/``).
        feature_set: Optional name of the feature set to use from
            ``features.json`` (e.g. ``"small"``, ``"medium"``, ``"all"``).
            When *None*, falls back to ``"medium"`` then the first available set.
        target_col: Name of the target column. Defaults to ``"target"``.

    Example:
        >>> loader = NumeraiDataLoader(Path("data/v5.2"), feature_set="medium")
        >>> ds = loader.load_split("train")
        >>> ds.X.shape, ds.n_features
    """

    KNOWN_SPLITS = ("train", "validation", "live")

    def __init__(
        self,
        data_dir: str | Path,
        *,
        feature_set: str | None = None,
        target_col: str = "target",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.feature_set = feature_set
        self.target_col = target_col

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")

    def load_split(
        self,
        split: str,
        *,
        subsample: float = 1.0,
        seed: int = 42,
    ) -> NumeraiDataset:
        """Load a single parquet split and resolve feature columns.

        Args:
            split: Name of the split file without extension
                (``"train"``, ``"validation"``, ``"live"``, or any custom name).
            subsample: Fraction of rows to sample (1.0 = all rows).
            seed: Random seed used when *subsample* < 1.0.

        Returns:
            A ``NumeraiDataset`` with the loaded DataFrame and resolved
            feature column metadata.

        Raises:
            FileNotFoundError: If the parquet file for *split* does not exist.
        """
        path = self.data_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}. Run scripts/download_dataset.py first."
            )

        feature_names = load_feature_names(self.data_dir, feature_set=self.feature_set)

        if feature_names:
            read_cols: list[str] | None = list(
                dict.fromkeys(feature_names + [self.target_col, "era", "id"])
            )
        else:
            read_cols = None

        df = pd.read_parquet(path, columns=read_cols)

        if feature_names:
            feature_cols = [c for c in feature_names if c in df.columns]
        else:
            feature_cols = resolve_feature_columns(df, self.data_dir, None)

        if not feature_cols:
            raise ValueError(
                f"No feature columns resolved for split '{split}' in {self.data_dir}"
            )

        if 0.0 < subsample < 1.0:
            df = df.sample(frac=subsample, random_state=seed)

        return NumeraiDataset(
            df=df,
            feature_columns=feature_cols,
            target_col=self.target_col,
            split=split,
        )
