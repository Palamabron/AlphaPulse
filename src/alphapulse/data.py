from __future__ import annotations

import json
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
        auxiliary_target_cols: Optional list of auxiliary target column names.
        feature_groups_map: Optional mapping of group name → feature column list
            from Numerai's native feature group metadata.
    """

    df: pd.DataFrame
    feature_columns: list[str]
    target_col: str = "target"
    split: str = ""
    auxiliary_target_cols: list[str] | None = None
    feature_groups_map: dict[str, list[str]] | None = None

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

    @property
    def y_aux(self) -> pd.DataFrame | None:
        if not self.auxiliary_target_cols:
            return None
        cols = [c for c in self.auxiliary_target_cols if c in self.df.columns]
        return self.df[cols] if cols else None


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

    def load_feature_groups(self) -> dict[str, list[str]]:
        """Load Numerai's native feature group mappings from ``features.json``.

        Numerai's ``features.json`` (v5+) contains a ``"feature_stats"`` or
        ``"features"`` section with per-feature metadata including group labels.
        This method extracts group → feature-list mappings.

        Returns:
            Dict mapping group names to lists of feature column names.
            Returns an empty dict if the file does not exist or has no group info.
        """
        features_path = self.data_dir / "features.json"
        if not features_path.exists():
            return {}
        with open(features_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}

        # v5 format: {"feature_stats": {"feature_xxx": {"group": "intelligence", ...}}}
        feature_stats = data.get("feature_stats")
        if isinstance(feature_stats, dict):
            groups: dict[str, list[str]] = {}
            for feat_name, feat_meta in feature_stats.items():
                if not isinstance(feat_meta, dict):
                    continue
                group = feat_meta.get("group")
                if isinstance(group, str):
                    groups.setdefault(group, []).append(feat_name)
            if groups:
                return groups

        # Fallback: look for a top-level "groups" key
        groups_data = data.get("groups")
        if isinstance(groups_data, dict):
            result: dict[str, list[str]] = {}
            for g, v in groups_data.items():
                if (
                    isinstance(g, str)
                    and isinstance(v, list)
                    and all(isinstance(x, str) for x in v)
                ):
                    result[g] = v
            return result

        return {}

    def load_split(
        self,
        split: str,
        *,
        subsample: float = 1.0,
        seed: int = 42,
        auxiliary_targets: list[str] | None = None,
    ) -> NumeraiDataset:
        """Load a single parquet split and resolve feature columns.

        Args:
            split: Name of the split file without extension
                (``"train"``, ``"validation"``, ``"live"``, or any custom name).
            subsample: Fraction of rows to sample (1.0 = all rows).
            seed: Random seed used when *subsample* < 1.0.
            auxiliary_targets: Optional list of additional target column names
                to load alongside the primary target (e.g.
                ``["target_cyrus_v4_20", "target_nomi_v4_20"]``).
                Columns that are missing from the file are silently dropped.

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

        extra_cols = [self.target_col, "era", "id"]
        if auxiliary_targets:
            extra_cols = list(dict.fromkeys(extra_cols + auxiliary_targets))

        if feature_names:
            read_cols: list[str] | None = list(
                dict.fromkeys(feature_names + extra_cols)
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

        aux_cols_present = (
            [c for c in auxiliary_targets if c in df.columns]
            if auxiliary_targets
            else None
        )

        return NumeraiDataset(
            df=df,
            feature_columns=feature_cols,
            target_col=self.target_col,
            split=split,
            auxiliary_target_cols=aux_cols_present,
        )
