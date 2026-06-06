import json
from pathlib import Path

import pandas as pd


def load_feature_names(
    data_dir: Path,
    feature_set: str | None = None,
) -> list[str]:
    """Load feature names from ``features.json``.

    Supports both a flat list and the Numerai v5 dict format
    ``{"feature_sets": {"small": [...], "medium": [...], ...}, ...}``.
    """
    features_path = data_dir / "features.json"
    if not features_path.exists():
        return []
    with open(features_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            return data
        return []

    if isinstance(data, dict):
        raw_feature_sets = data.get("feature_sets")
        if not isinstance(raw_feature_sets, dict):
            return []

        # `json.load()` returns `Any`, so we narrow to a concrete shape.
        feature_sets: dict[str, list[str]] = {}
        for k, v in raw_feature_sets.items():
            if not isinstance(k, str):
                continue
            if not isinstance(v, list):
                continue
            if all(isinstance(x, str) for x in v):
                feature_sets[k] = v

        if feature_set and feature_set in feature_sets:
            return feature_sets[feature_set]
        if "medium" in feature_sets:
            return feature_sets["medium"]
        if feature_sets:
            return next(iter(feature_sets.values()))
    return []


def resolve_feature_columns(
    train_df: pd.DataFrame,
    data_dir: Path,
    explicit: list[str] | None,
    benchmark_columns: list[str] | None = None,
) -> list[str]:
    excluded = set(benchmark_columns or [])
    if explicit:
        return [c for c in explicit if c in train_df.columns and c not in excluded]
    feature_names = load_feature_names(data_dir)
    if feature_names:
        return [c for c in feature_names if c in train_df.columns and c not in excluded]
    meta = {"id", "era", "target"} | excluded
    meta.update(c for c in train_df.columns if c.startswith("target_"))
    return [
        c
        for c in train_df.columns
        if c not in meta and str(train_df[c].dtype) in ("float64", "float32")
    ]


def load_train_val_frames(
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    feature_columns: list[str] | None,
    need_era: bool,
    benchmark_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    if train_subsample <= 0.0:
        raise ValueError(f"train_subsample must be > 0, got {train_subsample}")

    train_path = data_dir / "train.parquet"
    val_path = data_dir / "validation.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {val_path}. "
            "Run scripts/download_dataset.py first."
        )

    excluded = set(benchmark_columns or [])
    feature_names = feature_columns or load_feature_names(data_dir)
    extra_cols = [target_col] + (["era"] if need_era else [])
    if feature_names:
        read_cols = list(dict.fromkeys(feature_names + extra_cols + ["era"]))
        train_df = pd.read_parquet(train_path, columns=read_cols)
        val_df = pd.read_parquet(val_path, columns=read_cols)
        cols = [c for c in feature_names if c in train_df.columns and c not in excluded]
    else:
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        cols = resolve_feature_columns(
            train_df, data_dir, feature_columns, benchmark_columns
        )

    if not cols:
        raise ValueError("No feature columns resolved.")
    cols = [c for c in cols if c not in {"era", "id"}]
    if not cols:
        raise ValueError("No feature columns resolved after excluding metadata.")
    read_cols = cols + (["era"] if need_era else [])
    train_df = train_df.sample(frac=train_subsample, random_state=seed)
    return (
        train_df[read_cols],
        train_df[target_col],
        val_df[read_cols],
        val_df[target_col],
        val_df["era"],
        cols,
    )


def load_train_only_frame(
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    feature_columns: list[str] | None,
    need_era: bool,
    feature_set: str | None = None,
    benchmark_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load only train.parquet (no validation) with column pruning to reduce RAM."""
    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Expected {train_path}")

    excluded = set(benchmark_columns or [])
    feature_names = feature_columns or load_feature_names(
        data_dir, feature_set=feature_set
    )
    if feature_names:
        read_cols = list(
            dict.fromkeys(feature_names + [target_col] + (["era"] if need_era else []))
        )
        train_df = pd.read_parquet(train_path, columns=read_cols)
        cols = [c for c in feature_names if c in train_df.columns and c not in excluded]
    else:
        train_df = pd.read_parquet(train_path)
        cols = resolve_feature_columns(
            train_df, data_dir, feature_columns, benchmark_columns
        )

    if not cols:
        raise ValueError("No feature columns resolved.")
    cols = [c for c in cols if c not in {"era", "id"}]
    if not cols:
        raise ValueError("No feature columns resolved after excluding metadata.")
    read_cols = cols + (["era"] if need_era else [])

    train_df = train_df.sample(frac=train_subsample, random_state=seed)
    return train_df[read_cols], train_df[target_col], cols


def load_train_frame_with_era(
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    feature_columns: list[str] | None,
    need_era: bool,
    benchmark_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Expected {train_path}")

    excluded = set(benchmark_columns or [])
    feature_names = feature_columns or load_feature_names(data_dir)
    if feature_names:
        read_cols = list(dict.fromkeys(feature_names + [target_col, "era"]))
        train_df = pd.read_parquet(train_path, columns=read_cols)
        cols = [c for c in feature_names if c in train_df.columns and c not in excluded]
    else:
        train_df = pd.read_parquet(train_path)
        cols = resolve_feature_columns(
            train_df, data_dir, feature_columns, benchmark_columns
        )

    if not cols:
        raise ValueError("No feature columns resolved.")
    cols = [c for c in cols if c not in {"era", "id"}]
    if not cols:
        raise ValueError("No feature columns resolved after excluding metadata.")
    x_cols = cols + (["era"] if need_era else [])
    train_df = train_df.sample(frac=train_subsample, random_state=seed)
    return train_df[x_cols], train_df[target_col], train_df["era"], cols
