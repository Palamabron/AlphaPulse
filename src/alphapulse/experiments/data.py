import json
from pathlib import Path

import numpy as np
import pandas as pd

META_MODEL_COLUMN_CANDIDATES = (
    "numerai_meta_model",
    "meta_model",
    "prediction",
    "meta",
)


def meta_model_value_column(meta_df: pd.DataFrame) -> str | None:
    value_col = next(
        (c for c in META_MODEL_COLUMN_CANDIDATES if c in meta_df.columns),
        None,
    )
    if value_col is not None:
        return value_col
    numeric_cols = meta_df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        return None
    return str(numeric_cols[0])


def load_meta_model_series(
    data_dir: Path,
    index: pd.Index,
    *,
    meta_model_path: str | None = None,
) -> pd.Series | None:
    path = (
        Path(meta_model_path)
        if meta_model_path
        else Path(data_dir) / "meta_model.parquet"
    )
    if not path.exists():
        return None

    meta_df = pd.read_parquet(path)
    value_col = meta_model_value_column(meta_df)
    if value_col is None:
        return None

    if "id" in meta_df.columns:
        aligned = meta_df.set_index("id")[value_col]
        return aligned.reindex(index)
    if meta_df.index.name == "id" or (
        meta_df.index.dtype == object and not meta_df.index.equals(index)
    ):
        return meta_df[value_col].reindex(index)
    if meta_df.index.equals(index):
        return meta_df[value_col]
    if len(meta_df) == len(index):
        return pd.Series(meta_df[value_col].to_numpy(), index=index)
    return meta_df[value_col].reindex(index)


def load_mmc_validation_frame(
    data_dir: Path,
    *,
    feature_cols: list[str],
    target_col: str,
    train_subsample: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray] | None:
    """Load validation rows with aligned Numerai meta-model predictions for MMC scoring.

    ``meta_model.parquet`` covers validation ids only (not train), so MMC during HPO
    must be evaluated on this split rather than on train-era holdout.
    """
    meta_path = data_dir / "meta_model.parquet"
    val_path = data_dir / "validation.parquet"
    if not meta_path.exists() or not val_path.exists():
        return None

    meta_df = pd.read_parquet(meta_path)
    value_col = meta_model_value_column(meta_df)
    if value_col is None:
        return None

    read_cols = list(dict.fromkeys([*feature_cols, target_col, "era"]))
    val_df = pd.read_parquet(val_path, columns=read_cols)
    common_idx = val_df.index.intersection(meta_df.index)
    if common_idx.empty:
        return None

    val_df = val_df.loc[common_idx]
    meta_series = meta_df[value_col].reindex(common_idx)
    valid_meta = meta_series.notna()
    if not valid_meta.any():
        return None
    val_df = val_df.loc[valid_meta]
    meta_series = meta_series.loc[valid_meta]

    if not 0.0 < train_subsample <= 1.0:
        raise ValueError(f"train_subsample must be in (0, 1], got {train_subsample}")
    if train_subsample < 1.0:
        val_df = val_df.sample(frac=train_subsample, random_state=seed)
        meta_series = meta_series.loc[val_df.index]

    X_val = val_df[feature_cols + (["era"] if "era" in val_df.columns else [])]
    return (
        X_val,
        val_df[target_col],
        val_df["era"],
        meta_series.to_numpy(dtype=np.float64),
    )


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


def load_validation_frames(
    data_dir: Path,
    target_col: str,
    feature_cols: list[str],
    *,
    need_era: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    val_path = Path(data_dir) / "validation.parquet"
    if not val_path.exists():
        raise FileNotFoundError(
            f"Expected {val_path}. Run scripts/download_dataset.py first."
        )

    read_cols = list(dict.fromkeys(feature_cols + [target_col, "era"]))
    val_df = pd.read_parquet(val_path, columns=read_cols)
    x_cols = feature_cols + (["era"] if need_era else [])
    return val_df[x_cols], val_df[target_col], val_df["era"]


def load_train_val_frames(
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    feature_columns: list[str] | None,
    need_era: bool,
    benchmark_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    X_train, y_train, feature_cols = load_train_only_frame(
        data_dir,
        train_subsample=train_subsample,
        target_col=target_col,
        seed=seed,
        feature_columns=feature_columns,
        need_era=need_era,
        benchmark_columns=benchmark_columns,
    )
    X_val, y_val, era_val = load_validation_frames(
        data_dir, target_col, feature_cols, need_era=need_era
    )
    return X_train, y_train, X_val, y_val, era_val, feature_cols


def load_train_only_frame(
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    feature_columns: list[str] | None,
    need_era: bool,
    feature_set: str | None = None,
    benchmark_columns: list[str] | None = None,
    extra_target_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load only train.parquet (no validation) with column pruning to reduce RAM."""
    if not 0.0 < train_subsample <= 1.0:
        raise ValueError(f"train_subsample must be in (0, 1], got {train_subsample}")

    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Expected {train_path}")

    excluded = set(benchmark_columns or [])
    extra_targets = list(dict.fromkeys(extra_target_columns or []))
    feature_names = feature_columns or load_feature_names(
        data_dir, feature_set=feature_set
    )
    if feature_names:
        read_cols = list(
            dict.fromkeys(
                feature_names
                + [target_col]
                + extra_targets
                + (["era"] if need_era else [])
            )
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
    if "era" in train_df.columns:
        train_df = train_df.sort_values("era", kind="mergesort")
    return train_df[read_cols], train_df[target_col], cols


def load_train_targets_frame(
    data_dir: Path,
    train_subsample: float,
    primary_target: str,
    auxiliary_targets: list[str] | None,
    seed: int,
    feature_columns: list[str] | None,
    need_era: bool,
    benchmark_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Load train data with feature columns and targets for multi-target HPO."""
    aux = [c for c in dict.fromkeys(auxiliary_targets or []) if c != primary_target]
    target_cols = list(dict.fromkeys([primary_target, *aux]))

    if not 0.0 < train_subsample <= 1.0:
        raise ValueError(f"train_subsample must be in (0, 1], got {train_subsample}")

    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Expected {train_path}")

    excluded = set(benchmark_columns or [])
    feature_names = feature_columns or load_feature_names(data_dir)
    if not feature_names:
        raise ValueError("feature_columns required for multi-target HPO load")

    read_cols = list(
        dict.fromkeys(feature_names + target_cols + (["era"] if need_era else []))
    )
    train_df = pd.read_parquet(train_path, columns=read_cols)
    feature_cols = [
        c for c in feature_names if c in train_df.columns and c not in excluded
    ]
    if not feature_cols:
        raise ValueError("No feature columns resolved.")

    train_df = train_df.sample(frac=train_subsample, random_state=seed)
    if "era" in train_df.columns:
        train_df = train_df.sort_values("era", kind="mergesort")

    x_cols = feature_cols + (["era"] if need_era else [])
    X_train = train_df[x_cols]
    y_primary = train_df[primary_target]
    targets_df = train_df[target_cols].copy()
    return X_train, y_primary, targets_df, feature_cols


def load_train_frame_with_era(
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    feature_columns: list[str] | None,
    need_era: bool,
    benchmark_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    if not 0.0 < train_subsample <= 1.0:
        raise ValueError(f"train_subsample must be in (0, 1], got {train_subsample}")

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
    if "era" in train_df.columns:
        train_df = train_df.sort_values("era", kind="mergesort")
    return train_df[x_cols], train_df[target_col], train_df["era"], cols
