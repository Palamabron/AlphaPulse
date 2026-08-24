"""Data loading utilities for Numerai EDA — delegates to alphapulse.data."""

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from alphapulse.data import NumeraiDataLoader

from .config import DATA_DIR


@st.cache_data
def load_feature_metadata(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    import json

    features_path = data_dir / "features.json"
    if not features_path.exists():
        return {}
    with open(features_path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


@st.cache_data
def load_numerai_data(
    data_dir: str | Path = DATA_DIR,
    feature_set_name: str = "medium",
    subsample_eras: bool = True,
    selected_target: str = "target",
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Load Numerai training data via ``NumeraiDataLoader``.

    Returns (dataframe, feature_list, messages_dict).
    """
    messages: dict[str, list[str]] = {"info": [], "warning": [], "error": []}
    data_path = Path(data_dir)

    loader = NumeraiDataLoader(
        data_path,
        feature_set=feature_set_name,
        target_col=selected_target,
    )
    ds = loader.load_split("train")

    feature_set = ds.feature_columns
    messages["info"].append(
        f"✅ Loading feature set: {feature_set_name} ({len(feature_set)} features)"
    )

    all_target_cols = [c for c in ds.df.columns if c.startswith("target")]
    keep_cols = ["era"] + all_target_cols + feature_set
    keep_cols = [c for c in keep_cols if c in ds.df.columns]
    train = ds.df[list(dict.fromkeys(keep_cols))].copy()

    if subsample_eras and "era" in train.columns:
        unique_eras = train["era"].unique()
        sampled_eras = unique_eras[::4]
        train = train[train["era"].isin(sampled_eras)]
        messages["info"].append(
            f"📅 Subsampling: {len(sampled_eras)} of {len(unique_eras)} eras"
        )

    return train, feature_set, messages


def get_era_statistics(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    era_stats = (
        df.groupby("era")
        .agg({target_col: ["mean", "std", "min", "max"], "era": "size"})
        .reset_index()
    )
    era_stats.columns = [
        "era",
        "target_mean",
        "target_std",
        "target_min",
        "target_max",
        "count",
    ]
    return era_stats


def get_feature_correlations(
    df: pd.DataFrame, features: list[str], target: str = "target"
) -> pd.DataFrame:
    correlations = []
    for feature in features:
        corr = df[feature].corr(df[target])
        if pd.notna(corr):
            correlations.append(
                {
                    "Feature": feature,
                    "Correlation": float(corr),
                    "Abs_Correlation": abs(float(corr)),
                }
            )
    return pd.DataFrame(correlations).sort_values("Abs_Correlation", ascending=False)
