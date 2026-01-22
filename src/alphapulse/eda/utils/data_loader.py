"""Data loading utilities for Numerai EDA"""

import json
from typing import Any

import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import streamlit as st
from utils.config import FEATURES_JSON_PATH


@st.cache_data
def load_feature_metadata() -> dict[str, Any]:
    """Load feature metadata"""
    try:
        with open(FEATURES_JSON_PATH, encoding="utf-8") as f:
            feature_sets: dict[str, Any] = json.load(f)
            if not feature_sets:
                return {}
            return feature_sets
    except Exception:
        return {}


@st.cache_data
def load_numerai_data(
    data_path: str,
    feature_set_name: str = "medium",
    subsample_eras: bool = True,
    selected_target: str = "target",
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """
    Load Numerai training data efficiently using PyArrow metadata

    Args:
        data_path: Path to the parquet file
        feature_set_name: Name of feature set (small, medium, all)
        subsample_eras: If True, use every 4th era
        selected_target: Target column name to load

    Returns:
        tuple: (dataframe, feature_list, messages_dict)
    """
    messages: dict[str, list[str]] = {"info": [], "warning": [], "error": []}

    try:
        pf = pq.ParquetFile(data_path)
        all_columns = pf.schema.names

        metadata = load_feature_metadata()

        if metadata and "feature_sets" in metadata:
            feature_sets = metadata["feature_sets"]
            if feature_set_name in feature_sets:
                feature_set = feature_sets[feature_set_name]
                messages["info"].append(
                    f"✅ Loading feature set: {feature_set_name} "
                    f"({len(feature_set)} features)"
                )
            else:
                messages["warning"].append(
                    f"⚠️ Feature set '{feature_set_name}' not found. Using all features."
                )
                feature_set = [
                    col
                    for col in all_columns
                    if col not in ["era", "id"] and col.startswith("feature_")
                ]
        else:
            messages["warning"].append("⚠️ features.json not found. Using all features.")
            feature_set = [
                col
                for col in all_columns
                if col not in ["era", "id"] and col.startswith("feature_")
            ]

        all_targets = [col for col in all_columns if col.startswith("target")]

        columns_to_load = ["era"] + all_targets + feature_set

        train = pf.read(columns=columns_to_load).to_pandas()

        if subsample_eras:
            unique_eras = train["era"].unique()
            sampled_eras = unique_eras[::4]
            train = train[train["era"].isin(sampled_eras)]
            messages["info"].append(
                f"📅 Subsampling: {len(sampled_eras)} of {len(unique_eras)} eras"
            )

        return train, feature_set, messages

    except Exception as e:
        messages["error"].append(f"Error loading data: {e}")
        raise


def get_era_statistics(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """Calculate era-level statistics"""
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
    """Calculate correlations between features and target"""
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
