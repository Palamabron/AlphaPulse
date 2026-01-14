"""Data loading utilities for Numerai EDA"""

import json
from typing import Any

import pandas as pd
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
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load Numerai training data with specified feature set and target

    Args:
        data_path: Path to the parquet file
        feature_set_name: Name of feature set (small, medium, all)
        subsample_eras: If True, use every 4th era
        selected_target: Target column name to load

    Returns:
        tuple: (dataframe, feature_list)
    """
    try:
        # Load feature metadata
        metadata = load_feature_metadata()

        # Extract feature_sets from nested JSON structure
        if metadata and "feature_sets" in metadata:
            feature_sets = metadata["feature_sets"]
            if feature_set_name in feature_sets:
                feature_set = feature_sets[feature_set_name]
                st.info(
                    f"✅ Ładuję zestaw cech: {feature_set_name} ({len(feature_set)} cech)"
                )
            else:
                st.warning(
                    f"⚠️ Zestaw '{feature_set_name}' nie znaleziony. Dostępne: {list(feature_sets.keys())}"
                )
                # Use fallback
                feature_set = None
        else:
            st.warning(
                "⚠️ features.json nie zawiera 'feature_sets'. Fallback do auto-detect."
            )
            feature_set = None

        # Fallback if feature_set is still None
        if feature_set is None:
            st.warning("Używam wszystkich dostępnych cech (fallback)")
            temp_df = pd.read_parquet(data_path)
            feature_set = [
                col
                for col in temp_df.columns
                if col not in ["era", "id"] and col.startswith("feature_")
            ]
            del temp_df

        # Load ALL target columns + era + selected features
        # First get all available targets
        all_cols = pd.read_parquet(data_path).columns.tolist()
        all_targets = [col for col in all_cols if col.startswith("target")]

        # Build columns to load
        columns_to_load = ["era"] + all_targets + feature_set

        # Load data
        train = pd.read_parquet(data_path, columns=columns_to_load)

        # Subsample eras if requested
        if subsample_eras:
            unique_eras = train["era"].unique()
            sampled_eras = unique_eras[::4]
            train = train[train["era"].isin(sampled_eras)]
            st.info(f"📅 Subsampling: {len(sampled_eras)} z {len(unique_eras)} er")

        return train, feature_set

    except Exception as e:
        st.error(f"Błąd podczas ładowania danych: {e}")
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
        if pd.notna(corr):  # Sprawdź czy nie NaN
            correlations.append(
                {
                    "Cecha": feature,
                    "Korelacja": float(corr),
                    "Abs_Korelacja": abs(float(corr)),
                }
            )
    return pd.DataFrame(correlations).sort_values("Abs_Korelacja", ascending=False)
