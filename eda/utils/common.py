"""Common utilities for EDA pages to reduce code duplication."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def compute_feature_target_correlations(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, method: str = "pearson"
) -> pd.DataFrame:
    """Compute correlations between features and target.

    Args:
        df: DataFrame containing features and target
        feature_cols: List of feature column names
        target_col: Target column name
        method: Correlation method ("pearson" or "spearman")

    Returns:
        DataFrame with columns: feature, correlation, abs_correlation
    """
    correlations = []

    for feat in feature_cols:
        if feat in df.columns and target_col in df.columns:
            corr = df[[feat, target_col]].corr(method=method).iloc[0, 1]
            correlations.append(
                {"feature": feat, "correlation": corr, "abs_correlation": abs(corr)}
            )

    if not correlations:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation"])

    return pd.DataFrame(correlations).sort_values("abs_correlation", ascending=False)


def compute_era_statistics(
    df: pd.DataFrame, era_col: str, value_col: str
) -> pd.DataFrame:
    """Compute per-era statistics for a value column.

    Args:
        df: DataFrame containing era and value columns
        era_col: Era column name (typically "era")
        value_col: Value column to compute statistics on

    Returns:
        DataFrame with columns: era, mean, std, min, max, count
    """
    if era_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=["era", "mean", "std", "min", "max", "count"])

    stats = (
        df.groupby(era_col)
        .agg({value_col: ["mean", "std", "min", "max"], era_col: "size"})
        .reset_index()
    )

    stats.columns = ["era", "mean", "std", "min", "max", "count"]

    return stats


def create_download_button(
    df: pd.DataFrame,
    filename: str,
    label: str | None = None,
    key: str | None = None,
    lang: str = "English",
) -> None:
    """Create a CSV download button for a DataFrame with translation support.

    Args:
        df: DataFrame to export
        filename: Output filename (e.g., "correlations.csv")
        label: Button label (auto-translated if None)
        key: Unique key for the button widget
        lang: Language ("English" or "Polski")
    """
    if df.empty:
        return

    csv = df.to_csv(index=False)

    if label is None:
        label = "Download CSV" if lang == "English" else "Pobierz CSV"

    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def get_plotly_theme() -> dict[str, str]:
    """Return consistent color scheme for Plotly charts.

    Returns:
        Dict with color palette for charts
    """
    return {
        "primary": "#1f77b4",  # Blue
        "secondary": "#ff7f0e",  # Orange
        "success": "#2ca02c",  # Green
        "danger": "#d62728",  # Red
        "warning": "#ffbb78",  # Light orange
        "info": "#9467bd",  # Purple
        "neutral": "#7f7f7f",  # Gray
    }


def apply_chart_layout(
    fig: go.Figure,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Apply standardized layout to a Plotly figure.

    Args:
        fig: Plotly figure to update
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Chart height in pixels

    Returns:
        Updated figure with standardized layout
    """
    layout_updates: dict = {
        "template": "plotly_white",
        "hovermode": "closest",
        "showlegend": True,
    }

    if title:
        layout_updates["title"] = {
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        }

    if xaxis_title:
        layout_updates["xaxis_title"] = xaxis_title

    if yaxis_title:
        layout_updates["yaxis_title"] = yaxis_title

    if height:
        layout_updates["height"] = height

    fig.update_layout(**layout_updates)

    return fig
