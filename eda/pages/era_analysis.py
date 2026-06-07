"""Era Analysis Page — Temporal analysis of features and target across eras."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from eda.utils import get_translations

st.set_page_config(page_title="Era Analysis", page_icon="⏰", layout="wide")

t = get_translations()

if "data_loaded" not in st.session_state:
    st.warning(t["errors.data_not_loaded"])
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]
all_targets = st.session_state.get("all_targets", [])

st.title(t["era_analysis.title"])
st.markdown(t["era_analysis.description"])

st.sidebar.header(t["era_analysis.sidebar_header"])

if len(all_targets) == 0:
    all_targets = [col for col in train.columns if col.startswith("target")]

if len(all_targets) == 0:
    all_targets = ["target"]

selected_target = st.sidebar.selectbox(
    t["era_analysis.target_select"],
    all_targets,
    index=0,
    help=t["era_analysis.target_select_help"],
)

st.info(t.format("era_analysis.target_info", target=selected_target))

st.divider()

st.header(t["era_analysis.era_overview_header"])

era_stats = (
    train.groupby("era")
    .agg({selected_target: ["mean", "std", "min", "max"], "era": "size"})
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

era_count_col, avg_obs_col, min_obs_col, max_obs_col = st.columns(4)

with era_count_col:
    st.metric(t["era_analysis.num_eras"], train["era"].nunique())

with avg_obs_col:
    st.metric(t["era_analysis.avg_obs_per_era"], f"{era_stats['count'].mean():.0f}")

with min_obs_col:
    st.metric(t["era_analysis.min_obs_per_era"], f"{era_stats['count'].min()}")

with max_obs_col:
    st.metric(t["era_analysis.max_obs_per_era"], f"{era_stats['count'].max()}")

st.divider()

st.header(t["era_analysis.obs_per_era_header"])

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=era_stats["era"],
        y=era_stats["count"],
        marker={
            "color": era_stats["count"],
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": t["era_analysis.obs_count"]},
        },
        text=era_stats["count"],
        textposition="outside",
        hovertemplate=t["era_analysis.obs_per_era_hover"],
    )
)

fig.update_layout(
    title=t["era_analysis.obs_per_era_title"],
    xaxis_title=t["common.era"],
    yaxis_title=t["era_analysis.obs_per_era_yaxis"],
    height=500,
    showlegend=False,
)

fig.update_xaxes(tickangle=45)
st.plotly_chart(fig, width="stretch")

era_count_col, avg_obs_col, min_obs_col = st.columns(3)

with era_count_col:
    cv = (era_stats["count"].std() / era_stats["count"].mean()) * 100
    st.metric(
        t["era_analysis.era_cv"],
        f"{cv:.2f}%",
        help=t["era_analysis.era_cv_help"],
    )

with avg_obs_col:
    st.metric(
        t["era_analysis.era_size_range"],
        f"{era_stats['count'].max() - era_stats['count'].min()}",
    )

with min_obs_col:
    st.metric(t["era_analysis.median_obs"], f"{era_stats['count'].median():.0f}")

csv_era = era_stats.to_csv(index=False).encode("utf-8")
st.download_button(
    label=t["common.download_csv"],
    data=csv_era,
    file_name=f"era_statistics_{selected_target}.csv",
    mime="text/csv",
    key="download_era_stats",
)

st.divider()

st.header(t.format("era_analysis.behavior_header", target=selected_target.upper()))

tab1, tab2, tab3 = st.tabs(
    [
        t["era_analysis.tab_mean"],
        t["era_analysis.tab_variability"],
        t["era_analysis.tab_range"],
    ]
)

with tab1:
    st.subheader(
        t.format("era_analysis.mean_subheader", target=selected_target.upper())
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=era_stats["era"],
            y=era_stats["target_mean"],
            mode="lines+markers",
            name=t.format("era_analysis.mean_trace_name", target=selected_target),
            line={"color": "blue", "width": 2},
            marker={"size": 6},
            hovertemplate="Era: %{x}<br>Mean: %{y:.6f}",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=era_stats["era"],
            y=era_stats["target_mean"] + era_stats["target_std"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=era_stats["era"],
            y=era_stats["target_mean"] - era_stats["target_std"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(0,100,200,0.2)",
            line={"width": 0},
            name="±1 Std",
            hoverinfo="skip",
        )
    )

    fig.add_hline(
        y=train[selected_target].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=t["era_analysis.global_mean_annotation"],
        annotation_position="right",
    )

    fig.update_layout(
        xaxis_title=t["common.era"],
        yaxis_title=t.format("era_analysis.mean_yaxis", target=selected_target.upper()),
        height=500,
        hovermode="x unified",
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")

    st.markdown(t["era_analysis.stability_formula_header"])

    st.latex(r"""
    \text{Stability} = \sigma(\bar{y}_{\text{era}})
    """)

    st.markdown(t["era_analysis.stability_formula_where"])
    st.markdown(f"- {t['era_analysis.stability_formula_era_mean']}")
    st.markdown(f"- {t['era_analysis.stability_formula_sigma']}")

    era_count_col, avg_obs_col, min_obs_col = st.columns(3)

    with era_count_col:
        mean_stability = era_stats["target_mean"].std()
        st.metric(
            t["era_analysis.stability_metric"],
            f"{mean_stability:.6f}",
            help=t["era_analysis.stability_help"],
        )

    with avg_obs_col:
        trend = np.polyfit(range(len(era_stats)), era_stats["target_mean"], 1)[0]
        st.metric(
            t["era_analysis.trend_metric"],
            f"{trend:.8f}",
            help=t["era_analysis.trend_help"],
        )

    with min_obs_col:
        range_mean = era_stats["target_mean"].max() - era_stats["target_mean"].min()
        st.metric(t["era_analysis.range_of_means"], f"{range_mean:.6f}")

with tab2:
    st.subheader(
        t.format("era_analysis.variability_subheader", target=selected_target.upper())
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=era_stats["era"],
            y=era_stats["target_std"],
            mode="lines+markers",
            line={"color": "red", "width": 2},
            marker={"size": 6},
            fill="tozeroy",
            fillcolor="rgba(255,0,0,0.1)",
            hovertemplate="Era: %{x}<br>Std: %{y:.6f}",
        )
    )

    fig.add_hline(
        y=era_stats["target_std"].mean(),
        line_dash="dash",
        line_color="orange",
        annotation_text=t["era_analysis.avg_variability_annotation"],
    )

    fig.update_layout(
        xaxis_title=t["common.era"],
        yaxis_title=f"Std Dev {selected_target.upper()}",
        height=500,
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")

    st.markdown(t["era_analysis.high_vol_header"])

    high_vol_threshold = era_stats["target_std"].quantile(0.75)
    high_vol_eras = era_stats[era_stats["target_std"] >= high_vol_threshold]

    st.dataframe(
        high_vol_eras[["era", "target_std", "target_mean"]].sort_values(
            "target_std", ascending=False
        ),
        width="stretch",
    )

with tab3:
    st.subheader(
        t.format("era_analysis.range_subheader", target=selected_target.upper())
    )

    era_stats["target_range"] = era_stats["target_max"] - era_stats["target_min"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=era_stats["era"],
            y=era_stats["target_range"],
            mode="lines+markers",
            line={"color": "green", "width": 2},
            marker={"size": 6},
            fill="tozeroy",
            fillcolor="rgba(0,255,0,0.1)",
            hovertemplate="Era: %{x}<br>Range: %{y:.6f}",
        )
    )

    fig.update_layout(
        xaxis_title=t["common.era"],
        yaxis_title=f"{selected_target.upper()} Range (Max - Min)",
        height=500,
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")

st.divider()

st.header(t["era_analysis.feature_time_header"])

selected_features_time = st.multiselect(
    t["era_analysis.feature_time_select"],
    feature_set,
    default=feature_set[:3],
)

if len(selected_features_time) > 5:
    st.warning(t["era_analysis.too_many_features"])
    selected_features_time = selected_features_time[:5]

if len(selected_features_time) > 0:
    feature_era_stats = (
        train.groupby("era")[selected_features_time].agg(["mean", "std"]).reset_index()
    )

    st.subheader(t["era_analysis.feature_mean_subheader"])

    fig = go.Figure()

    for feature in selected_features_time:
        fig.add_trace(
            go.Scatter(
                x=feature_era_stats["era"],
                y=feature_era_stats[feature]["mean"],
                mode="lines+markers",
                name=feature,
                hovertemplate=(f"{feature}<br>Era: %{{x}}<br>Mean: %{{y:.4f}}"),
            )
        )

    fig.update_layout(
        title=t["era_analysis.feature_mean_title"],
        xaxis_title=t["common.era"],
        yaxis_title=t["era_analysis.feature_mean_yaxis"],
        height=500,
        hovermode="x unified",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
        },
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader(
        t.format(
            "era_analysis.corr_stability_subheader", target=selected_target.upper()
        )
    )

    correlation_over_time = []

    for era in sorted(train["era"].unique()):
        era_data = train[train["era"] == era]
        row = {"era": era}

        for feature in selected_features_time:
            corr = era_data[[feature, selected_target]].corr().iloc[0, 1]
            row[feature] = corr

        correlation_over_time.append(row)

    corr_time_df = pd.DataFrame(correlation_over_time)

    fig = go.Figure()

    for feature in selected_features_time:
        fig.add_trace(
            go.Scatter(
                x=corr_time_df["era"],
                y=corr_time_df[feature],
                mode="lines+markers",
                name=feature,
                hovertemplate=(f"{feature}<br>Era: %{{x}}<br>Corr: %{{y:.6f}}"),
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=t.format(
            "era_analysis.corr_stability_title",
            target=selected_target.upper(),
        ),
        xaxis_title=t["common.era"],
        yaxis_title="Correlation",
        height=500,
        hovermode="x unified",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
        },
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")

    st.markdown(t["era_analysis.corr_stability_formula_header"])

    st.latex(r"""
    \text{Stability}_{corr} = \sigma(\rho_{\text{era}})
    """)

    st.markdown(t["era_analysis.corr_stability_where"])
    st.markdown(f"- {t['era_analysis.corr_stability_rho']}")
    st.markdown(f"- {t['era_analysis.corr_stability_sigma']}")

    st.markdown(t["era_analysis.corr_metrics_header"])

    stability_metrics = []

    for feature in selected_features_time:
        corr_values = corr_time_df[feature]

        if isinstance(selected_target, list | tuple):
            if not selected_target:
                continue
            target_col = selected_target[0]
        else:
            target_col = selected_target
        global_corr = train[[feature, target_col]].corr().iloc[0, 1]

        stability_metrics.append(
            {
                "Feature": feature,
                "Global_Correlation": global_corr,
                "Mean_Corr_per_Era": corr_values.mean(),
                "Stability_sigma": corr_values.std(),
                "Min_Corr": corr_values.min(),
                "Max_Corr": corr_values.max(),
                "Sign_Changes": (
                    corr_values[:-1].values * corr_values[1:].values < 0
                ).sum(),
            }
        )

    stability_df = pd.DataFrame(stability_metrics)

    st.dataframe(
        stability_df.style.background_gradient(subset=["Stability_sigma"], cmap="Reds"),
        width="stretch",
    )

    st.info(t["era_analysis.stability_interpretation"])

st.divider()

st.header(t["era_analysis.rolling_header"])

window_size = st.slider(t["era_analysis.rolling_window"], 3, 20, 5)

era_stats["rolling_mean"] = (
    era_stats["target_mean"].rolling(window=window_size, center=True).mean()
)
era_stats["rolling_std"] = (
    era_stats["target_std"].rolling(window=window_size, center=True).mean()
)

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=(
        t.format("era_analysis.mean_subheader", target=selected_target.upper())
        + f" (window={window_size})",
        f"Std Dev {selected_target.upper()} (window={window_size})",
    ),
    shared_xaxes=True,
    vertical_spacing=0.12,
)

fig.add_trace(
    go.Scatter(
        x=era_stats["era"],
        y=era_stats["target_mean"],
        mode="markers",
        name=t["era_analysis.actual"],
        marker={"size": 4, "color": "lightblue"},
        showlegend=True,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=era_stats["era"],
        y=era_stats["rolling_mean"],
        mode="lines",
        name="Rolling Mean",
        line={"color": "blue", "width": 3},
        showlegend=True,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=era_stats["era"],
        y=era_stats["target_std"],
        mode="markers",
        name=t["era_analysis.actual_std"],
        marker={"size": 4, "color": "lightcoral"},
        showlegend=False,
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=era_stats["era"],
        y=era_stats["rolling_std"],
        mode="lines",
        name="Rolling Std",
        line={"color": "red", "width": 3},
        showlegend=True,
    ),
    row=2,
    col=1,
)

fig.update_xaxes(title_text=t["common.era"], row=2, col=1, tickangle=45)
fig.update_yaxes(title_text=f"{selected_target.upper()} Mean", row=1, col=1)
fig.update_yaxes(title_text=f"{selected_target.upper()} Std", row=2, col=1)

fig.update_layout(height=700, hovermode="x unified")
st.plotly_chart(fig, width="stretch")

st.divider()

st.markdown(t.format("era_analysis.footer", target=selected_target))
