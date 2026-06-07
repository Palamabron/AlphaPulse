"""Target Analysis Page — Comprehensive target variable exploration."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from eda.utils import get_translations

st.set_page_config(page_title="Target Analysis", page_icon="🎯", layout="wide")

t = get_translations()

if "data_loaded" not in st.session_state:
    st.warning(t["errors.data_not_loaded"])
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]
all_targets = st.session_state.get("all_targets", [])

st.title(t["target_analysis.title"])
st.markdown(t["target_analysis.description"])

st.sidebar.header(t["target_analysis.sidebar_header"])

if len(all_targets) == 0:
    all_targets = [col for col in train.columns if col.startswith("target")]

if len(all_targets) == 0:
    all_targets = ["target"]

selected_target = st.sidebar.selectbox(
    t["target_analysis.target_select"],
    all_targets,
    index=0,
    help=t["target_analysis.target_select_help"],
)

if selected_target not in train.columns:
    st.error(t.format("target_analysis.target_not_found", target=selected_target))
    st.stop()

st.info(t.format("target_analysis.target_info", target=selected_target))

st.divider()

st.header(t["target_analysis.stats_header"])

target_stats = train[selected_target].describe()
skewness = train[selected_target].skew()
kurtosis = train[selected_target].kurtosis()

col_mean_std, col_median_iqr, col_min_max, col_q1_q3, col_skew_kurtosis = st.columns(5)

with col_mean_std:
    st.metric(t["target_analysis.mean"], f"{target_stats['mean']:.6f}")
    st.metric(t["target_analysis.std_dev"], f"{target_stats['std']:.6f}")

with col_median_iqr:
    st.metric(t["target_analysis.median"], f"{target_stats['50%']:.6f}")
    st.metric(
        t["target_analysis.iqr"], f"{(target_stats['75%'] - target_stats['25%']):.6f}"
    )

with col_min_max:
    st.metric(t["target_analysis.min"], f"{target_stats['min']:.6f}")
    st.metric(t["target_analysis.max"], f"{target_stats['max']:.6f}")

with col_q1_q3:
    st.metric(t["target_analysis.q25"], f"{target_stats['25%']:.6f}")
    st.metric(t["target_analysis.q75"], f"{target_stats['75%']:.6f}")

with col_skew_kurtosis:
    st.metric(t["target_analysis.skewness"], f"{skewness:.6f}")
    st.metric(t["target_analysis.kurtosis"], f"{kurtosis:.6f}")

summary_stats = pd.DataFrame(
    {
        "metric": [
            "mean",
            "std",
            "median",
            "iqr",
            "min",
            "max",
            "q25",
            "q75",
            "skewness",
            "kurtosis",
        ],
        "value": [
            target_stats["mean"],
            target_stats["std"],
            target_stats["50%"],
            target_stats["75%"] - target_stats["25%"],
            target_stats["min"],
            target_stats["max"],
            target_stats["25%"],
            target_stats["75%"],
            skewness,
            kurtosis,
        ],
    }
)

csv_summary = summary_stats.to_csv(index=False).encode("utf-8")
st.download_button(
    label=t["common.download_csv"],
    data=csv_summary,
    file_name=f"target_summary_{selected_target}.csv",
    mime="text/csv",
    key="download_target_summary",
)

st.divider()

st.header(t["target_analysis.ridgeplot_header"])

st.markdown(t["target_analysis.ridgeplot_description"])

era_sample_rate = st.slider(
    t["target_analysis.ridgeplot_era_sample"], 1, 10, 3, key="ridgeplot_sample"
)

sample_eras = sorted(train["era"].unique())[::era_sample_rate]

ridgeplot_data = []

for era in sample_eras:
    era_data = train[train["era"] == era][selected_target].dropna()
    ridgeplot_data.append({"era": era, "values": era_data.values})

fig = go.Figure()

for idx, era_dict in enumerate(ridgeplot_data):
    era = era_dict["era"]
    values = era_dict["values"]
    fig.add_trace(
        go.Violin(
            x=values,
            y=[era] * len(values),
            name=str(era),
            orientation="h",
            side="positive",
            width=2,
            points=False,
            line_color="rgba(31, 119, 180, 0.8)",
            fillcolor=f"rgba(31, 119, 180, {0.3 + idx * 0.02})",
            opacity=0.7,
            showlegend=False,
        )
    )

fig.update_layout(
    title=t.format(
        "target_analysis.ridgeplot_title",
        target=selected_target,
        n=era_sample_rate,
    ),
    xaxis_title=t.format("target_analysis.ridgeplot_xaxis", target=selected_target),
    yaxis_title=t["common.era"],
    height=max(600, len(sample_eras) * 30),
    yaxis={"categoryorder": "category ascending", "tickmode": "linear"},
    hovermode="closest",
)

st.plotly_chart(fig, width="stretch")

st.divider()

st.header(t["target_analysis.scatter_header"])

st.markdown(t["target_analysis.scatter_description"])

show_sample = st.checkbox(t["target_analysis.show_sample"], value=True)

if show_sample:
    plot_data = train.sample(min(10000, len(train)), random_state=42)
else:
    plot_data = train

scatter_title = (
    t.format("target_analysis.scatter_title_sample", target=selected_target)
    if show_sample
    else t.format("target_analysis.scatter_title_full", target=selected_target)
)

fig = px.scatter(
    plot_data,
    x="era",
    y=selected_target,
    title=scatter_title,
    labels={
        "era": t["common.era"],
        selected_target: t.format(
            "target_analysis.scatter_target_label", target=selected_target
        ),
    },
    opacity=0.3,
    color=selected_target,
    color_continuous_scale="RdBu_r",
)

era_stats = train.groupby("era")[selected_target].mean().reset_index()

fig.add_trace(
    go.Scatter(
        x=era_stats["era"],
        y=era_stats[selected_target],
        mode="lines",
        name=t["target_analysis.mean_per_era"],
        line={"color": "red", "width": 2},
    )
)

fig.add_hline(
    y=train[selected_target].mean(),
    line_dash="dash",
    line_color="green",
    annotation_text=t["target_analysis.global_mean_annotation"],
)

fig.update_layout(
    xaxis_title=t["common.era"],
    yaxis_title=t.format("target_analysis.scatter_yaxis", target=selected_target),
    height=600,
    hovermode="closest",
)

fig.update_xaxes(tickangle=45)
st.plotly_chart(fig, width="stretch")

st.divider()

st.header(t["target_analysis.era_analysis_header"])

try:
    era_stats_full = (
        train.groupby("era")
        .agg({selected_target: ["mean", "std", "min", "max"], "era": "size"})
        .reset_index()
    )
    era_stats_full.columns = [
        "era",
        "target_mean",
        "target_std",
        "target_min",
        "target_max",
        "count",
    ]
except Exception as e:
    st.error(t.format("target_analysis.stats_error", error=e))
    st.stop()

col_mean_std, col_median_iqr = st.columns(2)

with col_mean_std:
    st.subheader(
        t.format("target_analysis.mean_per_era_subheader", target=selected_target)
    )
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=era_stats_full["era"],
            y=era_stats_full["target_mean"],
            mode="lines+markers",
            name=t["target_analysis.mean_trace"],
            line={"color": "blue", "width": 2},
            marker={"size": 6},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=era_stats_full["era"],
            y=era_stats_full["target_mean"] + era_stats_full["target_std"],
            mode="lines",
            name="Mean + Std",
            line={"width": 0},
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=era_stats_full["era"],
            y=era_stats_full["target_mean"] - era_stats_full["target_std"],
            mode="lines",
            name="Mean - Std",
            fill="tonexty",
            fillcolor="rgba(0,100,200,0.2)",
            line={"width": 0},
            showlegend=False,
        )
    )

    fig.add_hline(
        y=train[selected_target].mean(),
        line_dash="dash",
        annotation_text=t["target_analysis.global_mean_annotation"],
        line_color="red",
    )

    fig.update_layout(
        xaxis_title=t["common.era"],
        yaxis_title=t.format(
            "target_analysis.mean_per_era_subheader", target=selected_target
        ),
        height=400,
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")

with col_median_iqr:
    st.subheader(
        t.format("target_analysis.variability_subheader", target=selected_target)
    )
    fig = px.line(
        era_stats_full,
        x="era",
        y="target_std",
        title=t.format("target_analysis.std_title", target=selected_target),
        labels={"era": t["common.era"], "target_std": "Std Dev"},
        markers=True,
    )

    fig.add_hline(
        y=era_stats_full["target_std"].mean(),
        line_dash="dash",
        annotation_text=t["target_analysis.avg_variability_annotation"],
        line_color="orange",
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400)
    fig.update_yaxes(title_text=f"Std Dev {selected_target}")
    st.plotly_chart(fig, width="stretch")

csv_era = era_stats_full.to_csv(index=False).encode("utf-8")
st.download_button(
    label=t["common.download_csv"],
    data=csv_era,
    file_name=f"target_era_stats_{selected_target}.csv",
    mime="text/csv",
    key="download_target_era_stats",
)

st.subheader(t["target_analysis.comprehensive_subheader"])

fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=(
        t.format("target_analysis.mean_subplot", target=selected_target),
        t["target_analysis.std_subplot"],
        t["target_analysis.range_subplot"],
    ),
    vertical_spacing=0.1,
    shared_xaxes=True,
)

fig.add_trace(
    go.Scatter(
        x=era_stats_full["era"],
        y=era_stats_full["target_mean"],
        mode="lines+markers",
        name="Mean",
        line={"color": "blue"},
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=era_stats_full["era"],
        y=era_stats_full["target_std"],
        mode="lines+markers",
        name="Std",
        line={"color": "red"},
    ),
    row=2,
    col=1,
)

era_stats_full["range"] = era_stats_full["target_max"] - era_stats_full["target_min"]
fig.add_trace(
    go.Scatter(
        x=era_stats_full["era"],
        y=era_stats_full["range"],
        mode="lines+markers",
        name="Range",
        line={"color": "green"},
    ),
    row=3,
    col=1,
)

fig.update_xaxes(title_text=t["common.era"], row=3, col=1, tickangle=45)
fig.update_yaxes(title_text="Mean", row=1, col=1)
fig.update_yaxes(title_text="Std Dev", row=2, col=1)
fig.update_yaxes(title_text="Range", row=3, col=1)
fig.update_layout(height=900, showlegend=False)
st.plotly_chart(fig, width="stretch")

st.divider()

st.header(t["target_analysis.stability_header"])

col_mean_std, col_median_iqr, col_min_max = st.columns(3)

with col_mean_std:
    cv = (
        era_stats_full["target_mean"].std() / era_stats_full["target_mean"].mean()
    ) * 100
    st.metric(
        t["target_analysis.cv_metric"],
        f"{cv:.2f}%",
        help=t["target_analysis.cv_help"],
    )

with col_median_iqr:
    mean_std = era_stats_full["target_std"].mean()
    st.metric(
        t["target_analysis.avg_variability_metric"],
        f"{mean_std:.6f}",
        help=t["target_analysis.avg_variability_help"],
    )

with col_min_max:
    range_variability = era_stats_full["range"].std()
    st.metric(
        t["target_analysis.range_variability_metric"],
        f"{range_variability:.6f}",
        help=t["target_analysis.range_variability_help"],
    )

st.subheader(t["target_analysis.stability_map_header"])

window = 5
era_stats_full["rolling_mean"] = (
    era_stats_full["target_mean"].rolling(window=window).mean()
)
era_stats_full["rolling_std"] = (
    era_stats_full["target_std"].rolling(window=window).mean()
)

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=(
        t.format("target_analysis.mean_subplot", target=selected_target)
        + f" (rolling window={window})",
        f"Std Dev {selected_target} (rolling window={window})",
    ),
    shared_xaxes=True,
    vertical_spacing=0.15,
)

fig.add_trace(
    go.Scatter(
        x=era_stats_full["era"],
        y=era_stats_full["target_mean"],
        mode="markers",
        name=t["target_analysis.actual_mean"],
        marker={"size": 4, "color": "lightblue"},
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=era_stats_full["era"],
        y=era_stats_full["rolling_mean"],
        mode="lines",
        name=t["target_analysis.rolling_mean"],
        line={"color": "blue", "width": 3},
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=era_stats_full["era"],
        y=era_stats_full["target_std"],
        mode="markers",
        name=t["target_analysis.actual_std"],
        marker={"size": 4, "color": "lightcoral"},
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=era_stats_full["era"],
        y=era_stats_full["rolling_std"],
        mode="lines",
        name=t["target_analysis.rolling_std"],
        line={"color": "red", "width": 3},
    ),
    row=2,
    col=1,
)

fig.update_xaxes(title_text=t["common.era"], row=2, col=1, tickangle=45)
fig.update_yaxes(title_text=f"{selected_target} Mean", row=1, col=1)
fig.update_yaxes(title_text=f"{selected_target} Std", row=2, col=1)
fig.update_layout(height=700, showlegend=True)
st.plotly_chart(fig, width="stretch")

st.divider()
