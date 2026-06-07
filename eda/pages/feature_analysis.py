"""Feature Analysis Page — Individual feature exploration."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from eda.utils import get_translations

st.set_page_config(page_title="Feature Analysis", page_icon="🔍", layout="wide")

t = get_translations()

if "data_loaded" not in st.session_state:
    st.warning(t["errors.data_not_loaded"])
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]

discrete_label = t["common.discrete"]
continuous_label = t["common.continuous"]


def is_discrete_feature(series: pd.Series) -> bool:
    unique_vals = series.dropna().unique()
    return set(unique_vals).issubset({0.0, 0.25, 0.5, 0.75, 1.0})


st.title(t["feature_analysis.title"])
st.markdown(t["feature_analysis.description"])

st.sidebar.header(t["feature_analysis.sidebar_header"])

analysis_modes = [
    t["feature_analysis.mode_single"],
    t["feature_analysis.mode_comparison"],
    t["feature_analysis.mode_batch"],
]

analysis_mode = st.sidebar.radio(t["feature_analysis.mode_label"], analysis_modes)

if analysis_mode == analysis_modes[0]:
    selected_feature = st.sidebar.selectbox(
        t["feature_analysis.feature_select"], feature_set, index=0
    )

    st.header(t.format("feature_analysis.feature_header", feature=selected_feature))

    feature_data = train[selected_feature]
    is_discrete = is_discrete_feature(feature_data)

    if is_discrete:
        st.success(t["feature_analysis.discrete_ok"])
    else:
        st.warning(t["feature_analysis.continuous_warning"])

    col_mean, col_median, col_std_dev, col_missing, col_correlation = st.columns(5)

    with col_mean:
        st.metric(t["feature_analysis.mean"], f"{feature_data.mean():.4f}")
    with col_median:
        st.metric(t["feature_analysis.median"], f"{feature_data.median():.4f}")
    with col_std_dev:
        st.metric(t["feature_analysis.std_dev"], f"{feature_data.std():.4f}")
    with col_missing:
        st.metric(t["feature_analysis.missing"], f"{feature_data.isnull().sum()}")
    with col_correlation:
        selected_target = st.session_state.get("selected_target", "target")

        if isinstance(selected_target, list | tuple):
            target_col = selected_target[0]
        else:
            target_col = selected_target

        corr = train[[selected_feature, target_col]].corr().iloc[0, 1]
        st.metric(t["feature_analysis.correlation_with_target"], f"{corr:.6f}")

    st.divider()

    col_mean, col_median = st.columns([2, 1])

    with col_mean:
        st.subheader(t["feature_analysis.value_distribution"])

        if is_discrete:
            value_counts = feature_data.value_counts().sort_index()

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    text=value_counts.values,
                    texttemplate="%{text:,}",
                    textposition="outside",
                    marker={
                        "color": value_counts.values,
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": t["feature_analysis.count_colorbar"]},
                    },
                    hovertemplate=t["feature_analysis.value_hover"],
                )
            )

            fig.update_layout(
                title=t.format("feature_analysis.dist_title", feature=selected_feature),
                xaxis_title=t["feature_analysis.feature_value_xaxis"],
                yaxis_title=t["feature_analysis.count_yaxis"],
                height=400,
            )
        else:
            fig = px.histogram(
                feature_data,
                nbins=50,
                title=t.format("feature_analysis.dist_title", feature=selected_feature),
                labels={selected_feature: t["feature_analysis.feature_value_xaxis"]},
            )
            fig.update_layout(
                xaxis_title=t["feature_analysis.feature_value_xaxis"],
                yaxis_title=t["feature_analysis.count_yaxis"],
                height=400,
            )

        st.plotly_chart(fig, width="stretch")

    with col_median:
        st.subheader(t["feature_analysis.stats_subheader"])

        if is_discrete:
            value_counts = feature_data.value_counts().sort_index()
            value_pct = value_counts / len(feature_data) * 100

            stats_df = pd.DataFrame(
                {
                    "Value": value_counts.index,
                    "Count": value_counts.values,
                    "Percent": value_pct.values.round(2),
                }
            )

            st.dataframe(stats_df, width="stretch", height=400)

            entropy = -(value_pct / 100 * np.log2(value_pct / 100 + 1e-10)).sum()
            st.metric(
                t["feature_analysis.entropy"],
                f"{entropy:.4f}",
                help=t["feature_analysis.entropy_help"],
            )
        else:
            percentiles_df = pd.DataFrame(
                {
                    t["feature_analysis.percentile_col"]: [
                        "Min",
                        "25%",
                        "50%",
                        "75%",
                        "Max",
                    ],
                    "Value": [
                        feature_data.min(),
                        feature_data.quantile(0.25),
                        feature_data.quantile(0.50),
                        feature_data.quantile(0.75),
                        feature_data.max(),
                    ],
                }
            )
            st.dataframe(percentiles_df, width="stretch")
            st.metric(t["feature_analysis.unique_values"], f"{feature_data.nunique()}")

    st.divider()

    st.subheader(t["feature_analysis.target_relation"])

    if is_discrete:
        tab1, tab2, tab3 = st.tabs(
            [
                t["feature_analysis.tab_boxplot"],
                t["feature_analysis.tab_violin"],
                t["feature_analysis.tab_stats_per_value"],
            ]
        )

        with tab1:
            fig = px.box(
                train,
                x=selected_feature,
                y="target",
                title=t.format(
                    "feature_analysis.target_dist_title", feature=selected_feature
                ),
                labels={
                    selected_feature: t["feature_analysis.feature_value_label"],
                    "target": t["feature_analysis.target_label"],
                },
                color=selected_feature,
                color_discrete_sequence=px.colors.sequential.Viridis,
            )
            fig.update_layout(height=500, showlegend=False)
            fig.update_yaxes(title_text=t["feature_analysis.target_yaxis"])
            st.plotly_chart(fig, width="stretch")

        with tab2:
            fig = px.violin(
                train,
                x=selected_feature,
                y="target",
                title=t.format(
                    "feature_analysis.target_dist_title", feature=selected_feature
                ),
                labels={
                    selected_feature: t["feature_analysis.feature_value_label"],
                    "target": t["feature_analysis.target_label"],
                },
                color=selected_feature,
                box=True,
                points="outliers",
            )
            fig.update_layout(height=500, showlegend=False)
            fig.update_yaxes(title_text=t["feature_analysis.target_yaxis"])
            st.plotly_chart(fig, width="stretch")

        with tab3:
            target_by_feature = (
                train.groupby(selected_feature)["target"]
                .agg(["count", "mean", "std", "min", "max"])
                .reset_index()
            )

            target_by_feature.columns = [
                "Value",
                "Count",
                "Mean_Target",
                "Std_Target",
                "Min_Target",
                "Max_Target",
            ]

            st.dataframe(
                target_by_feature.style.background_gradient(
                    subset=["Mean_Target"], cmap="RdYlGn"
                ),
                width="stretch",
            )

            fig = px.bar(
                target_by_feature,
                x="Value",
                y="Mean_Target",
                error_y="Std_Target",
                title=t["feature_analysis.mean_target_per_value"],
                labels={
                    "Mean_Target": t["feature_analysis.mean_target_label"],
                    "Value": t["feature_analysis.feature_value_label"],
                },
                text="Mean_Target",
            )
            fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
            fig.add_hline(
                y=train["target"].mean(),
                line_dash="dash",
                annotation_text=t["feature_analysis.global_mean_annotation"],
                line_color="red",
            )
            fig.update_layout(
                xaxis_title=t["feature_analysis.feature_value_xaxis"],
                yaxis_title=t["feature_analysis.mean_target_yaxis"],
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.info(t["feature_analysis.continuous_info"])

        tab1, tab2 = st.tabs(
            [
                t["feature_analysis.tab_violin"],
                t["feature_analysis.tab_stats_per_bin"],
            ]
        )

        with tab1:
            fig = px.violin(
                train,
                x=selected_feature,
                y="target",
                title=t.format(
                    "feature_analysis.target_dist_title", feature=selected_feature
                ),
                labels={
                    selected_feature: t["feature_analysis.feature_value_label"],
                    "target": t["feature_analysis.target_label"],
                },
                box=True,
                points="outliers",
            )
            fig.update_layout(height=500)
            fig.update_yaxes(title_text=t["feature_analysis.target_yaxis"])
            st.plotly_chart(fig, width="stretch")

        with tab2:
            train["feature_binned"] = pd.qcut(
                train[selected_feature],
                q=4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop",
            )
            target_by_bin = (
                train.groupby("feature_binned")["target"]
                .agg(["count", "mean", "std", "min", "max"])
                .reset_index()
            )

            target_by_bin.columns = [
                "Quartile",
                "Count",
                "Mean_Target",
                "Std_Target",
                "Min_Target",
                "Max_Target",
            ]

            st.dataframe(target_by_bin, width="stretch")

    st.divider()

    st.subheader(t["feature_analysis.era_behavior"])

    feature_era = (
        train.groupby("era")[selected_feature].agg(["mean", "std"]).reset_index()
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            t["feature_analysis.mean_per_era"],
            t["feature_analysis.variability_per_era"],
        ),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=feature_era["era"],
            y=feature_era["mean"],
            mode="lines+markers",
            name="Mean",
            line={"color": "blue"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=feature_era["era"],
            y=feature_era["std"],
            mode="lines+markers",
            name="Std",
            line={"color": "red"},
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=feature_data.mean(),
        line_dash="dash",
        line_color="gray",
        row=1,
        col=1,
        annotation_text=t["feature_analysis.global_mean_label"],
    )

    fig.update_xaxes(title_text=t["common.era"], row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text=t["feature_analysis.mean_feature_yaxis"], row=1, col=1)
    fig.update_yaxes(title_text=t["feature_analysis.std_yaxis"], row=2, col=1)

    fig.update_layout(height=700, showlegend=False)

    st.plotly_chart(fig, width="stretch")

    st.subheader(t["feature_analysis.corr_per_era"])

    era_correlations = []
    for era in train["era"].unique():
        era_data = train[train["era"] == era]
        era_corr = era_data[[selected_feature, "target"]].corr().iloc[0, 1]
        era_correlations.append({"era": era, "correlation": era_corr})

    era_corr_df = pd.DataFrame(era_correlations)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=era_corr_df["era"],
            y=era_corr_df["correlation"],
            mode="lines+markers",
            line={"color": "purple", "width": 2},
            marker={"size": 6},
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(
        y=corr,
        line_dash="dash",
        line_color="red",
        annotation_text=t["feature_analysis.global_corr_annotation"],
    )

    fig.update_layout(
        title=t.format(
            "feature_analysis.corr_stability_title", feature=selected_feature
        ),
        xaxis_title=t["common.era"],
        yaxis_title=t["feature_analysis.corr_yaxis"],
        height=500,
    )
    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, width="stretch")

    col_mean, col_median, col_std_dev = st.columns(3)

    with col_mean:
        st.metric(
            t["feature_analysis.mean_corr"],
            f"{era_corr_df['correlation'].mean():.6f}",
        )
    with col_median:
        st.metric(
            t["feature_analysis.std_corr"],
            f"{era_corr_df['correlation'].std():.6f}",
        )
    with col_std_dev:
        stability = (era_corr_df["correlation"] > 0).sum() / len(era_corr_df) * 100
        st.metric(t["feature_analysis.pct_positive_eras"], f"{stability:.1f}%")

elif analysis_mode == analysis_modes[1]:
    compare_features = st.sidebar.multiselect(
        t["feature_analysis.comparison_select"],
        feature_set,
        default=feature_set[:3],
    )

    if len(compare_features) < 2:
        st.warning(t["feature_analysis.too_few_features"])
        st.stop()

    if len(compare_features) > 5:
        st.warning(t["feature_analysis.too_many_features"])
        compare_features = compare_features[:5]

    st.header(
        t.format("feature_analysis.comparison_header", count=len(compare_features))
    )

    feature_types = {}
    normalized_features = []

    for feat in compare_features:
        feature_name = feat[0] if isinstance(feat, list) else feat
        normalized_features.append(feature_name)
        feature_types[feature_name] = (
            discrete_label
            if is_discrete_feature(train[feature_name])
            else continuous_label
        )

    st.subheader(t["feature_analysis.comparison_stats"])
    comparison_stats = []

    for feature_name in normalized_features:
        feat_corr = train[[feature_name, "target"]].corr().iloc[0, 1]
        comparison_stats.append(
            {
                "Feature": feature_name,
                "Type": feature_types[feature_name],
                "Mean": train[feature_name].mean(),
                "Std": train[feature_name].std(),
                "Missing": train[feature_name].isnull().sum(),
                "Correlation": feat_corr,
                "Abs_Correlation": abs(feat_corr),
            }
        )

    comp_df = pd.DataFrame(comparison_stats).sort_values(
        "Abs_Correlation", ascending=False
    )

    st.dataframe(
        comp_df.style.background_gradient(
            subset=["Correlation"], cmap="RdYlGn", vmin=-0.1, vmax=0.1
        ),
        width="stretch",
    )

    st.divider()

    discrete_features = [
        f for f in compare_features if feature_types[f] == discrete_label
    ]
    continuous_features = [
        f for f in compare_features if feature_types[f] == continuous_label
    ]

    if discrete_features:
        st.subheader(t["feature_analysis.discrete_comparison_header"])

        comparison_data = []
        for feature in discrete_features:
            value_counts = train[feature].value_counts(normalize=True).sort_index()
            for value, pct in value_counts.items():
                comparison_data.append(
                    {
                        "Feature": feature,
                        "Value": value,
                        "Percent": pct * 100,
                    }
                )

        comp_dist_df = pd.DataFrame(comparison_data)

        fig = px.bar(
            comp_dist_df,
            x="Value",
            y="Percent",
            color="Feature",
            barmode="group",
            title=t["feature_analysis.discrete_comparison_title"],
            labels={
                "Percent": t["common.percent"],
                "Value": t["feature_analysis.feature_value_xaxis"],
            },
            height=500,
        )
        fig.update_layout(
            xaxis_title=t["feature_analysis.feature_value_xaxis"],
            yaxis_title=t["common.percent"],
        )

        st.plotly_chart(fig, width="stretch")

    if continuous_features:
        st.subheader(t["feature_analysis.continuous_distributions_header"])
        st.info(
            t.format(
                "feature_analysis.continuous_detected_info",
                count=len(continuous_features),
            )
        )

        for feature in continuous_features:
            fig = px.histogram(
                train,
                x=feature,
                nbins=50,
                title=t.format(
                    "feature_analysis.continuous_dist_title", feature=feature
                ),
                labels={feature: t["feature_analysis.feature_value_xaxis"]},
            )
            fig.update_layout(
                xaxis_title=t["feature_analysis.feature_value_xaxis"],
                yaxis_title=t["feature_analysis.count_yaxis"],
                height=400,
            )
            st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader(t["feature_analysis.target_corr_header"])

    fig = px.bar(
        comp_df,
        x="Feature",
        y="Correlation",
        title=t["feature_analysis.target_corr_title"],
        color="Correlation",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        text="Correlation",
    )
    fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        xaxis_title=t["common.feature"],
        yaxis_title=t["feature_analysis.target_corr_yaxis"],
    )

    st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader(t["feature_analysis.target_per_feature_header"])

    for feature in compare_features:
        with st.expander(f"📊 {feature} ({feature_types[feature]})"):
            col_mean, col_median = st.columns(2)

            with col_mean:
                if feature_types[feature] == discrete_label:
                    fig = px.box(
                        train,
                        x=feature,
                        y="target",
                        title=f"Target vs {feature}",
                        labels={
                            feature: t["feature_analysis.feature_value_label"],
                            "target": t["feature_analysis.target_label"],
                        },
                        color=feature,
                    )
                    fig.update_layout(height=400, showlegend=False)
                    fig.update_yaxes(title_text=t["feature_analysis.target_yaxis"])
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info(t["feature_analysis.boxplot_omitted"])

            with col_median:
                if feature_types[feature] == discrete_label:
                    target_by_feat = (
                        train.groupby(feature)["target"].mean().reset_index()
                    )
                else:
                    train["temp_binned"] = pd.qcut(
                        train[feature],
                        q=4,
                        labels=["Q1", "Q2", "Q3", "Q4"],
                        duplicates="drop",
                    )
                    target_by_feat = (
                        train.groupby("temp_binned")["target"].mean().reset_index()
                    )
                    target_by_feat.columns = [feature, "target"]

                fig = px.bar(
                    target_by_feat,
                    x=feature,
                    y="target",
                    title=t.format(
                        "feature_analysis.mean_target_per_feature", feature=feature
                    ),
                    labels={
                        feature: t["feature_analysis.feature_value_label"],
                        "target": t["feature_analysis.mean_target_label"],
                    },
                    text="target",
                )
                fig.update_traces(
                    texttemplate="%{text:.6f}",
                    textposition="outside",
                )
                fig.add_hline(
                    y=train["target"].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=t["feature_analysis.global_mean_annotation"],
                )
                fig.update_layout(height=400)
                fig.update_yaxes(title_text=t["feature_analysis.mean_target_yaxis"])
                st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader(t["feature_analysis.inter_feature_header"])

    corr_matrix = train[compare_features].corr()

    fig = px.imshow(
        corr_matrix,
        title=t["feature_analysis.inter_feature_matrix_title"],
        labels={"color": t["common.correlation"]},
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".3f",
    )
    fig.update_xaxes(side="bottom")

    st.plotly_chart(fig, width="stretch")

else:
    num_features_batch = st.sidebar.slider(
        t["feature_analysis.batch_stats_header"],
        10,
        min(100, len(feature_set)),
        30,
        10,
    )

    batch_features = feature_set[:num_features_batch]

    st.header(t.format("feature_analysis.batch_header", count=num_features_batch))

    st.subheader(t["feature_analysis.batch_stats_header"])

    with st.spinner(t["feature_analysis.batch_computing"]):
        batch_stats = []

        selected_target = st.session_state.get("selected_target", "target")
        if isinstance(selected_target, list | tuple | np.ndarray | pd.Index):
            if len(selected_target) > 0:
                selected_target = selected_target[0]
            else:
                selected_target = "target"
        for feature in batch_features:
            feature_data = train[feature]
            batch_corr = train[[feature, selected_target]].corr().iloc[0, 1]

            value_counts = feature_data.value_counts(normalize=True)
            entropy = -(value_counts * np.log2(value_counts + 1e-10)).sum()

            feat_type = (
                discrete_label
                if is_discrete_feature(feature_data)
                else continuous_label
            )

            batch_stats.append(
                {
                    "Feature": feature,
                    "Type": feat_type,
                    "Mean": feature_data.mean(),
                    "Std": feature_data.std(),
                    "Missing": feature_data.isnull().sum(),
                    "Unique_Values": feature_data.nunique(),
                    "Entropy": entropy,
                    "Correlation": batch_corr,
                    "Abs_Correlation": abs(batch_corr),
                }
            )

        batch_df = pd.DataFrame(batch_stats)

    discrete_count = (batch_df["Type"] == discrete_label).sum()
    continuous_count = (batch_df["Type"] == continuous_label).sum()

    col_mean, col_median = st.columns(2)
    col_mean.metric(t["feature_analysis.discrete_count"], discrete_count)
    col_median.metric(t["feature_analysis.continuous_count"], continuous_count)

    st.dataframe(
        batch_df.style.background_gradient(
            subset=["Correlation"],
            cmap="RdYlGn",
            vmin=-0.1,
            vmax=0.1,
        ),
        width="stretch",
        height=400,
    )

    csv = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=t["feature_analysis.batch_download"],
        data=csv,
        file_name="feature_batch_analysis.csv",
        mime="text/csv",
    )

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            t["feature_analysis.tab_corr_distribution"],
            t["feature_analysis.tab_mean_vs_std"],
            t["feature_analysis.tab_entropy_vs_corr"],
        ]
    )

    with tab1:
        fig = px.histogram(
            batch_df,
            x="Correlation",
            nbins=50,
            title=t["feature_analysis.corr_dist_title"],
            marginal="box",
            labels={"Correlation": t["feature_analysis.corr_value_xaxis"]},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(
            xaxis_title=t["feature_analysis.corr_value_xaxis"],
            yaxis_title=t["feature_analysis.feature_count_yaxis"],
        )
        st.plotly_chart(fig, width="stretch")

    with tab2:
        fig = px.scatter(
            batch_df,
            x="Mean",
            y="Std",
            hover_data=["Feature"],
            color="Abs_Correlation",
            size="Abs_Correlation",
            title=t["feature_analysis.mean_vs_std_title"],
            labels={
                "Mean": t["feature_analysis.mean_value_label"],
                "Std": t["feature_analysis.std_label"],
            },
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            xaxis_title=t["feature_analysis.mean_value_label"],
            yaxis_title=t["feature_analysis.std_label"],
        )
        st.plotly_chart(fig, width="stretch")

    with tab3:
        fig = px.scatter(
            batch_df,
            x="Entropy",
            y="Abs_Correlation",
            hover_data=["Feature"],
            color="Mean",
            size="Std",
            title=t["feature_analysis.entropy_vs_corr_title"],
            labels={
                "Entropy": t["feature_analysis.entropy_label"],
                "Abs_Correlation": t["feature_analysis.abs_corr_label"],
            },
            color_continuous_scale="Plasma",
        )
        fig.update_layout(
            xaxis_title=t["feature_analysis.entropy_label"],
            yaxis_title=t["feature_analysis.abs_corr_label"],
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader(t["feature_analysis.top20_header"])

    col_mean, col_median = st.columns(2)

    with col_mean:
        st.markdown(t["feature_analysis.top_positive"])
        top_pos = batch_df.nlargest(20, "Correlation")

        fig = px.bar(
            top_pos,
            y="Feature",
            x="Correlation",
            orientation="h",
            title=t["feature_analysis.top_positive_title"],
            text="Correlation",
            color="Correlation",
            color_continuous_scale="Blues",
        )
        fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
        fig.update_layout(height=600)
        fig.update_xaxes(title_text=t["feature_analysis.corr_xaxis"])
        st.plotly_chart(fig, width="stretch")

    with col_median:
        st.markdown(t["feature_analysis.top_negative"])
        top_neg = batch_df.nsmallest(20, "Correlation")

        fig = px.bar(
            top_neg,
            y="Feature",
            x="Correlation",
            orientation="h",
            title=t["feature_analysis.top_negative_title"],
            text="Correlation",
            color="Correlation",
            color_continuous_scale="Reds",
        )
        fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
        fig.update_layout(height=600)
        fig.update_xaxes(title_text=t["feature_analysis.corr_xaxis"])
        st.plotly_chart(fig, width="stretch")

st.divider()
