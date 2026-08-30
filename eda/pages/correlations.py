"""Correlation Analysis Page — Feature-target and inter-feature correlations."""

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eda.utils import get_translations

st.set_page_config(page_title="Correlation Analysis", page_icon="🔗", layout="wide")

t = get_translations()

if "data_loaded" not in st.session_state:
    st.warning(t["errors.data_not_loaded"])
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]

st.title(t["correlations.title"])
st.markdown(t["correlations.description"])

st.sidebar.header(t["correlations.sidebar_header"])

target_columns = [
    col for col in train.columns if col == "target" or col.startswith("target")
]
if len(target_columns) == 0:
    target_columns = ["target"]

selected_target = st.sidebar.selectbox(
    t["correlations.target_select"],
    target_columns,
    index=0,
    help=t["correlations.target_help"],
)

st.sidebar.info(t.format("correlations.target_info", target=selected_target))

analysis_modes = [
    t["correlations.mode_feature_vs_target"],
    t["correlations.mode_inter_feature"],
    t["correlations.mode_matrix"],
    t["correlations.mode_network"],
]

corr_type = st.sidebar.selectbox(t["correlations.analysis_type"], analysis_modes)

num_features = st.sidebar.slider(
    t["correlations.num_features_label"],
    min_value=10,
    max_value=min(150, len(feature_set)),
    value=50,
    step=10,
)

sample_features = feature_set[:num_features]

st.divider()


def _normalize_col_name(col: str | int | float | list | tuple) -> str:
    if isinstance(col, list | tuple):
        return str(col[0]) if col else "unknown"
    return str(col)


if corr_type == analysis_modes[0]:
    st.header(
        t.format(
            "correlations.feature_vs_target_header", target=selected_target.upper()
        )
    )
    st.info(
        t.format(
            "correlations.feature_vs_target_info",
            count=len(feature_set),
            target=selected_target,
        )
    )

    try:
        with st.spinner(t["correlations.computing"]):
            correlations = []
            target_col = _normalize_col_name(selected_target)
            for feat in feature_set:
                feat_col = _normalize_col_name(feat)
                corr = train[[feat_col, target_col]].corr().iloc[0, 1]
                correlations.append(
                    {
                        "Feature": feat_col,
                        "Correlation": corr,
                        "Abs_Correlation": abs(corr),
                    }
                )
            corr_df = pd.DataFrame(correlations).sort_values(
                "Abs_Correlation", ascending=False
            )
    except KeyError as e:
        st.error(t.format("errors.column_not_found", column=str(e)))
        st.stop()
    except ValueError as e:
        st.error(t.format("errors.computation_error", error=str(e)))
        st.stop()
    except Exception as e:
        st.error(t.format("errors.unexpected_error", error=str(e)))
        st.stop()

    avg_corr_col, max_positive_col, max_negative_col, threshold_count_col = st.columns(
        4
    )

    with avg_corr_col:
        st.metric(
            t["correlations.avg_abs_corr"], f"{corr_df['Abs_Correlation'].mean():.6f}"
        )
    with max_positive_col:
        st.metric(t["correlations.max_positive"], f"{corr_df['Correlation'].max():.6f}")
    with max_negative_col:
        st.metric(t["correlations.max_negative"], f"{corr_df['Correlation'].min():.6f}")
    with threshold_count_col:
        above_threshold = (corr_df["Abs_Correlation"] > 0.01).sum()
        st.metric(t["correlations.features_above_threshold"], above_threshold)

    st.divider()

    st.subheader(t["correlations.distribution_header"])

    avg_corr_col, max_positive_col = st.columns(2)

    with avg_corr_col:
        fig = px.histogram(
            corr_df,
            x="Correlation",
            nbins=60,
            title=t.format("correlations.histogram_title", target=selected_target),
            labels={"Correlation": "Correlation"},
            marginal="box",
            color_discrete_sequence=["#3498db"],
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero")
        fig.add_vline(
            x=corr_df["Correlation"].mean(),
            line_dash="dash",
            line_color="green",
            annotation_text=t["correlations.mean_annotation"],
        )
        st.plotly_chart(fig, width="stretch")

    with max_positive_col:
        fig = px.histogram(
            corr_df,
            x="Abs_Correlation",
            nbins=40,
            title=t["correlations.abs_histogram_title"],
            labels={"Abs_Correlation": "Absolute Correlation"},
            marginal="box",
            color_discrete_sequence=["#e74c3c"],
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader(t["correlations.top_correlations_header"])

    top_n = st.slider(t["correlations.top_n_slider"], 10, 50, 20)

    avg_corr_col, max_positive_col = st.columns(2)

    with avg_corr_col:
        st.markdown(t["correlations.top_positive"])
        top_pos = corr_df.nlargest(top_n, "Correlation")

        fig = px.bar(
            top_pos,
            y="Feature",
            x="Correlation",
            orientation="h",
            title=t.format("correlations.top_positive_title", n=top_n),
            text="Correlation",
            color="Correlation",
            color_continuous_scale="Blues",
        )
        fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
        fig.update_layout(height=max(400, top_n * 20))
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            top_pos[["Feature", "Correlation"]].reset_index(drop=True), width="stretch"
        )

    with max_positive_col:
        st.markdown(t["correlations.top_negative"])
        top_neg = corr_df.nsmallest(top_n, "Correlation")

        fig = px.bar(
            top_neg,
            y="Feature",
            x="Correlation",
            orientation="h",
            title=t.format("correlations.top_negative_title", n=top_n),
            text="Correlation",
            color="Correlation",
            color_continuous_scale="Reds_r",
        )
        fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
        fig.update_layout(height=max(400, top_n * 20))
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            top_neg[["Feature", "Correlation"]].reset_index(drop=True), width="stretch"
        )

    st.divider()
    csv = corr_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=t["correlations.download_all"],
        data=csv,
        file_name=f"feature_{selected_target}_correlations.csv",
        mime="text/csv",
    )


elif corr_type == analysis_modes[1]:
    st.header(t["correlations.inter_feature_header"])
    st.info(t.format("correlations.inter_feature_info", count=num_features))

    with st.spinner(t["correlations.computing_matrix"]):
        corr_matrix = train[sample_features].corr()

    avg_corr_col, max_positive_col, max_negative_col, threshold_count_col = st.columns(
        4
    )

    upper_triangle = np.triu(corr_matrix, k=1)
    upper_values = upper_triangle[upper_triangle != 0]

    with avg_corr_col:
        st.metric(t["correlations.avg_corr"], f"{upper_values.mean():.6f}")
    with max_positive_col:
        st.metric(t["correlations.max_corr"], f"{upper_values.max():.6f}")
    with max_negative_col:
        st.metric(t["correlations.min_corr"], f"{upper_values.min():.6f}")
    with threshold_count_col:
        st.metric(t["correlations.std_corr"], f"{upper_values.std():.6f}")

    st.divider()

    st.subheader(t["correlations.heatmap_header"])

    show_values = st.checkbox(t["correlations.show_values"], value=False)

    fig = px.imshow(
        corr_matrix,
        title=t.format("correlations.heatmap_title", count=num_features),
        labels={"color": "Correlation"},
        x=sample_features,
        y=sample_features,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        text_auto=".2f" if show_values else False,
    )

    fig.update_xaxes(side="bottom", tickangle=90, showticklabels=num_features <= 50)
    fig.update_yaxes(showticklabels=num_features <= 50)
    fig.update_layout(height=max(600, num_features * 8))

    st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader(t["correlations.strong_pairs_header"])

    threshold = st.slider(t["correlations.threshold_slider"], 0.0, 1.0, 0.5, 0.05)

    corr_pairs = []
    for i in range(len(sample_features)):
        for j in range(i + 1, len(sample_features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                corr_pairs.append(
                    {
                        "Feature_1": sample_features[i],
                        "Feature_2": sample_features[j],
                        "Correlation": corr_val,
                        "Abs_Correlation": abs(corr_val),
                    }
                )

    if len(corr_pairs) > 0:
        pairs_df = pd.DataFrame(corr_pairs).sort_values(
            "Abs_Correlation", ascending=False
        )

        st.success(
            t.format(
                "correlations.pairs_found", count=len(pairs_df), threshold=threshold
            )
        )

        avg_corr_col, max_positive_col = st.columns([2, 1])

        with avg_corr_col:
            top_pairs = pairs_df.head(30)
            fig = px.bar(
                top_pairs,
                x="Abs_Correlation",
                y=[
                    f"{row['Feature_1']} - {row['Feature_2']}"
                    for _, row in top_pairs.iterrows()
                ],
                orientation="h",
                title="Top 30 Correlated Pairs",
                color="Correlation",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                text="Correlation",
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(height=max(500, len(top_pairs) * 20))
            st.plotly_chart(fig, width="stretch")

        with max_positive_col:
            st.markdown(t["correlations.pairs_stats"])
            st.metric(t["correlations.num_pairs"], len(pairs_df))
            st.metric(
                t["correlations.avg_abs_corr"],
                f"{pairs_df['Abs_Correlation'].mean():.4f}",
            )
            st.metric(
                t["correlations.max_corr"], f"{pairs_df['Correlation'].max():.4f}"
            )
            st.metric(
                t["correlations.min_corr"], f"{pairs_df['Correlation'].min():.4f}"
            )

        st.dataframe(
            pairs_df.style.background_gradient(
                subset=["Correlation"], cmap="RdBu_r", vmin=-1, vmax=1
            ),
            width="stretch",
            height=400,
        )

        csv = pairs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=t["correlations.pairs_download"],
            data=csv,
            file_name="correlated_feature_pairs.csv",
            mime="text/csv",
        )
    else:
        st.warning(t.format("correlations.no_pairs", threshold=threshold))

    st.divider()

    st.subheader(t["correlations.all_corr_distribution"])

    fig = px.histogram(
        x=upper_values,
        nbins=50,
        title=t["correlations.all_corr_histogram_title"],
        labels={"x": "Correlation", "y": "Number of Pairs"},
        marginal="box",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, width="stretch")

elif corr_type == analysis_modes[2]:
    st.header(t.format("correlations.matrix_header", target=selected_target.upper()))
    st.info(
        t.format("correlations.matrix_info", count=num_features, target=selected_target)
    )

    with st.spinner(t["correlations.computing_matrix"]):
        target_col = (
            selected_target[0]
            if isinstance(selected_target, list | tuple)
            else selected_target
        )
        if isinstance(target_col, list):
            target_col = target_col[0]

        columns_to_analyze = sample_features + [target_col]
        corr_with_target = train[columns_to_analyze].corr()

    st.subheader(t["correlations.full_matrix_subheader"])

    fig = px.imshow(
        corr_with_target,
        title=t.format(
            "correlations.full_matrix_title", count=num_features, target=selected_target
        ),
        labels={"color": "Correlation"},
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )

    fig.update_xaxes(side="bottom", tickangle=90, showticklabels=num_features <= 40)
    fig.update_yaxes(showticklabels=num_features <= 40)
    fig.update_layout(height=max(700, num_features * 10))

    st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader(
        t.format("correlations.target_corr_header", target=selected_target.upper())
    )

    target_corr = (
        corr_with_target[selected_target]
        .drop(selected_target)
        .sort_values(key=abs, ascending=False)
    )

    top_n_vis = st.slider(t["correlations.top_n_vis_slider"], 10, 50, 30)
    top_target_corr = target_corr.head(top_n_vis)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=top_target_corr.values,
            y=top_target_corr.index,
            orientation="h",
            marker={
                "color": top_target_corr.values,
                "colorscale": "RdBu_r",
                "cmin": -max(abs(top_target_corr)),
                "cmax": max(abs(top_target_corr)),
                "colorbar": {"title": "Correlation"},
            },
            text=top_target_corr.values.round(6),
            textposition="outside",
            hovertemplate="%{y}<br>Correlation: %{x:.6f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=t.format(
            "correlations.top_features_title", n=top_n_vis, target=selected_target
        ),
        xaxis_title="Correlation",
        yaxis_title="Feature",
        height=max(500, top_n_vis * 20),
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    st.plotly_chart(fig, width="stretch")


else:
    st.header(t["correlations.network_header"])
    st.markdown(t["correlations.network_description"])

    network_features = st.slider(
        t["correlations.network_features_slider"],
        10,
        min(60, len(feature_set)),
        30,
    )

    edge_threshold = st.slider(
        t["correlations.edge_threshold_slider"], 0.1, 0.9, 0.5, 0.05
    )

    network_sample = feature_set[:network_features]

    with st.spinner(t["correlations.creating_graph"]):
        corr_matrix = train[network_sample].corr()

    edges = []
    for i in range(len(network_sample)):
        for j in range(i + 1, len(network_sample)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= edge_threshold:
                edges.append(
                    {
                        "source": network_sample[i],
                        "target": network_sample[j],
                        "weight": abs(corr_val),
                        "correlation": corr_val,
                    }
                )

    edges_df = pd.DataFrame(edges)

    if len(edges_df) > 0:
        st.success(
            t.format(
                "correlations.graph_info", edges=len(edges_df), nodes=network_features
            )
        )

        avg_corr_col, max_positive_col, max_negative_col = st.columns(3)

        with avg_corr_col:
            st.metric(t["correlations.nodes"], network_features)
        with max_positive_col:
            st.metric(t["correlations.edges"], len(edges_df))
        with max_negative_col:
            density = (
                len(edges_df) / (network_features * (network_features - 1) / 2) * 100
            )
            st.metric(t["correlations.graph_density"], f"{density:.1f}%")

        G: nx.Graph = nx.Graph()
        for _, row in edges_df.iterrows():
            G.add_edge(row["source"], row["target"], weight=row["weight"])

        pos = nx.spring_layout(G, k=2, iterations=50)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line={"width": 0.5, "color": "#888"},
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_degrees = [G.degree(node) for node in G.nodes()]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text if network_features <= 30 else None,
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            marker={
                "showscale": True,
                "colorscale": "YlOrRd",
                "size": [d * 3 + 10 for d in node_degrees],
                "color": node_degrees,
                "colorbar": {
                    "title": t["correlations.connections"],
                    "thickness": 15,
                    "len": 0.7,
                },
                "line": {"width": 2, "color": "white"},
            },
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=t.format("correlations.graph_title", count=network_features),
            showlegend=False,
            hovermode="closest",
            height=800,
            xaxis={
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
            },
            yaxis={
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
            },
        )

        st.plotly_chart(fig, width="stretch")

        st.divider()
        st.subheader(t["correlations.most_connected_header"])

        degree_df = pd.DataFrame(
            {
                "Feature": list(G.nodes()),
                "Connections": [G.degree(node) for node in G.nodes()],
            }
        ).sort_values("Connections", ascending=False)

        avg_corr_col, max_positive_col = st.columns([2, 1])

        with avg_corr_col:
            fig = px.bar(
                degree_df.head(20),
                y="Feature",
                x="Connections",
                orientation="h",
                title=t["correlations.most_connected_bar_title"],
                color="Connections",
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")

        with max_positive_col:
            st.dataframe(degree_df, width="stretch", height=500)
    else:
        st.warning(t.format("correlations.no_edges", threshold=edge_threshold))

st.divider()
