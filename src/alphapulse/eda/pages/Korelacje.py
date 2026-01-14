"""
Correlation Analysis Page - Explore feature-target and inter-feature correlations
"""

import os
import sys

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


st.set_page_config(page_title="Analiza Korelacji", page_icon="🔗", layout="wide")

if "data_loaded" not in st.session_state:
    st.warning("⚠️ Dane nie zostały załadowane. Przejdź do strony głównej.")
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]

st.title("🔗 Analiza Korelacji")
st.markdown(
    "Badaj korelacje cech z targetem oraz między cechami identyfikując zależności."
)

# ============================================================================
# SIDEBAR: TARGET SELECTION + CORRELATION TYPE
# ============================================================================
st.sidebar.header("⚙️ Ustawienia Korelacji")

# TARGET SELECTION DROPDOWN
target_columns = [
    col for col in train.columns if col == "target" or col.startswith("target")
]
if len(target_columns) == 0:
    target_columns = ["target"]

selected_target = st.sidebar.selectbox(
    "Wybierz Target:",
    target_columns,
    index=0,
    help="Target = zmienna docelowa do przewidzenia",
)

st.sidebar.info(f"📊 Analizowany Target: **{selected_target}**")

# Correlation type
corr_type = st.sidebar.selectbox(
    "Typ analizy:",
    [
        "Cechy vs Target",
        "Korelacje między cechami",
        "Macierz korelacji",
        "Network Graph",
    ],
)

num_features = st.sidebar.slider(
    "Liczba cech do analizy:",
    min_value=10,
    max_value=min(150, len(feature_set)),
    value=50,
    step=10,
)

sample_features = feature_set[:num_features]

st.divider()

# ============================================================================
# FEATURES VS TARGET
# ============================================================================
if corr_type == "Cechy vs Target":
    st.header(f"📊 Korelacja Cech z {selected_target.upper()}")
    st.info(f"Analizuję **{len(feature_set)}** cech względem **{selected_target}**...")

    # Calculate correlations
    with st.spinner("Obliczanie korelacji..."):
        correlations = []
        for feat in feature_set:
            corr = train[[feat, selected_target]].corr().iloc[0, 1]
            correlations.append(
                {"Cecha": feat, "Korelacja": corr, "Abs_Korelacja": abs(corr)}
            )
        corr_df = pd.DataFrame(correlations).sort_values(
            "Abs_Korelacja", ascending=False
        )

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Średnia |Korelacja|", f"{corr_df['Abs_Korelacja'].mean():.6f}")
    with col2:
        st.metric("Max Dodatnia", f"{corr_df['Korelacja'].max():.6f}")
    with col3:
        st.metric("Max Ujemna", f"{corr_df['Korelacja'].min():.6f}")
    with col4:
        above_threshold = (corr_df["Abs_Korelacja"] > 0.01).sum()
        st.metric("Cechy z |r| > 0.01", above_threshold)

    st.divider()

    # Distribution of correlations
    st.subheader("Rozkład korelacji")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            corr_df,
            x="Korelacja",
            nbins=60,
            title=f"Histogram korelacji z {selected_target}",
            labels={"Korelacja": "Correlation"},
            marginal="box",
            color_discrete_sequence=["#3498db"],
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero")
        fig.add_vline(
            x=corr_df["Korelacja"].mean(),
            line_dash="dash",
            line_color="green",
            annotation_text="Średnia",
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.histogram(
            corr_df,
            x="Abs_Korelacja",
            nbins=40,
            title="Histogram wartości bezwzględnych korelacji",
            labels={"Abs_Korelacja": "Absolute Correlation"},
            marginal="box",
            color_discrete_sequence=["#e74c3c"],
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # Top correlations
    st.subheader("Top Korelacje")

    top_n = st.slider("Liczba top cech do wyświetlenia:", 10, 50, 20)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔵 Top Dodatnie Korelacje**")
        top_pos = corr_df.nlargest(top_n, "Korelacja")

        fig = px.bar(
            top_pos,
            y="Cecha",
            x="Korelacja",
            orientation="h",
            title=f"Top {top_n} Dodatnie Korelacje",
            text="Korelacja",
            color="Korelacja",
            color_continuous_scale="Blues",
        )
        fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
        fig.update_layout(height=max(400, top_n * 20))
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            top_pos[["Cecha", "Korelacja"]].reset_index(drop=True), width="stretch"
        )

    with col2:
        st.markdown("**🔴 Top Ujemne Korelacje**")
        top_neg = corr_df.nsmallest(top_n, "Korelacja")

        fig = px.bar(
            top_neg,
            y="Cecha",
            x="Korelacja",
            orientation="h",
            title=f"Top {top_n} Ujemne Korelacje",
            text="Korelacja",
            color="Korelacja",
            color_continuous_scale="Reds_r",
        )
        fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
        fig.update_layout(height=max(400, top_n * 20))
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            top_neg[["Cecha", "Korelacja"]].reset_index(drop=True), width="stretch"
        )

    # REMOVED: "Wszystkie Korelacje" section completely

    # Download only top correlations
    st.divider()
    csv = corr_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Pobierz wszystkie korelacje jako CSV",
        data=csv,
        file_name=f"feature_{selected_target}_correlations.csv",
        mime="text/csv",
    )

# ============================================================================
# INTER-FEATURE CORRELATIONS
# ============================================================================
elif corr_type == "Korelacje między cechami":
    st.header("🔄 Korelacje Między Cechami")
    st.info(
        f"Analiza {num_features} cech - \
            obliczanie macierzy {num_features}x{num_features}"
    )

    # Compute correlation matrix
    with st.spinner("Obliczanie macierzy korelacji..."):
        corr_matrix = train[sample_features].corr()

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    upper_triangle = np.triu(corr_matrix, k=1)
    upper_values = upper_triangle[upper_triangle != 0]

    with col1:
        st.metric("Średnia Korelacja", f"{upper_values.mean():.6f}")
    with col2:
        st.metric("Max Korelacja", f"{upper_values.max():.6f}")
    with col3:
        st.metric("Min Korelacja", f"{upper_values.min():.6f}")
    with col4:
        st.metric("Std Korelacji", f"{upper_values.std():.6f}")

    st.divider()

    # Heatmap
    st.subheader("Mapa Cieplna Korelacji")

    show_values = st.checkbox("Pokaż wartości na mapie", value=False)

    fig = px.imshow(
        corr_matrix,
        title=f"Macierz Korelacji ({num_features} cech)",
        labels={"color": "Korelacja"},
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

    # Highly correlated pairs
    st.subheader("Silnie Skorelowane Pary Cech")

    threshold = st.slider("Próg korelacji (wartość bezwzględna):", 0.0, 1.0, 0.5, 0.05)

    # Find correlated pairs
    corr_pairs = []
    for i in range(len(sample_features)):
        for j in range(i + 1, len(sample_features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                corr_pairs.append(
                    {
                        "Cecha_1": sample_features[i],
                        "Cecha_2": sample_features[j],
                        "Korelacja": corr_val,
                        "Abs_Korelacja": abs(corr_val),
                    }
                )

    if len(corr_pairs) > 0:
        pairs_df = pd.DataFrame(corr_pairs).sort_values(
            "Abs_Korelacja", ascending=False
        )

        st.success(
            f"✅ Znaleziono **{len(pairs_df)}** par z |korelacją| >= {threshold}"
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            top_pairs = pairs_df.head(30)
            fig = px.bar(
                top_pairs,
                x="Abs_Korelacja",
                y=[
                    f"{row['Cecha_1']} - {row['Cecha_2']}"
                    for _, row in top_pairs.iterrows()
                ],
                orientation="h",
                title="Top 30 Skorelowanych Par",
                color="Korelacja",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                text="Korelacja",
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(height=max(500, len(top_pairs) * 20))
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("**Statystyki Par:**")
            st.metric("Liczba Par", len(pairs_df))
            st.metric("Średnia |Korelacja|", f"{pairs_df['Abs_Korelacja'].mean():.4f}")
            st.metric("Max Korelacja", f"{pairs_df['Korelacja'].max():.4f}")
            st.metric("Min Korelacja", f"{pairs_df['Korelacja'].min():.4f}")

        st.dataframe(
            pairs_df.style.background_gradient(
                subset=["Korelacja"], cmap="RdBu_r", vmin=-1, vmax=1
            ),
            width="stretch",
            height=400,
        )

        csv = pairs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Pobierz pary jako CSV",
            data=csv,
            file_name="correlated_feature_pairs.csv",
            mime="text/csv",
        )
    else:
        st.warning(f"⚠️ Nie znaleziono par z |korelacją| >= {threshold}")

    st.divider()

    # Distribution
    st.subheader("Rozkład Wszystkich Korelacji Między Cechami")

    fig = px.histogram(
        x=upper_values,
        nbins=50,
        title="Histogram korelacji między cechami",
        labels={"x": "Korelacja", "y": "Liczba Par"},
        marginal="box",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, width="stretch")

# ============================================================================
# CORRELATION MATRIX WITH TARGET
# ============================================================================
elif corr_type == "Macierz korelacji":
    st.header(f"🎯 Macierz Korelacji z {selected_target.upper()}")
    st.info(f"Analiza {num_features} cech + {selected_target}")

    # Include target
    with st.spinner("Obliczanie macierzy korelacji..."):
        corr_with_target = train[sample_features + [selected_target]].corr()

    # Full heatmap
    st.subheader("Pełna Macierz Korelacji")

    fig = px.imshow(
        corr_with_target,
        title=f"Macierz Korelacji ({num_features} cech + {selected_target})",
        labels={"color": "Korelacja"},
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

    # Target correlation specifically
    st.subheader(f"Korelacje z {selected_target.upper()}")

    target_corr = (
        corr_with_target[selected_target]
        .drop(selected_target)
        .sort_values(key=abs, ascending=False)
    )

    top_n_vis = st.slider("Liczba cech do wizualizacji:", 10, 50, 30)
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
                "colorbar": {"title": "Korelacja"},
            },
            text=top_target_corr.values.round(6),
            textposition="outside",
            hovertemplate="%{y}<br>Korelacja: %{x:.6f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Top {top_n_vis} Cech - Korelacja z {selected_target}",
        xaxis_title="Korelacja",
        yaxis_title="Cecha",
        height=max(500, top_n_vis * 20),
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    st.plotly_chart(fig, width="stretch")

# ============================================================================
# NETWORK GRAPH
# ============================================================================
else:  # Network Graph
    st.header("🕸️ Network Graph - Zależności Między Cechami")
    st.markdown("""
Wizualizacja sieci zależności gdzie:
- **Węzły** = Cechy
- **Krawędzie** = Silne korelacje (powyżej progu)
    """)

    # Settings
    network_features = st.slider(
        "Liczba cech w grafie:", 10, min(60, len(feature_set)), 30
    )

    edge_threshold = st.slider("Próg korelacji dla krawędzi:", 0.1, 0.9, 0.5, 0.05)

    network_sample = feature_set[:network_features]

    with st.spinner("Tworzenie grafu sieci..."):
        corr_matrix = train[network_sample].corr()

    # Build edge list
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
            f"Graf zawiera {len(edges_df)} krawędzi między {network_features} węzłami"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Węzły (Cechy)", network_features)
        with col2:
            st.metric("Krawędzie", len(edges_df))
        with col3:
            density = (
                len(edges_df) / (network_features * (network_features - 1) / 2) * 100
            )
            st.metric("Gęstość Grafu", f"{density:.1f}%")

        # Network visualization

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
                    "title": "Liczba<br>Połączeń",
                    "thickness": 15,
                    "len": 0.7,
                },
                "line": {"width": 2, "color": "white"},
            },
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=f"Network Graph - Zależności między {network_features} cechami",
            showlegend=False,
            hovermode="closest",
            height=800,
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        )

        st.plotly_chart(fig, width="stretch")

        # Top connected features
        st.divider()
        st.subheader("Najbardziej Połączone Cechy")

        degree_df = pd.DataFrame(
            {
                "Cecha": list(G.nodes()),
                "Liczba_Połączeń": [G.degree(node) for node in G.nodes()],
            }
        ).sort_values("Liczba_Połączeń", ascending=False)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(
                degree_df.head(20),
                y="Cecha",
                x="Liczba_Połączeń",
                orientation="h",
                title="Top 20 - Najbardziej Połączone Cechy",
                color="Liczba_Połączeń",
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.dataframe(degree_df, width="stretch", height=500)
    else:
        st.warning(f"⚠️ Brak krawędzi dla progu {edge_threshold}. Zmniejsz próg.")

# Footer
st.divider()
