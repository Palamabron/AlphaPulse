"""
Feature Analysis Page - Individual feature exploration
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

st.set_page_config(page_title="Analiza Cech", page_icon="🔍", layout="wide")

if "data_loaded" not in st.session_state:
    st.warning("⚠️ Dane nie zostały załadowane. Przejdź do strony głównej.")
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]


# ============================================================================
# HELPER FUNCTION: Check if feature is discrete
# ============================================================================
def is_discrete_feature(series: pd.Series) -> bool:
    """Check if feature has only discrete values (0, 0.25, 0.5, 0.75, 1.0)"""
    unique_vals = series.dropna().unique()
    return set(unique_vals).issubset({0.0, 0.25, 0.5, 0.75, 1.0})


st.title("🔍 Analiza Cech")
st.markdown("""
Szczegółowa analiza indywidualnych cech i ich relacji z target.
**Target** = Znormalizowany zwrot z akcji (wartość do przewidzenia w turnieju Numerai).
""")

# ============================================================================
# FEATURE SELECTION
# ============================================================================
st.sidebar.header("⚙️ Wybór Cech")

analysis_mode = st.sidebar.radio(
    "Tryb analizy:", ["Pojedyncza cecha", "Porównanie cech", "Batch analysis"]
)

if analysis_mode == "Pojedyncza cecha":
    # ========================================================================
    # SINGLE FEATURE ANALYSIS
    # ========================================================================

    selected_feature = st.sidebar.selectbox("Wybierz cechę:", feature_set, index=0)

    st.header(f"📊 Analiza cechy: `{selected_feature}`")

    # Feature statistics
    feature_data = train[selected_feature]
    is_discrete = is_discrete_feature(feature_data)

    # Display feature type
    if is_discrete:
        st.success("✅ Cecha dyskretna (wartości: 0, 0.25, 0.5, 0.75, 1.0)")
    else:
        st.warning("⚠️ Cecha ciągła (wykryto wartości spoza zestawu dyskretnego)")

    col_mean, col_median, col_std_dev, col_missing, col_correlation = st.columns(5)

    with col_mean:
        st.metric("Średnia", f"{feature_data.mean():.4f}")
    with col_median:
        st.metric("Mediana", f"{feature_data.median():.4f}")
    with col_std_dev:
        st.metric("Std Dev", f"{feature_data.std():.4f}")
    with col_missing:
        st.metric("Brakujące", f"{feature_data.isnull().sum()}")
    with col_correlation:
        selected_target = st.session_state.get("selected_target", "target")

        # Upewnij się, że selected_target jest stringiem, nie listą
        if isinstance(selected_target, list | tuple):
            target_col = selected_target[0]
        else:
            target_col = selected_target

        corr = train[[selected_feature, target_col]].corr().iloc[0, 1]
        st.metric("Korelacja z Target", f"{corr:.6f}")

    st.divider()

    # Value distribution
    col_mean, col_median = st.columns([2, 1])

    with col_mean:
        st.subheader("Rozkład wartości cechy")

        if is_discrete:
            # Discrete: bar chart
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
                        "colorbar": {"title": "Liczba"},
                    },
                    hovertemplate="Wartość cechy:"
                    "%{x}<br>Liczba wystąpień: %{y:,}<extra></extra>",
                )
            )

            fig.update_layout(
                title=f"Rozkład wartości: {selected_feature}",
                xaxis_title="Wartość cechy",
                yaxis_title="Liczba wystąpień",
                height=400,
            )
        else:
            # Continuous: histogram
            fig = px.histogram(
                feature_data,
                nbins=50,
                title=f"Rozkład wartości: {selected_feature}",
                labels={selected_feature: "Wartość cechy"},
            )
            fig.update_layout(
                xaxis_title="Wartość cechy", yaxis_title="Liczba wystąpień", height=400
            )

        st.plotly_chart(fig, width="stretch")

    with col_median:
        st.subheader("Statystyki")

        if is_discrete:
            value_counts = feature_data.value_counts().sort_index()
            value_pct = value_counts / len(feature_data) * 100

            stats_df = pd.DataFrame(
                {
                    "Wartość": value_counts.index,
                    "Liczba": value_counts.values,
                    "Procent": value_pct.values.round(2),
                }
            )

            st.dataframe(stats_df, width="stretch", height=400)

            # Entropy
            entropy = -(value_pct / 100 * np.log2(value_pct / 100 + 1e-10)).sum()
            st.metric(
                "Entropia",
                f"{entropy:.4f}",
                help="Miara niepewności rozkładu (max=2.32 dla 5 wartości)",
            )
        else:
            # For continuous features
            percentiles_df = pd.DataFrame(
                {
                    "Percentyl": ["Min", "25%", "50%", "75%", "Max"],
                    "Wartość": [
                        feature_data.min(),
                        feature_data.quantile(0.25),
                        feature_data.quantile(0.50),
                        feature_data.quantile(0.75),
                        feature_data.max(),
                    ],
                }
            )
            st.dataframe(percentiles_df, width="stretch")
            st.metric("Unikalne wartości", f"{feature_data.nunique()}")

    st.divider()

    # Feature vs Target - Show boxplot only for discrete features
    st.subheader("Relacja z Target (znormalizowany zwrot z akcji)")

    if is_discrete:
        # For discrete features: show Boxplot and Violin Plot
        tab1, tab2, tab3 = st.tabs(["Boxplot", "Violin Plot", "Statystyki per wartość"])

        with tab1:
            fig = px.box(
                train,
                x=selected_feature,
                y="target",
                title=f"Rozkład Target dla wartości cechy {selected_feature}",
                labels={
                    selected_feature: "Wartość cechy",
                    "target": "Target (zwrot z akcji)",
                },
                color=selected_feature,
                color_discrete_sequence=px.colors.sequential.Viridis,
            )
            fig.update_layout(height=500, showlegend=False)
            fig.update_yaxes(title_text="Target (znormalizowany zwrot z akcji)")
            st.plotly_chart(fig, width="stretch")

        with tab2:
            fig = px.violin(
                train,
                x=selected_feature,
                y="target",
                title=f"Rozkład Target dla wartości cechy {selected_feature}",
                labels={
                    selected_feature: "Wartość cechy",
                    "target": "Target (zwrot z akcji)",
                },
                color=selected_feature,
                box=True,
                points="outliers",
            )
            fig.update_layout(height=500, showlegend=False)
            fig.update_yaxes(title_text="Target (znormalizowany zwrot z akcji)")
            st.plotly_chart(fig, width="stretch")

        with tab3:
            target_by_feature = (
                train.groupby(selected_feature)["target"]
                .agg(["count", "mean", "std", "min", "max"])
                .reset_index()
            )

            target_by_feature.columns = [
                "Wartość",
                "Liczba",
                "Średnia Target",
                "Std Target",
                "Min Target",
                "Max Target",
            ]

            st.dataframe(
                target_by_feature.style.background_gradient(
                    subset=["Średnia Target"], cmap="RdYlGn"
                ),
                width="stretch",
            )

            # Visualization
            fig = px.bar(
                target_by_feature,
                x="Wartość",
                y="Średnia Target",
                error_y="Std Target",
                title="Średnia Target per wartość cechy",
                labels={
                    "Średnia Target": "Średnia Target (zwrot)",
                    "Wartość": "Wartość cechy",
                },
                text="Średnia Target",
            )
            fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
            fig.add_hline(
                y=train["target"].mean(),
                line_dash="dash",
                annotation_text="Globalna średnia",
                line_color="red",
            )
            fig.update_layout(
                xaxis_title="Wartość cechy",
                yaxis_title="Średnia Target (zwrot z akcji)",
            )
            st.plotly_chart(fig, width="stretch")
    else:
        # For continuous features: only show Violin Plot and stats
        st.info(
            "Dla cech ciągłych pokazano "
            "tylko Violin Plot i statystyki (boxplot pominięty)"
        )

        tab1, tab2 = st.tabs(["Violin Plot", "Statystyki per binowany zakres"])

        with tab1:
            fig = px.violin(
                train,
                x=selected_feature,
                y="target",
                title=f"Rozkład Target dla cechy {selected_feature}",
                labels={
                    selected_feature: "Wartość cechy",
                    "target": "Target (zwrot z akcji)",
                },
                box=True,
                points="outliers",
            )
            fig.update_layout(height=500)
            fig.update_yaxes(title_text="Target (znormalizowany zwrot z akcji)")
            st.plotly_chart(fig, width="stretch")

        with tab2:
            # Bin continuous feature into quartiles
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
                "Kwartyl",
                "Liczba",
                "Średnia Target",
                "Std Target",
                "Min Target",
                "Max Target",
            ]

            st.dataframe(target_by_bin, width="stretch")

    st.divider()

    # Feature behavior over eras
    st.subheader("Zachowanie w czasie (Ery)")

    # Mean feature value per era
    feature_era = (
        train.groupby("era")[selected_feature].agg(["mean", "std"]).reset_index()
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Średnia wartość per Era", "Zmienność per Era"),
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
        annotation_text="Średnia globalna",
    )

    fig.update_xaxes(title_text="Era", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="Średnia wartość cechy", row=1, col=1)
    fig.update_yaxes(title_text="Odchylenie standardowe", row=2, col=1)

    fig.update_layout(height=700, showlegend=False)

    st.plotly_chart(fig, width="stretch")

    # Correlation with target per era
    st.subheader("Korelacja z Target per Era")

    era_correlations = []
    for era in train["era"].unique():
        era_data = train[train["era"] == era]
        corr = era_data[[selected_feature, "target"]].corr().iloc[0, 1]
        era_correlations.append({"era": era, "correlation": corr})

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
        y=corr, line_dash="dash", line_color="red", annotation_text="Korelacja globalna"
    )

    fig.update_layout(
        title=f"Stabilność korelacji {selected_feature} z Target",
        xaxis_title="Era",
        yaxis_title="Wartość korelacji",
        height=500,
    )
    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, width="stretch")

    col_mean, col_median, col_std_dev = st.columns(3)

    with col_mean:
        st.metric("Średnia Korelacja", f"{era_corr_df['correlation'].mean():.6f}")
    with col_median:
        st.metric("Std Korelacji", f"{era_corr_df['correlation'].std():.6f}")
    with col_std_dev:
        stability = (era_corr_df["correlation"] > 0).sum() / len(era_corr_df) * 100
        st.metric("% Er z dodatnią korelacją", f"{stability:.1f}%")

elif analysis_mode == "Porównanie cech":
    # ========================================================================
    # FEATURE COMPARISON
    # ========================================================================

    compare_features = st.sidebar.multiselect(
        "Wybierz cechy do porównania (2-5):", feature_set, default=feature_set[:3]
    )

    if len(compare_features) < 2:
        st.warning("Wybierz co najmniej 2 cechy do porównania.")
        st.stop()

    if len(compare_features) > 5:
        st.warning("Zbyt wiele cech. Wyświetlane będą tylko pierwsze 5.")
        compare_features = compare_features[:5]

    st.header(f"🔄 Porównanie {len(compare_features)} cech")

    # Check which features are discrete vs continuous
    feature_types = {}
    normalized_features = []

    for feat in compare_features:
        feature_name = feat[0] if isinstance(feat, list) else feat
        normalized_features.append(feature_name)
        feature_types[feature_name] = (
            "Dyskretna" if is_discrete_feature(train[feature_name]) else "Ciągła"
        )
    # Comparison statistics
    st.subheader("Statystyki porównawcze")
    comparison_stats = []

    for feature_name in normalized_features:
        corr = train[[feature_name, "target"]].corr().iloc[0, 1]
        comparison_stats.append(
            {
                "Cecha": feature_name,
                "Typ": feature_types[feature_name],
                "Mean": train[feature_name].mean(),
                "Std": train[feature_name].std(),
                "Missing": train[feature_name].isnull().sum(),
                "Korelacja": corr,
                "Abs_Korelacja": abs(corr),
            }
        )

    comp_df = pd.DataFrame(comparison_stats).sort_values(
        "Abs_Korelacja", ascending=False
    )

    st.dataframe(
        comp_df.style.background_gradient(
            subset=["Korelacja"], cmap="RdYlGn", vmin=-0.1, vmax=0.1
        ),
        width="stretch",
    )

    st.divider()

    # Separate discrete and continuous features
    discrete_features = [f for f in compare_features if feature_types[f] == "Dyskretna"]
    continuous_features = [f for f in compare_features if feature_types[f] == "Ciągła"]

    # Distribution comparison for discrete features
    if discrete_features:
        st.subheader("Porównanie rozkładów - Cechy Dyskretne")

        comparison_data = []
        for feature in discrete_features:
            value_counts = train[feature].value_counts(normalize=True).sort_index()
            for value, pct in value_counts.items():
                comparison_data.append(
                    {"Cecha": feature, "Wartość": value, "Procent": pct * 100}
                )

        comp_dist_df = pd.DataFrame(comparison_data)

        fig = px.bar(
            comp_dist_df,
            x="Wartość",
            y="Procent",
            color="Cecha",
            barmode="group",
            title="Rozkład wartości - Cechy Dyskretne",
            labels={"Procent": "Procent (%)", "Wartość": "Wartość cechy"},
            height=500,
        )
        fig.update_layout(xaxis_title="Wartość cechy", yaxis_title="Procent (%)")

        st.plotly_chart(fig, width="stretch")

    # Distribution for continuous features
    if continuous_features:
        st.subheader("Rozkłady - Cechy Ciągłe")
        st.info(f"Wykryto {len(continuous_features)} cech ciągłych")

        for feature in continuous_features:
            fig = px.histogram(
                train,
                x=feature,
                nbins=50,
                title=f"Rozkład: {feature} (ciągła)",
                labels={feature: "Wartość cechy"},
            )
            fig.update_layout(
                xaxis_title="Wartość cechy", yaxis_title="Liczba wystąpień", height=400
            )
            st.plotly_chart(fig, width="stretch")

    st.divider()

    # Correlation with target comparison
    st.subheader("Porównanie korelacji z Target (zwrot z akcji)")

    fig = px.bar(
        comp_df,
        x="Cecha",
        y="Korelacja",
        title="Korelacja z Target",
        color="Korelacja",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        text="Korelacja",
    )
    fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_xaxes(tickangle=45)
    fig.update_layout(xaxis_title="Cecha", yaxis_title="Wartość korelacji z Target")

    st.plotly_chart(fig, width="stretch")

    st.divider()

    # Target distribution per feature value (only boxplot for discrete)
    st.subheader("Target (zwrot z akcji) per wartość cechy")

    for feature in compare_features:
        with st.expander(f"📊 {feature} ({feature_types[feature]})"):
            col_mean, col_median = st.columns(2)

            with col_mean:
                # Show boxplot only for discrete features
                if feature_types[feature] == "Dyskretna":
                    fig = px.box(
                        train,
                        x=feature,
                        y="target",
                        title=f"Target vs {feature}",
                        labels={feature: "Wartość cechy", "target": "Target (zwrot)"},
                        color=feature,
                    )
                    fig.update_layout(height=400, showlegend=False)
                    fig.update_yaxes(title_text="Target (zwrot z akcji)")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Boxplot pominięty dla cechy ciągłej")

            with col_median:
                if feature_types[feature] == "Dyskretna":
                    target_by_feat = (
                        train.groupby(feature)["target"].mean().reset_index()
                    )
                else:
                    # Bin continuous features
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
                    title=f"Średnia Target per {feature}",
                    labels={
                        feature: "Wartość cechy",
                        "target": "Średnia Target (zwrot)",
                    },
                    text="target",
                )
                fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
                fig.add_hline(
                    y=train["target"].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Średnia globalna",
                )
                fig.update_layout(height=400)
                fig.update_yaxes(title_text="Średnia Target (zwrot z akcji)")
                st.plotly_chart(fig, width="stretch")

    st.divider()

    # Pairwise correlations
    st.subheader("Korelacje między cechami")

    corr_matrix = train[compare_features].corr()

    fig = px.imshow(
        corr_matrix,
        title="Macierz korelacji wybranych cech",
        labels={"color": "Korelacja"},
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".3f",
    )
    fig.update_xaxes(side="bottom")

    st.plotly_chart(fig, width="stretch")

else:
    # ========================================================================
    # BATCH ANALYSIS
    # ========================================================================

    num_features_batch = st.sidebar.slider(
        "Liczba cech do analizy:", 10, min(100, len(feature_set)), 30, 10
    )

    batch_features = feature_set[:num_features_batch]

    st.header(f"📦 Batch Analysis: {num_features_batch} cech")

    st.subheader("Statystyki zbiorcze")

    with st.spinner("Obliczanie statystyk..."):
        batch_stats = []

        selected_target = st.session_state.get("selected_target", "target")
        # Ensure selected_target is a single, hashable column name
        if isinstance(selected_target, list | tuple | np.ndarray | pd.Index):
            if len(selected_target) > 0:
                selected_target = selected_target[0]
            else:
                selected_target = "target"
        for feature in batch_features:
            feature_data = train[feature]
            corr = train[[feature, selected_target]].corr().iloc[0, 1]

            value_counts = feature_data.value_counts(normalize=True)
            entropy = -(value_counts * np.log2(value_counts + 1e-10)).sum()

            feat_type = "Dyskretna" if is_discrete_feature(feature_data) else "Ciągła"

            batch_stats.append(
                {
                    "Cecha": feature,
                    "Typ": feat_type,
                    "Mean": feature_data.mean(),
                    "Std": feature_data.std(),
                    "Missing": feature_data.isnull().sum(),
                    "Unique_Values": feature_data.nunique(),
                    "Entropy": entropy,
                    "Korelacja": corr,
                    "Abs_Korelacja": abs(corr),
                }
            )

        batch_df = pd.DataFrame(batch_stats)

    # Count discrete vs continuous
    discrete_count = (batch_df["Typ"] == "Dyskretna").sum()
    continuous_count = (batch_df["Typ"] == "Ciągła").sum()

    col_mean, col_median = st.columns(2)
    col_mean.metric("Cechy Dyskretne", discrete_count)
    col_median.metric("Cechy Ciągłe", continuous_count)

    st.dataframe(
        batch_df.style.background_gradient(
            subset=["Korelacja"], cmap="RdYlGn", vmin=-0.1, vmax=0.1
        ),
        width="stretch",
        height=400,
    )

    # Download option
    csv = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Pobierz jako CSV",
        data=csv,
        file_name="feature_batch_analysis.csv",
        mime="text/csv",
    )

    st.divider()

    # Visualizations
    tab1, tab2, tab3 = st.tabs(
        ["Rozkład korelacji", "Mean vs Std", "Entropia vs Korelacja"]
    )

    with tab1:
        fig = px.histogram(
            batch_df,
            x="Korelacja",
            nbins=50,
            title="Rozkład korelacji z Target (zwrot z akcji)",
            marginal="box",
            labels={"Korelacja": "Wartość korelacji"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(xaxis_title="Wartość korelacji", yaxis_title="Liczba cech")
        st.plotly_chart(fig, width="stretch")

    with tab2:
        fig = px.scatter(
            batch_df,
            x="Mean",
            y="Std",
            hover_data=["Cecha"],
            color="Abs_Korelacja",
            size="Abs_Korelacja",
            title="Mean vs Standard Deviation",
            labels={"Mean": "Średnia wartość", "Std": "Odchylenie standardowe"},
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            xaxis_title="Średnia wartość", yaxis_title="Odchylenie standardowe"
        )
        st.plotly_chart(fig, width="stretch")

    with tab3:
        fig = px.scatter(
            batch_df,
            x="Entropy",
            y="Abs_Korelacja",
            hover_data=["Cecha"],
            color="Mean",
            size="Std",
            title="Entropy vs Absolute Correlation",
            labels={
                "Entropy": "Entropia",
                "Abs_Korelacja": "Wartość bezwzględna korelacji",
            },
            color_continuous_scale="Plasma",
        )
        fig.update_layout(
            xaxis_title="Entropia", yaxis_title="Wartość bezwzględna korelacji"
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # Top features by correlation
    st.subheader("Top 20 cech według korelacji z Target")

    col_mean, col_median = st.columns(2)

    with col_mean:
        st.markdown("**Najsilniejsze dodatnie**")
        top_pos = batch_df.nlargest(20, "Korelacja")

        fig = px.bar(
            top_pos,
            y="Cecha",
            x="Korelacja",
            orientation="h",
            title="Top 20 dodatnie korelacje",
            text="Korelacja",
            color="Korelacja",
            color_continuous_scale="Blues",
        )
        fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
        fig.update_layout(height=600)
        fig.update_xaxes(title_text="Wartość korelacji")
        st.plotly_chart(fig, width="stretch")

    with col_median:
        st.markdown("**Najsilniejsze ujemne**")
        top_neg = batch_df.nsmallest(20, "Korelacja")

        fig = px.bar(
            top_neg,
            y="Cecha",
            x="Korelacja",
            orientation="h",
            title="Top 20 ujemne korelacje",
            text="Korelacja",
            color="Korelacja",
            color_continuous_scale="Reds",
        )
        fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
        fig.update_layout(height=600)
        fig.update_xaxes(title_text="Wartość korelacji")
        st.plotly_chart(fig, width="stretch")

# Footer
st.divider()
