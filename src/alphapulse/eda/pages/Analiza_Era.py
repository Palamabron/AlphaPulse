"""
Era Analysis Page - Temporal analysis of features and target across eras
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


st.set_page_config(page_title="Analiza Era", page_icon="⏰", layout="wide")

if "data_loaded" not in st.session_state:
    st.warning("⚠️ Dane nie zostały załadowane. Przejdź do strony głównej.")
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]
all_targets = st.session_state.get("all_targets", [])

st.title("⏰ Analiza Temporalna (Ery)")
st.markdown("""
Ery w Numerai reprezentują różne punkty w czasie (piątki każdego tygodnia).

**Target** = Znormalizowany zwrot z akcji (wartość do przewidzenia w turnieju Numerai).

Analizuj charakterystyki danych i stabilność w czasie.

""")

# ============================================================================
# TARGET SELECTION (DROPDOWN) - WITH ALL TARGETS
# ============================================================================
st.sidebar.header("⚙️ Ustawienia Analizy")

# Get all target columns from data
if len(all_targets) == 0:
    all_targets = [col for col in train.columns if col.startswith("target")]

if len(all_targets) == 0:
    all_targets = ["target"]

selected_target = st.sidebar.selectbox(
    "Wybierz Target do analizy:",
    all_targets,
    index=0,
    help="Kliknij aby wybrać inny target",
)

st.info(f"📊 Analizowany Target: **{selected_target}** (znormalizowany zwrot z akcji)")

st.divider()

# ============================================================================
# ERA OVERVIEW
# ============================================================================
st.header("📊 Przegląd Er")

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

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Liczba Er", train["era"].nunique())

with col2:
    st.metric("Średnia Obs. per Era", f"{era_stats['count'].mean():.0f}")

with col3:
    st.metric("Min Obs. per Era", f"{era_stats['count'].min()}")

with col4:
    st.metric("Max Obs. per Era", f"{era_stats['count'].max()}")

st.divider()

# ============================================================================
# OBSERVATIONS PER ERA
# ============================================================================
st.header("📈 Obserwacje per Era")

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=era_stats["era"],
        y=era_stats["count"],
        marker={
            "color": era_stats["count"],
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": "LiczbaObs."},
        },
        text=era_stats["count"],
        textposition="outside",
        hovertemplate="Era: %{x}<br>Obserwacje: %{y:,}",
    )
)

fig.update_layout(
    title="Liczba obserwacji w każdej Erze",
    xaxis_title="Era",
    yaxis_title="Liczba Obserwacji",
    height=500,
    showlegend=False,
)

fig.update_xaxes(tickangle=45)
st.plotly_chart(fig, width="stretch")

# Statistics about era sizes
col1, col2, col3 = st.columns(3)

with col1:
    cv = (era_stats["count"].std() / era_stats["count"].mean()) * 100
    st.metric("CV Rozmiaru Er", f"{cv:.2f}%", help="Współczynnik zmienności rozmiaru")

with col2:
    st.metric(
        "Zakres Rozmiaru", f"{era_stats['count'].max() - era_stats['count'].min()}"
    )

with col3:
    st.metric("Mediana Obs.", f"{era_stats['count'].median():.0f}")

st.divider()

# ============================================================================
# TARGET ANALYSIS OVER TIME
# ============================================================================
st.header(f"🎯 Zachowanie {selected_target.upper()} w Czasie")

tab1, tab2, tab3 = st.tabs(["Średnia", "Zmienność", "Zakres"])

with tab1:
    st.subheader(f"Średnia {selected_target.upper()} per Era")

    fig = go.Figure()

    # Main line
    fig.add_trace(
        go.Scatter(
            x=era_stats["era"],
            y=era_stats["target_mean"],
            mode="lines+markers",
            name=f"Średnia {selected_target}",
            line={"color": "blue", "width": 2},
            marker={"size": 6},
            hovertemplate="Era: %{x}<br>Mean: %{y:.6f}",
        )
    )

    # Confidence band (mean ± std)
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

    # Global mean
    fig.add_hline(
        y=train[selected_target].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text="Średnia Globalna",
        annotation_position="right",
    )

    fig.update_layout(
        xaxis_title="Era",
        yaxis_title=f"Średnia {selected_target.upper()} (zwrot z akcji)",
        height=500,
        hovermode="x unified",
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")

    # Statistics with formula
    st.markdown("### 📐 Wzór Matematyczny - Stabilność:")

    st.latex(r"""
    \text{Stabilność} = \sigma(\bar{y}_{\text{era}})
    """)

    st.markdown("""
    gdzie:

    - \\(\\bar{y}_{\\text{era}}\\) = średnia wartość target w danej erze
    - \\(\\sigma\\) = odchylenie standardowe średnich między erami
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        mean_stability = era_stats["target_mean"].std()
        st.metric(
            "Stabilność (σ średnich)",
            f"{mean_stability:.6f}",
            help="Odchylenie standardowe średnich między"
            "erami - im niższe, tym stabilniejszy target",
        )

    with col2:
        trend = np.polyfit(range(len(era_stats)), era_stats["target_mean"], 1)[0]
        st.metric("Trend", f"{trend:.8f}", help="Nachylenie trendu liniowego")

    with col3:
        range_mean = era_stats["target_mean"].max() - era_stats["target_mean"].min()
        st.metric("Zakres Średnich", f"{range_mean:.6f}")

with tab2:
    st.subheader(f"Zmienność {selected_target.upper()} per Era (Std Dev)")

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
        annotation_text="Średnia Zmienność",
    )

    fig.update_layout(
        xaxis_title="Era", yaxis_title=f"Std Dev {selected_target.upper()}", height=500
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")

    # Volatility periods
    st.markdown("**Okresy Wysokiej Zmienności:**")

    high_vol_threshold = era_stats["target_std"].quantile(0.75)
    high_vol_eras = era_stats[era_stats["target_std"] >= high_vol_threshold]

    st.dataframe(
        high_vol_eras[["era", "target_std", "target_mean"]].sort_values(
            "target_std", ascending=False
        ),
        width="stretch",
    )

with tab3:
    st.subheader(f"Zakres {selected_target.upper()} per Era (Max - Min)")

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
        xaxis_title="Era",
        yaxis_title=f"{selected_target.upper()} Range (Max - Min)",
        height=500,
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")

st.divider()

# ============================================================================
# FEATURE BEHAVIOR OVER TIME
# ============================================================================
st.header("🔍 Zachowanie Cech w Czasie")

# Feature selection
selected_features_time = st.multiselect(
    "Wybierz cechy do analizy temporalnej (max 5):",
    feature_set,
    default=feature_set[:3],
)

if len(selected_features_time) > 5:
    st.warning("Zbyt wiele cech. Wyświetlane będą tylko pierwsze 5.")
    selected_features_time = selected_features_time[:5]

if len(selected_features_time) > 0:
    # Calculate feature stats per era
    feature_era_stats = (
        train.groupby("era")[selected_features_time].agg(["mean", "std"]).reset_index()
    )

    # Mean values over time
    st.subheader("Średnie Wartości Cech per Era")

    fig = go.Figure()

    for feature in selected_features_time:
        fig.add_trace(
            go.Scatter(
                x=feature_era_stats["era"],
                y=feature_era_stats[feature]["mean"],
                mode="lines+markers",
                name=feature,
                hovertemplate=f"{feature}<br>Era: %{{x}}<br>Mean: %{{y:.4f}}",
            )
        )

    fig.update_layout(
        title="Średnie wartości cech w czasie",
        xaxis_title="Era",
        yaxis_title="Średnia Wartość",
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

    # Correlation stability
    st.subheader(f"Stabilność Korelacji z {selected_target.upper()}")

    # Calculate correlation per era for each feature
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
                hovertemplate=f"{feature}<br>Era: %{{x}}<br>Corr: %{{y:.6f}}",
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"Korelacja z {selected_target.upper()} (zwrot z akcji) w czasie",
        xaxis_title="Era",
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

    # Correlation stability metrics
    st.markdown("### 📐 Wzór Matematyczny - Stabilność Korelacji:")

    st.latex(r"""
    \text{Stabilność}_{corr} = \sigma(\rho_{\text{era}})
    """)

    st.markdown("""
    gdzie:

    - \\(\\rho_{\\text{era}}\\) = korelacja między cechą a target w danej erze
    - \\(\\sigma\\) = odchylenie standardowe korelacji między erami
    """)

    st.markdown("**Metryki Stabilności Korelacji:**")

    stability_metrics = []

    for feature in selected_features_time:
        corr_values = corr_time_df[feature]

        # Ensure selected_target is a single, hashable column name
        if isinstance(selected_target, list | tuple):
            # If no target is selected, skip this feature to avoid invalid indexing
            if not selected_target:
                continue
            target_col = selected_target[0]
        else:
            target_col = selected_target
        global_corr = train[[feature, target_col]].corr().iloc[0, 1]

        stability_metrics.append(
            {
                "Cecha": feature,
                "Korelacja_Globalna": global_corr,
                "Średnia_Korelacja_per_Era": corr_values.mean(),
                "Stabilność_σ(ρ)": corr_values.std(),
                "Min_Korelacja": corr_values.min(),
                "Max_Korelacja": corr_values.max(),
                "Sign_Changes": (
                    corr_values[:-1].values * corr_values[1:].values < 0
                ).sum(),
            }
        )

    stability_df = pd.DataFrame(stability_metrics)

    st.dataframe(
        stability_df.style.background_gradient(subset=["Stabilność_σ(ρ)"], cmap="Reds"),
        width="stretch",
    )

    st.info("""
    **Interpretacja:**

    - **Niska Stabilność σ(ρ)**: Stabilna relacja z target ✅
    - **Wysoka Stabilność σ(ρ)**: Niestabilna, zależna od czasu ⚠️
    - **Sign Changes**: Liczba zmian znaku korelacji (niestabilność) ❌
    """)

st.divider()

# ============================================================================
# ROLLING STATISTICS
# ============================================================================
st.header("📉 Statystyki Kroczące (Rolling)")

window_size = st.slider("Rozmiar okna:", 3, 20, 5)

# Calculate rolling statistics
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
        f"Średnia {selected_target.upper()} (okno={window_size})",
        f"Std Dev {selected_target.upper()} (okno={window_size})",
    ),
    shared_xaxes=True,
    vertical_spacing=0.12,
)

# Mean
fig.add_trace(
    go.Scatter(
        x=era_stats["era"],
        y=era_stats["target_mean"],
        mode="markers",
        name="Rzeczywiste",
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

# Std
fig.add_trace(
    go.Scatter(
        x=era_stats["era"],
        y=era_stats["target_std"],
        mode="markers",
        name="Rzeczywiste Std",
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

fig.update_xaxes(title_text="Era", row=2, col=1, tickangle=45)
fig.update_yaxes(title_text=f"{selected_target.upper()} Mean", row=1, col=1)
fig.update_yaxes(title_text=f"{selected_target.upper()} Std", row=2, col=1)

fig.update_layout(height=700, hovermode="x unified")
st.plotly_chart(fig, width="stretch")

st.divider()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown(
    f"---\n"
    f"**📊 Aktualnie Analizuję:** {selected_target}\n"
    f"\n"
    f"**💡 Wskazówka:** Zmień target w sidebaru aby "
    f"zobaczyć jak zmieniają się wszystkie wykresy"
)
