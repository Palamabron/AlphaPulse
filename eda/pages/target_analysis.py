"""Target Analysis Page — Comprehensive target variable exploration."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Analiza Target", page_icon="🎯", layout="wide")

if "data_loaded" not in st.session_state:
    st.warning("⚠️ Dane nie zostały załadowane. Przejdź do strony głównej.")
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]
all_targets = st.session_state.get("all_targets", [])

st.title("🎯 Analiza Zmiennej Target")
st.markdown("""
Szczegółowa analiza zmiennej docelowej oraz jej zachowania w czasie.

**Target** = Znormalizowany zwrot z akcji (wartość do przewidzenia w turnieju Numerai).

""")

st.sidebar.header("⚙️ Ustawienia Analizy")

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

if selected_target not in train.columns:
    st.error(f"❌ Target '{selected_target}' nie istnieje w danych!")
    st.stop()

st.info(f"📊 Analizowany Target: **{selected_target}** (znormalizowany zwrot z akcji)")

st.divider()

st.header("📊 Statystyki Target")

target_stats = train[selected_target].describe()
skewness = train[selected_target].skew()
kurtosis = train[selected_target].kurtosis()

col_mean_std, col_median_iqr, col_min_max, col_q1_q3, col_skew_kurtosis = st.columns(5)

with col_mean_std:
    st.metric("Średnia", f"{target_stats['mean']:.6f}")
    st.metric("Std Dev", f"{target_stats['std']:.6f}")

with col_median_iqr:
    st.metric("Mediana", f"{target_stats['50%']:.6f}")
    st.metric("IQR", f"{(target_stats['75%'] - target_stats['25%']):.6f}")

with col_min_max:
    st.metric("Min", f"{target_stats['min']:.6f}")
    st.metric("Max", f"{target_stats['max']:.6f}")

with col_q1_q3:
    st.metric("25% Percentyl", f"{target_stats['25%']:.6f}")
    st.metric("75% Percentyl", f"{target_stats['75%']:.6f}")

with col_skew_kurtosis:
    st.metric("Skewness", f"{skewness:.6f}")
    st.metric("Kurtosis", f"{kurtosis:.6f}")

st.divider()

st.header("📈 Ridgeplot - Rozkład Target dla wszystkich Er")

st.markdown("""
Ridgeplot pokazuje rozkład wartości Target dla każdej Ery na jednym wykresie.

Umożliwia wizualną ocenę stabilności rozkładu w czasie.

""")

era_sample_rate = st.slider(
    "Co która Era na wykresie:", 1, 10, 3, key="ridgeplot_sample"
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
    title=(
        f"Ridgeplot - Rozkład {selected_target} per Era "
        f"(co {era_sample_rate} Era)"
    ),
    xaxis_title=f"{selected_target} (znormalizowany zwrot z akcji)",
    yaxis_title="Era",
    height=max(600, len(sample_eras) * 30),
    yaxis={"categoryorder": "category ascending", "tickmode": "linear"},
    hovermode="closest",
)

st.plotly_chart(fig, width="stretch")

st.divider()

st.header("⏰ Wykres Era vs Target")

st.markdown("""
Wykres przedstawia wartości Target w funkcji Ery (oś X: Era, oś Y: Target).

""")

show_sample = st.checkbox(
    "Pokaż próbkę danych (szybsze renderowanie)", value=True
)

if show_sample:
    plot_data = train.sample(min(10000, len(train)), random_state=42)
else:
    plot_data = train

fig = px.scatter(
    plot_data,
    x="era",
    y=selected_target,
    title=(
        f"Era vs {selected_target}"
        + ("(próbka 10k punktów)" if show_sample else "(wszystkie punkty)")
    ),
    labels={
        "era": "Era",
        selected_target: f"{selected_target} (zwrot z akcji)",
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
        name="Średnia per Era",
        line={"color": "red", "width": 2},
    )
)

fig.add_hline(
    y=train[selected_target].mean(),
    line_dash="dash",
    line_color="green",
    annotation_text="Średnia globalna",
)

fig.update_layout(
    xaxis_title="Era",
    yaxis_title=f"{selected_target} (znormalizowany zwrot z akcji)",
    height=600,
    hovermode="closest",
)

fig.update_xaxes(tickangle=45)
st.plotly_chart(fig, width="stretch")

st.divider()

st.header("⏰ Analiza Target przez Ery")

try:
    era_stats_full = (
        train.groupby("era")
        .agg(
            {selected_target: ["mean", "std", "min", "max"], "era": "size"}
        )
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
    st.error(f"❌ Błąd przy obliczaniu statystyk: {e}")
    st.stop()

col_mean_std, col_median_iqr = st.columns(2)

with col_mean_std:
    st.subheader(f"Średnia {selected_target} per Era")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=era_stats_full["era"],
            y=era_stats_full["target_mean"],
            mode="lines+markers",
            name="Średnia",
            line={"color": "blue", "width": 2},
            marker={"size": 6},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=era_stats_full["era"],
            y=era_stats_full["target_mean"]
            + era_stats_full["target_std"],
            mode="lines",
            name="Mean + Std",
            line={"width": 0},
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=era_stats_full["era"],
            y=era_stats_full["target_mean"]
            - era_stats_full["target_std"],
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
        annotation_text="Średnia globalna",
        line_color="red",
    )

    fig.update_layout(
        xaxis_title="Era",
        yaxis_title=f"Średnia {selected_target}",
        height=400,
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width="stretch")

with col_median_iqr:
    st.subheader(f"Zmienność {selected_target} per Era")
    fig = px.line(
        era_stats_full,
        x="era",
        y="target_std",
        title=f"Odchylenie standardowe {selected_target}",
        labels={"era": "Era", "target_std": "Std Dev"},
        markers=True,
    )

    fig.add_hline(
        y=era_stats_full["target_std"].mean(),
        line_dash="dash",
        annotation_text="Średnia zmienność",
        line_color="orange",
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400)
    fig.update_yaxes(title_text=f"Std Dev {selected_target}")
    st.plotly_chart(fig, width="stretch")

st.subheader("Kompleksowa analiza w czasie")

fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=(
        f"Średnia {selected_target}",
        "Odchylenie Standardowe",
        "Zakres (Max - Min)",
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

era_stats_full["range"] = (
    era_stats_full["target_max"] - era_stats_full["target_min"]
)
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

fig.update_xaxes(title_text="Era", row=3, col=1, tickangle=45)
fig.update_yaxes(title_text="Mean", row=1, col=1)
fig.update_yaxes(title_text="Std Dev", row=2, col=1)
fig.update_yaxes(title_text="Range", row=3, col=1)
fig.update_layout(height=900, showlegend=False)
st.plotly_chart(fig, width="stretch")

st.divider()

st.header("🎲 Metryki Stabilności Target")

col_mean_std, col_median_iqr, col_min_max = st.columns(3)

with col_mean_std:
    cv = (
        era_stats_full["target_mean"].std()
        / era_stats_full["target_mean"].mean()
    ) * 100
    st.metric(
        "Coefficient of Variation",
        f"{cv:.2f}%",
        help="Zmienność średniej target między erami",
    )

with col_median_iqr:
    mean_std = era_stats_full["target_std"].mean()
    st.metric(
        "Średnia Zmienność per Era",
        f"{mean_std:.6f}",
        help="Średnie odchylenie standardowe w erach",
    )

with col_min_max:
    range_variability = era_stats_full["range"].std()
    st.metric(
        "Zmienność Zakresu",
        f"{range_variability:.6f}",
        help="Jak bardzo zmienia się zakres między erami",
    )

st.subheader("Mapa stabilności Target")

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
        f"Średnia {selected_target} (rolling window={window})",
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
        name="Actual Mean",
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
        name="Rolling Mean",
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
        name="Actual Std",
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
        name="Rolling Std",
        line={"color": "red", "width": 3},
    ),
    row=2,
    col=1,
)

fig.update_xaxes(title_text="Era", row=2, col=1, tickangle=45)
fig.update_yaxes(
    title_text=f"{selected_target} Mean", row=1, col=1
)
fig.update_yaxes(
    title_text=f"{selected_target} Std", row=2, col=1
)
fig.update_layout(height=700, showlegend=True)
st.plotly_chart(fig, width="stretch")

st.divider()
