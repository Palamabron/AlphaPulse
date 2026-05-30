"""HPO Results Analysis — Analiza wyników optymalizacji hiperparametrów."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eda.utils import translate

st.set_page_config(page_title="HPO Analysis", page_icon="🔬", layout="wide")

lang = st.session_state.get("lang", "English")

st.title(translate("🔬 HPO Results Analysis", "🔬 Analiza wyników HPO"))
st.markdown(
    translate(
        "Load `all_trials.json` from the `artifacts/` directory to analyze "
        "hyperparameter optimization results.",
        "Załaduj plik `all_trials.json` z katalogu `artifacts/` aby przeanalizować "
        "wyniki przeszukiwania hiperparametrów.",
    )
)

# ---------------------------------------------------------------------------
# Sidebar — file picker
# ---------------------------------------------------------------------------
st.sidebar.header(translate("📁 Data Source", "📁 Źródło danych"))
trials_path = st.sidebar.text_input(
    translate("Path to all_trials.json", "Ścieżka do all_trials.json"),
    value="artifacts/hpo/all_trials.json",
    help=translate(
        "Relative or absolute path to all_trials.json file.",
        "Względna lub bezwzględna ścieżka do pliku all_trials.json.",
    ),
)
min_sharpe = st.sidebar.slider(
    translate(
        "Min. sharpe (filter invalid trials)",
        "Min. sharpe (filtr błędnych prób)",
    ),
    min_value=-10.0,
    max_value=5.0,
    value=-5.0,
    step=0.1,
)


# ---------------------------------------------------------------------------
# Data loading & flattening
# ---------------------------------------------------------------------------
@st.cache_data
def load_trials(path: str, _min_sharpe: float) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for t in raw:
        if t.get("error"):
            continue
        params = t.get("params", {})
        metrics = t.get("metrics", {})
        num = params.get("num_models", 1)
        model_types = "+".join(
            str(params.get(f"model_{i}_type", "?")) for i in range(1, num + 1)
        )
        row: dict = {
            "trial": t["trial"],
            "sharpe": t["sharpe"],
            "model_types": model_types,
            "model_1_type": params.get("model_1_type", "?"),
            "num_models": num,
            "scaler_type": params.get("scaler_type", "?"),
            "use_packboost": params.get("use_packboost", False),
            "n_subs": params.get("n_subs", 10),
            "ensemble_method": params.get("ensemble_method", "single"),
            "use_neutralization": params.get("use_neutralization", False),
            "neutralization_proportion": params.get("neutralization_proportion", 0.0),
            "xgb_max_depth": params.get("xgb_max_depth"),
            "xgb_learning_rate": params.get("xgb_learning_rate"),
            "xgb_n_rounds": params.get("xgb_n_rounds"),
            "lgbm_num_leaves": params.get("lgbm_num_leaves"),
            "lgbm_learning_rate": params.get("lgbm_learning_rate"),
            "lgbm_n_rounds": params.get("lgbm_n_rounds"),
            "elapsed_seconds": t.get("elapsed_seconds", 0.0),
            "mean_era_corr": float(metrics.get("mean_per_era_correlation", 0.0)),
            "std_era_corr": float(metrics.get("std_per_era_correlation", 0.0)),
            "corr_sharpe": float(metrics.get("corr_sharpe", t["sharpe"])),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "pct_positive_eras": float(metrics.get("pct_positive_eras", 0.0)),
            "mmc_sharpe": metrics.get("mmc_sharpe"),
            "payout_score": metrics.get("payout_score"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[df["sharpe"] >= _min_sharpe].reset_index(drop=True)


try:
    df = load_trials(trials_path, min_sharpe)
except FileNotFoundError:
    st.warning(
        translate(
            f"File not found: `{trials_path}`. Provide a valid path in the sidebar.",
            f"Plik nie znaleziony: `{trials_path}`. "
            "Podaj poprawną ścieżkę w panelu bocznym.",
        )
    )
    st.stop()
except Exception as exc:
    st.error(
        translate(
            f"Error loading data: {exc}",
            f"Błąd ładowania danych: {exc}",
        )
    )
    st.stop()

if df.empty:
    st.warning(
        translate(
            "No valid trials after filtering.",
            "Brak poprawnych prób po filtracji.",
        )
    )
    st.stop()

# ---------------------------------------------------------------------------
# Summary KPIs
# ---------------------------------------------------------------------------
st.header(translate("📊 Summary", "📊 Podsumowanie"))
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(translate("Number of trials", "Liczba prób"), len(df))
c2.metric(translate("Best Sharpe", "Najlepszy Sharpe"), f"{df['sharpe'].max():.4f}")
c3.metric(translate("Median Sharpe", "Mediana Sharpe"), f"{df['sharpe'].median():.4f}")
best_row = df.loc[df["sharpe"].idxmax()]
c4.metric(translate("Best model", "Najlepszy model"), best_row["model_types"])
c5.metric(
    translate("Best ensemble method", "Najlepsza metoda ensemble"),
    best_row["ensemble_method"],
)

st.divider()

# ---------------------------------------------------------------------------
# Trial history
# ---------------------------------------------------------------------------
st.header(translate("📈 Trial History", "📈 Historia prób"))
running_best = df.sort_values("trial")["sharpe"].cummax()
fig_history = go.Figure()
fig_history.add_trace(
    go.Scatter(
        x=df["trial"],
        y=df["sharpe"],
        mode="markers",
        name=translate("Trial Sharpe", "Sharpe próby"),
        marker={
            "color": df["sharpe"],
            "colorscale": "Viridis",
            "size": 6,
            "showscale": True,
        },
        text=df["model_types"],
        hovertemplate=translate(
            "Trial %{x}<br>Sharpe: %{y:.4f}<br>Models: %{text}",
            "Próba %{x}<br>Sharpe: %{y:.4f}<br>Modele: %{text}",
        ),
    )
)
fig_history.add_trace(
    go.Scatter(
        x=df.sort_values("trial")["trial"],
        y=running_best.values,
        mode="lines",
        name=translate("Best Sharpe (cumulative)", "Najlepszy Sharpe (narastająco)"),
        line={"color": "red", "width": 2, "dash": "dash"},
    )
)
fig_history.update_layout(
    xaxis_title=translate("Trial number", "Numer próby"),
    yaxis_title="Sharpe",
    height=350,
    legend={"yanchor": "bottom", "y": 0.01, "xanchor": "right", "x": 0.99},
)
st.plotly_chart(fig_history, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Model type & preprocessor comparison
# ---------------------------------------------------------------------------
st.header(
    translate(
        "🤖 Model Types & Preprocessors Comparison",
        "🤖 Porównanie typów modeli i preprocessorów",
    )
)

col_model, col_prep = st.columns(2)

with col_model:
    st.subheader(translate("Sharpe by model type", "Sharpe wg typu modelu"))
    fig_model = px.violin(
        df,
        x="model_1_type",
        y="sharpe",
        color="model_1_type",
        box=True,
        points="all",
        title=translate(
            "Sharpe distribution for each model type",
            "Rozkład Sharpe dla każdego typu modelu",
        ),
        labels={
            "model_1_type": translate("Model type", "Typ modelu"),
            "sharpe": "Sharpe",
        },
    )
    fig_model.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig_model, use_container_width=True)

with col_prep:
    st.subheader(
        translate("Packboost preprocessor impact", "Wpływ Packboost preprocessora")
    )
    df_pack = df.copy()
    with_pb = translate("With Packboost", "Z Packboost")
    without_pb = translate("Without Packboost", "Bez Packboost")
    df_pack["Packboost"] = df_pack["use_packboost"].map(
        {True: with_pb, False: without_pb}
    )
    fig_pack = px.violin(
        df_pack,
        x="Packboost",
        y="sharpe",
        color="Packboost",
        box=True,
        points="all",
        title=translate(
            "Sharpe with and without Packboost", "Sharpe z i bez Packboost"
        ),
        labels={"sharpe": "Sharpe"},
        color_discrete_map={with_pb: "#2196F3", without_pb: "#9E9E9E"},
    )
    fig_pack.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig_pack, use_container_width=True)

col_ens, col_neut = st.columns(2)

with col_ens:
    st.subheader(translate("Ensemble method", "Metoda ensemble"))
    fig_ens = px.box(
        df,
        x="ensemble_method",
        y="sharpe",
        color="ensemble_method",
        points="all",
        title=translate("Sharpe by ensemble method", "Sharpe wg metody ensemble"),
        labels={
            "ensemble_method": translate("Method", "Metoda"),
            "sharpe": "Sharpe",
        },
    )
    fig_ens.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_ens, use_container_width=True)

with col_neut:
    st.subheader(translate("Neutralization impact", "Wpływ neutralizacji"))
    df_neut = df.copy()
    with_neut = translate("With neutralization", "Z neutralizacją")
    without_neut = translate("Without neutralization", "Bez neutralizacji")
    df_neut["Neutralizacja"] = df_neut["use_neutralization"].map(
        {True: with_neut, False: without_neut}
    )
    fig_neut = px.box(
        df_neut,
        x="Neutralizacja",
        y="sharpe",
        color="Neutralizacja",
        points="all",
        title=translate(
            "Sharpe with and without prediction neutralization",
            "Sharpe z i bez neutralizacji predykcji",
        ),
        labels={"sharpe": "Sharpe"},
        color_discrete_map={with_neut: "#4CAF50", without_neut: "#9E9E9E"},
    )
    fig_neut.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_neut, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Era ensemble size (n_subs)
# ---------------------------------------------------------------------------
st.header(
    translate(
        "🔢 Impact of sub-model count (n_subs)", "🔢 Wpływ liczby sub-modeli (n_subs)"
    )
)
fig_nsubs = px.strip(
    df,
    x="n_subs",
    y="sharpe",
    color="model_1_type",
    title=translate(
        "Sharpe vs. EraEnsemble sub-model count",
        "Sharpe vs. liczba sub-modeli EraEnsemble",
    ),
    labels={"n_subs": "n_subs", "sharpe": "Sharpe", "model_1_type": "Model"},
)
mean_nsubs = df.groupby("n_subs")["sharpe"].mean().reset_index()
fig_nsubs.add_trace(
    go.Scatter(
        x=mean_nsubs["n_subs"],
        y=mean_nsubs["sharpe"],
        mode="lines+markers",
        name=translate("Average", "Średnia"),
        line={"color": "black", "width": 2},
        marker={"size": 8, "symbol": "diamond"},
    )
)
fig_nsubs.update_layout(height=380)
st.plotly_chart(fig_nsubs, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Numeric hyperparameter scatter plots
# ---------------------------------------------------------------------------
st.header(
    translate(
        "📉 Impact of numeric hyperparameters",
        "📉 Wpływ hiperparametrów numerycznych",
    )
)

param_tabs = st.tabs(["XGBoost", "LightGBM"])

with param_tabs[0]:
    xgb_df = df[df["model_1_type"] == "XGBoost"].dropna(subset=["xgb_learning_rate"])
    if xgb_df.empty:
        st.info(translate("No XGBoost trials found.", "Brak prób z modelem XGBoost."))
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                xgb_df,
                x="xgb_learning_rate",
                y="sharpe",
                color="xgb_max_depth",
                log_x=True,
                title="XGBoost: learning_rate vs Sharpe",
                labels={
                    "xgb_learning_rate": "Learning rate (log)",
                    "sharpe": "Sharpe",
                    "xgb_max_depth": "max_depth",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(
                xgb_df,
                x="xgb_n_rounds",
                y="sharpe",
                color="xgb_max_depth",
                title="XGBoost: n_rounds vs Sharpe",
                labels={
                    "xgb_n_rounds": "n_rounds",
                    "sharpe": "Sharpe",
                    "xgb_max_depth": "max_depth",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

with param_tabs[1]:
    lgbm_df = df[df["model_1_type"] == "LightGBM"].dropna(subset=["lgbm_learning_rate"])
    if lgbm_df.empty:
        st.info(translate("No LightGBM trials found.", "Brak prób z modelem LightGBM."))
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                lgbm_df,
                x="lgbm_learning_rate",
                y="sharpe",
                color="lgbm_num_leaves",
                log_x=True,
                title="LightGBM: learning_rate vs Sharpe",
                labels={
                    "lgbm_learning_rate": "Learning rate (log)",
                    "sharpe": "Sharpe",
                    "lgbm_num_leaves": "num_leaves",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(
                lgbm_df,
                x="lgbm_num_leaves",
                y="sharpe",
                color="lgbm_learning_rate",
                title="LightGBM: num_leaves vs Sharpe",
                labels={
                    "lgbm_num_leaves": "num_leaves",
                    "sharpe": "Sharpe",
                    "lgbm_learning_rate": "learning_rate",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Parallel categories (categorical choices → sharpe)
# ---------------------------------------------------------------------------
st.header(
    translate("🔗 Parallel Categories Diagram", "🔗 Równoległy wykres kategoryczny")
)
st.markdown(
    translate(
        "Sankey-style diagram showing configuration flows to best Sharpe. "
        "Path width reflects number of trials.",
        "Wykres typu Sankey pokazujący przepływ konfiguracji do najlepszych Sharpe. "
        "Szerokość ścieżki odzwierciedla liczbę prób.",
    )
)

cat_dims = ["model_1_type", "scaler_type", "ensemble_method"]
df_par = df.copy()
yes_text = translate("yes", "tak")
no_text = translate("no", "nie")
df_par["neutralizacja"] = df_par["use_neutralization"].map(
    {True: yes_text, False: no_text}
)
df_par["packboost"] = df_par["use_packboost"].map({True: yes_text, False: no_text})

# Quantile-based color: top 25% = green
q75 = df_par["sharpe"].quantile(0.75)
top_label = translate("Top 25%", "Top 25%")
rest_label = translate("Remaining", "Pozostałe")
df_par["sharpe_tier"] = (df_par["sharpe"] >= q75).map(
    {True: top_label, False: rest_label}
)

fig_par = px.parallel_categories(
    df_par,
    dimensions=["model_1_type", "packboost", "ensemble_method", "neutralizacja"],
    color="sharpe",
    color_continuous_scale="RdYlGn",
    title=translate(
        "Parallel categories: configurations → Sharpe",
        "Równoległy wykres kategoryczny: konfiguracje → Sharpe",
    ),
    labels={
        "model_1_type": "Model",
        "packboost": "Packboost",
        "ensemble_method": "Ensemble",
        "neutralizacja": translate("Neutralization", "Neutralizacja"),
    },
)
fig_par.update_layout(height=450)
st.plotly_chart(fig_par, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Metrics correlation heatmap
# ---------------------------------------------------------------------------
st.header(
    translate("📐 Metrics Correlation with Sharpe", "📐 Korelacja metryk z Sharpe")
)
numeric_cols = [
    "sharpe",
    "mean_era_corr",
    "std_era_corr",
    "max_drawdown",
    "pct_positive_eras",
    "elapsed_seconds",
]
if "payout_score" in df.columns and df["payout_score"].notna().any():
    numeric_cols.append("payout_score")
if "mmc_sharpe" in df.columns and df["mmc_sharpe"].notna().any():
    numeric_cols.append("mmc_sharpe")

corr_matrix = df[numeric_cols].corr()
fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    title=translate("Metrics correlation matrix", "Macierz korelacji metryk"),
    aspect="auto",
)
fig_corr.update_layout(height=420)
st.plotly_chart(fig_corr, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Leaderboard table
# ---------------------------------------------------------------------------
st.header(
    translate("🏆 Results Leaderboard — top trials", "🏆 Tabela wyników — top próby")
)
top_n = st.slider(
    translate("Number of trials to display", "Liczba wyświetlanych prób"),
    min_value=5,
    max_value=min(100, len(df)),
    value=20,
)
show_cols = [
    "trial",
    "sharpe",
    "mean_era_corr",
    "std_era_corr",
    "max_drawdown",
    "pct_positive_eras",
    "model_types",
    "ensemble_method",
    "use_packboost",
    "use_neutralization",
    "elapsed_seconds",
]
if "payout_score" in df.columns:
    show_cols.insert(2, "payout_score")

leaderboard = df.nlargest(top_n, "sharpe")[show_cols]
st.dataframe(
    leaderboard.style.background_gradient(subset=["sharpe"], cmap="RdYlGn"),
    use_container_width=True,
    height=420,
)

csv = leaderboard.to_csv(index=False).encode("utf-8")
st.download_button(
    label=translate(
        "📥 Download leaderboard as CSV", "📥 Pobierz leaderboard jako CSV"
    ),
    data=csv,
    file_name="hpo_leaderboard.csv",
    mime="text/csv",
)
