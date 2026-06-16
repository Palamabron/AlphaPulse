"""HPO Results Analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eda.utils import get_translations

st.set_page_config(page_title="HPO Analysis", page_icon="🔬", layout="wide")

t = get_translations()

st.title(t["hpo.title"])
st.markdown(t["hpo.description"])

st.sidebar.header(t["hpo.data_source"])
trials_path = st.sidebar.text_input(
    t["hpo.path_label"],
    value="artifacts/hpo/all_trials.json",
    help=t["hpo.path_help"],
)
min_sharpe = st.sidebar.slider(
    t["hpo.min_sharpe_label"],
    min_value=-10.0,
    max_value=5.0,
    value=-5.0,
    step=0.1,
)


@st.cache_data
def load_trials(path: str, _min_sharpe: float) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for trial in raw:
        if trial.get("error"):
            continue
        params = trial.get("params", {})
        metrics = trial.get("metrics", {})
        num = params.get("num_models", 1)
        model_types = "+".join(
            str(params.get(f"model_{i}_type", "?")) for i in range(1, num + 1)
        )
        row: dict = {
            "trial": trial["trial"],
            "sharpe": trial["sharpe"],
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
            "elapsed_seconds": trial.get("elapsed_seconds", 0.0),
            "mean_era_corr": float(metrics.get("mean_per_era_correlation", 0.0)),
            "std_era_corr": float(metrics.get("std_per_era_correlation", 0.0)),
            "corr_sharpe": float(metrics.get("corr_sharpe", trial["sharpe"])),
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
    st.warning(t.format("hpo.file_not_found", path=trials_path))
    st.stop()
except Exception as exc:
    st.error(t.format("hpo.error_loading", error=exc))
    st.stop()

if df.empty:
    st.warning(t["hpo.no_valid_trials"])
    st.stop()

st.header(t["hpo.summary"])
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(t["hpo.num_trials"], len(df))
c2.metric(t["hpo.best_sharpe"], f"{df['sharpe'].max():.4f}")
c3.metric(t["hpo.median_sharpe"], f"{df['sharpe'].median():.4f}")
best_row = df.loc[df["sharpe"].idxmax()]
c4.metric(t["hpo.best_model"], best_row["model_types"])
c5.metric(t["hpo.best_ensemble"], best_row["ensemble_method"])

st.divider()

st.header(t["hpo.trial_history"])
running_best = df.sort_values("trial")["sharpe"].cummax()
fig_history = go.Figure()
fig_history.add_trace(
    go.Scatter(
        x=df["trial"],
        y=df["sharpe"],
        mode="markers",
        name=t["hpo.trial_sharpe"],
        marker={
            "color": df["sharpe"],
            "colorscale": "Viridis",
            "size": 6,
            "showscale": True,
        },
        text=df["model_types"],
        hovertemplate=t["hpo.trial_hover"],
    )
)
fig_history.add_trace(
    go.Scatter(
        x=df.sort_values("trial")["trial"],
        y=running_best.values,
        mode="lines",
        name=t["hpo.best_sharpe_cumulative"],
        line={"color": "red", "width": 2, "dash": "dash"},
    )
)
fig_history.update_layout(
    xaxis_title=t["hpo.trial_number"],
    yaxis_title="Sharpe",
    height=350,
    legend={"yanchor": "bottom", "y": 0.01, "xanchor": "right", "x": 0.99},
)
st.plotly_chart(fig_history, use_container_width=True)

st.divider()

st.header(t["hpo.model_comparison"])

col_model, col_prep = st.columns(2)

with col_model:
    st.subheader(t["hpo.sharpe_by_model"])
    fig_model = px.violin(
        df,
        x="model_1_type",
        y="sharpe",
        color="model_1_type",
        box=True,
        points="all",
        title=t["hpo.sharpe_distribution"],
        labels={
            "model_1_type": t["hpo.model_type"],
            "sharpe": "Sharpe",
        },
    )
    fig_model.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig_model, use_container_width=True)

with col_prep:
    st.subheader(t["hpo.packboost_impact"])
    df_pack = df.copy()
    with_pb = t["hpo.with_packboost"]
    without_pb = t["hpo.without_packboost"]
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
        title=t["hpo.sharpe_with_without_packboost"],
        labels={"sharpe": "Sharpe"},
        color_discrete_map={with_pb: "#2196F3", without_pb: "#9E9E9E"},
    )
    fig_pack.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig_pack, use_container_width=True)

col_ens, col_neut = st.columns(2)

with col_ens:
    st.subheader(t["hpo.ensemble_method"])
    fig_ens = px.box(
        df,
        x="ensemble_method",
        y="sharpe",
        color="ensemble_method",
        points="all",
        title=t["hpo.sharpe_by_ensemble"],
        labels={
            "ensemble_method": t["hpo.method"],
            "sharpe": "Sharpe",
        },
    )
    fig_ens.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_ens, use_container_width=True)

with col_neut:
    st.subheader(t["hpo.neutralization_impact"])
    df_neut = df.copy()
    with_neut = t["hpo.with_neutralization"]
    without_neut = t["hpo.without_neutralization"]
    df_neut["Neutralization"] = df_neut["use_neutralization"].map(
        {True: with_neut, False: without_neut}
    )
    fig_neut = px.box(
        df_neut,
        x="Neutralization",
        y="sharpe",
        color="Neutralization",
        points="all",
        title=t["hpo.sharpe_with_without_neutralization"],
        labels={"sharpe": "Sharpe"},
        color_discrete_map={with_neut: "#4CAF50", without_neut: "#9E9E9E"},
    )
    fig_neut.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_neut, use_container_width=True)

st.divider()

st.header(t["hpo.nsubs_impact"])
fig_nsubs = px.strip(
    df,
    x="n_subs",
    y="sharpe",
    color="model_1_type",
    title=t["hpo.sharpe_vs_nsubs"],
    labels={"n_subs": "n_subs", "sharpe": "Sharpe", "model_1_type": "Model"},
)
mean_nsubs = df.groupby("n_subs")["sharpe"].mean().reset_index()
fig_nsubs.add_trace(
    go.Scatter(
        x=mean_nsubs["n_subs"],
        y=mean_nsubs["sharpe"],
        mode="lines+markers",
        name=t["hpo.average"],
        line={"color": "black", "width": 2},
        marker={"size": 8, "symbol": "diamond"},
    )
)
fig_nsubs.update_layout(height=380)
st.plotly_chart(fig_nsubs, use_container_width=True)

st.divider()

st.header(t["hpo.hyperparams_impact"])

param_tabs = st.tabs(["XGBoost", "LightGBM"])

with param_tabs[0]:
    xgb_df = df[df["model_1_type"] == "XGBoost"].dropna(subset=["xgb_learning_rate"])
    if xgb_df.empty:
        st.info(t["hpo.no_xgboost"])
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
        st.info(t["hpo.no_lightgbm"])
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

st.header(t["hpo.parallel_categories"])
st.markdown(t["hpo.parallel_description"])

cat_dims = ["model_1_type", "scaler_type", "ensemble_method"]
df_par = df.copy()
yes_text = t["hpo.yes"]
no_text = t["hpo.no"]
df_par["neutralization"] = df_par["use_neutralization"].map(
    {True: yes_text, False: no_text}
)
df_par["packboost"] = df_par["use_packboost"].map({True: yes_text, False: no_text})

q75 = df_par["sharpe"].quantile(0.75)
top_label = t["hpo.top_25"]
rest_label = t["hpo.remaining"]
df_par["sharpe_tier"] = (df_par["sharpe"] >= q75).map(
    {True: top_label, False: rest_label}
)

fig_par = px.parallel_categories(
    df_par,
    dimensions=["model_1_type", "packboost", "ensemble_method", "neutralization"],
    color="sharpe",
    color_continuous_scale="RdYlGn",
    title=t["hpo.parallel_title"],
    labels={
        "model_1_type": "Model",
        "packboost": "Packboost",
        "ensemble_method": "Ensemble",
        "neutralization": t["hpo.neutralization"],
    },
)
fig_par.update_layout(height=450)
st.plotly_chart(fig_par, use_container_width=True)

st.divider()

st.header(t["hpo.metrics_correlation"])
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
    title=t["hpo.correlation_matrix"],
    aspect="auto",
)
fig_corr.update_layout(height=420)
st.plotly_chart(fig_corr, use_container_width=True)

st.divider()

st.header(t["hpo.leaderboard"])
top_n = st.slider(
    t["hpo.num_trials_display"],
    min_value=5,
    max_value=min(100, len(df)),
    value=20,
)
rank_col = (
    "payout_score"
    if "payout_score" in df.columns and df["payout_score"].notna().any()
    else "sharpe"
)
show_cols = [
    "trial",
    rank_col,
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
if rank_col == "payout_score":
    show_cols.insert(2, "corr_sharpe" if "corr_sharpe" in df.columns else "sharpe")

leaderboard = df.nlargest(top_n, rank_col)[show_cols]
st.dataframe(
    leaderboard.style.background_gradient(subset=[rank_col], cmap="RdYlGn"),
    use_container_width=True,
    height=420,
)

csv = leaderboard.to_csv(index=False).encode("utf-8")
st.download_button(
    label=t["hpo.download_leaderboard"],
    data=csv,
    file_name="hpo_leaderboard.csv",
    mime="text/csv",
)
