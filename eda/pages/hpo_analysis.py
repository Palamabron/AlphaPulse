"""HPO results and robustness analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eda.utils import get_translations
from eda.utils.hpo import (
    RANKING_METRICS,
    discover_latest_trials,
    load_hpo_trials,
    rank_trials,
    recipe_summary,
)

PAGE_ICON = "🔬"
COLORS = {
    "primary": "#2563EB",
    "accent": "#14B8A6",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "muted": "#64748B",
}

st.set_page_config(page_title="AlphaPulse HPO", page_icon=PAGE_ICON, layout="wide")
translations = get_translations()


class HpoTranslations:
    def get(self, key: str, default: str) -> str:
        professional_key = key.replace("hpo.", "hpo_professional.", 1)
        translated = translations.get(professional_key)
        if translated is not None:
            return str(translated)
        return str(translations.get(key, default))


t = HpoTranslations()


@st.cache_data(show_spinner=False)
def cached_load_trials(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return load_hpo_trials(path)


def chart_layout(fig: go.Figure, *, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 16, "r": 16, "t": 54, "b": 16},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        hoverlabel={"namelength": -1},
    )
    return fig


def metric_value(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


latest_path = discover_latest_trials()
default_path = str(latest_path or Path("artifacts/hpo/all_trials.json"))

st.title(t.get("hpo.title", "AlphaPulse HPO"))
st.caption(
    t.get(
        "hpo.description",
        "Robustness, model recipes, and search progression across completed trials.",
    )
)

with st.sidebar:
    st.header(t.get("hpo.controls", "Analysis controls"))
    trials_path = st.text_input(
        t.get("hpo.path_label", "Trials file"),
        value=default_path,
        help=t.get(
            "hpo.path_help",
            "Path to an all_trials.json artifact. The newest artifact is "
            "selected by default.",
        ),
    )
    ranking_labels = {
        t.get("hpo.rank_holdout", "Holdout Sharpe · robust winner"): "holdout",
        t.get("hpo.rank_validation", "Validation payout · HPO objective"): "validation",
        t.get("hpo.rank_robust", "Robust payout · transfer-aware"): "robust",
    }
    ranking_label = st.selectbox(
        t.get("hpo.ranking_label", "Rank trials by"),
        list(ranking_labels),
    )
    ranking = ranking_labels[ranking_label]
    min_holdout = st.number_input(
        t.get("hpo.min_holdout_label", "Minimum holdout Sharpe"),
        value=-5.0,
        step=0.1,
        help=t.get(
            "hpo.min_holdout_help",
            "Applied to tables and comparisons, not to the search-progression chart.",
        ),
    )

source = Path(trials_path)
try:
    df_all = cached_load_trials(str(source), source.stat().st_mtime_ns)
except FileNotFoundError:
    st.warning(t.get("hpo.file_not_found", "Trials file not found."))
    st.stop()
except (OSError, ValueError, json.JSONDecodeError) as exc:
    st.error(
        t.get("hpo.error_loading", "Could not load trials: {error}").format(error=exc)
    )
    st.stop()

if df_all.empty:
    st.warning(t.get("hpo.no_valid_trials", "No completed, finite trials were found."))
    st.stop()

df = df_all[df_all["holdout_corr_sharpe"] >= min_holdout].copy()
ranked = rank_trials(df, ranking)
if ranked.empty:
    st.warning(t.get("hpo.no_matching_trials", "No trials match the current filter."))
    st.stop()

rank_metric = RANKING_METRICS[ranking]
best = ranked.iloc[0]
best_holdout = rank_trials(df_all, "holdout").iloc[0]
best_validation = rank_trials(df_all, "validation")
validation_winner = best_validation.iloc[0] if not best_validation.empty else None

metric_cols = st.columns(6)
metric_cols[0].metric(t.get("hpo.completed_trials", "Completed trials"), len(df_all))
metric_cols[1].metric(
    t.get("hpo.selected_trial", "Selected trial"), f"#{int(best['trial'])}"
)
metric_cols[2].metric(
    t.get("hpo.holdout_sharpe", "Holdout Sharpe"),
    metric_value(best["holdout_corr_sharpe"]),
)
metric_cols[3].metric(
    t.get("hpo.validation_payout", "Validation payout"),
    metric_value(best["payout_score"]),
)
metric_cols[4].metric(
    t.get("hpo.positive_eras", "Positive holdout eras"),
    f"{float(best['pct_positive_eras']):.1%}"
    if pd.notna(best["pct_positive_eras"])
    else "—",
)
metric_cols[5].metric(
    t.get("hpo.max_drawdown", "Max drawdown"),
    metric_value(best["max_drawdown"]),
)

transfer_df = df_all.dropna(subset=["payout_score", "holdout_corr_sharpe"])
transfer_corr = (
    transfer_df["payout_score"].corr(transfer_df["holdout_corr_sharpe"])
    if len(transfer_df) >= 2
    else None
)
if validation_winner is not None and int(validation_winner["trial"]) != int(
    best_holdout["trial"]
):
    st.warning(
        t.get(
            "hpo.winner_divergence",
            "Validation and holdout select different winners. Trial #{holdout_trial} "
            "leads holdout Sharpe ({holdout:.3f}); trial #{validation_trial} leads "
            "validation payout ({payout:.3f}) but has holdout Sharpe "
            "{validation_holdout:.3f}.",
        ).format(
            holdout_trial=int(best_holdout["trial"]),
            holdout=best_holdout["holdout_corr_sharpe"],
            validation_trial=int(validation_winner["trial"]),
            payout=validation_winner["payout_score"],
            validation_holdout=validation_winner["holdout_corr_sharpe"],
        )
    )

overview_tab, recipes_tab, progression_tab, leaderboard_tab = st.tabs(
    [
        t.get("hpo.tab_robustness", "Robustness"),
        t.get("hpo.tab_recipes", "Recipes"),
        t.get("hpo.tab_progression", "Search progression"),
        t.get("hpo.tab_leaderboard", "Leaderboard & config"),
    ]
)

with overview_tab:
    left, right = st.columns([1.35, 1])
    with left:
        fig_transfer = px.scatter(
            transfer_df,
            x="payout_score",
            y="holdout_corr_sharpe",
            color="num_models",
            size="pct_positive_eras",
            hover_name="recipe",
            hover_data={
                "trial": True,
                "val_corr_sharpe": ":.3f",
                "payout_score": ":.3f",
                "holdout_corr_sharpe": ":.3f",
                "num_models": False,
                "pct_positive_eras": ":.1%",
            },
            color_continuous_scale=["#94A3B8", COLORS["primary"]],
            title=t.get(
                "hpo.transfer_title",
                "Validation payout does not guarantee holdout performance",
            ),
            labels={
                "payout_score": t.get("hpo.validation_payout", "Validation payout"),
                "holdout_corr_sharpe": t.get("hpo.holdout_sharpe", "Holdout Sharpe"),
                "num_models": t.get("hpo.model_count", "Models"),
            },
        )
        fig_transfer.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"])
        if validation_winner is not None:
            fig_transfer.add_annotation(
                x=validation_winner["payout_score"],
                y=validation_winner["holdout_corr_sharpe"],
                text=t.get("hpo.validation_winner", "Validation winner"),
                showarrow=True,
                arrowcolor=COLORS["warning"],
            )
        fig_transfer.add_annotation(
            x=best_holdout["payout_score"],
            y=best_holdout["holdout_corr_sharpe"],
            text=t.get("hpo.holdout_winner", "Holdout winner"),
            showarrow=True,
            arrowcolor=COLORS["accent"],
        )
        st.plotly_chart(chart_layout(fig_transfer, height=470), width="stretch")
        st.caption(
            t.get(
                "hpo.transfer_correlation",
                "Validation payout ↔ holdout Sharpe correlation: {value}.",
            ).format(value=metric_value(transfer_corr, 2))
        )

    with right:
        st.subheader(t.get("hpo.selected_recipe", "Selected recipe"))
        st.markdown(f"### Trial #{int(best['trial'])}")
        st.write(best["recipe"])
        detail_rows = pd.DataFrame(
            {
                t.get("hpo.metric", "Metric"): [
                    t.get("hpo.mean_era_corr", "Mean era correlation"),
                    t.get("hpo.std_era_corr", "Era correlation volatility"),
                    t.get("hpo.robust_payout", "Robust payout"),
                    t.get("hpo.elapsed", "Runtime"),
                    t.get("hpo.features", "Routed features"),
                ],
                t.get("hpo.value", "Value"): [
                    metric_value(best["mean_era_corr"], 4),
                    metric_value(best["std_era_corr"], 4),
                    metric_value(best["robust_payout_score"]),
                    f"{float(best['elapsed_seconds']) / 60:.1f} min"
                    if pd.notna(best["elapsed_seconds"])
                    else "—",
                    str(int(best["routed_feature_count"]))
                    if pd.notna(best["routed_feature_count"])
                    else "—",
                ],
            }
        )
        st.dataframe(detail_rows, hide_index=True, width="stretch")
        st.info(
            t.get(
                "hpo.selection_note",
                "Holdout ranking measures robustness. Validation payout measures the "
                "optimization target. Treat disagreement as model-selection risk.",
            )
        )

with recipes_tab:
    recipes = recipe_summary(df)
    recipe_fig = px.bar(
        recipes.head(15).sort_values("median_holdout"),
        x="median_holdout",
        y="recipe",
        orientation="h",
        color="best_holdout",
        custom_data=["trials", "best_holdout", "median_payout"],
        color_continuous_scale=["#CBD5E1", COLORS["accent"]],
        title=t.get("hpo.recipe_title", "Most reliable model recipes"),
        labels={
            "median_holdout": t.get("hpo.median_holdout", "Median holdout Sharpe"),
            "recipe": "",
            "best_holdout": t.get("hpo.best_holdout", "Best holdout"),
        },
    )
    recipe_fig.update_traces(
        hovertemplate=(
            "%{y}<br>Median holdout: %{x:.3f}<br>"
            "Best holdout: %{customdata[1]:.3f}<br>"
            "Median payout: %{customdata[2]:.3f}<br>"
            "Trials: %{customdata[0]}<extra></extra>"
        )
    )
    st.plotly_chart(chart_layout(recipe_fig, height=520), width="stretch")

    recipe_cols = st.columns(3)
    for column, dimension, title in zip(
        recipe_cols,
        ["num_models", "ensemble_method", "use_augmentation"],
        [
            t.get("hpo.model_count", "Model count"),
            t.get("hpo.ensemble_method", "Ensemble method"),
            t.get("hpo.augmentation", "Synthetic augmentation"),
        ],
        strict=True,
    ):
        aggregate = (
            df.groupby(dimension)["holdout_corr_sharpe"]
            .agg(["count", "median", "max"])
            .reset_index()
        )
        fig = px.bar(
            aggregate,
            x=dimension,
            y="median",
            color="max",
            text="count",
            color_continuous_scale=["#CBD5E1", COLORS["primary"]],
            title=title,
            labels={"median": t.get("hpo.median_holdout", "Median holdout Sharpe")},
        )
        fig.update_traces(texttemplate="n=%{text}", textposition="outside")
        column.plotly_chart(chart_layout(fig, height=360), width="stretch")

with progression_tab:
    history = df_all.sort_values("trial").copy()
    history["best_holdout_so_far"] = history["holdout_corr_sharpe"].cummax()
    history["best_payout_so_far"] = history["payout_score"].cummax()

    holdout_fig = go.Figure()
    holdout_fig.add_trace(
        go.Scatter(
            x=history["trial"],
            y=history["holdout_corr_sharpe"],
            mode="markers",
            name=t.get("hpo.trials", "Trials"),
            marker={
                "color": history["holdout_corr_sharpe"],
                "colorscale": "Blues",
                "size": 7,
            },
            text=history["recipe"],
            customdata=history[["payout_score"]],
            hovertemplate=(
                "Trial %{x}<br>Holdout: %{y:.3f}<br>"
                "Validation payout: %{customdata[0]:.3f}<br>%{text}<extra></extra>"
            ),
        )
    )
    holdout_fig.add_trace(
        go.Scatter(
            x=history["trial"],
            y=history["best_holdout_so_far"],
            mode="lines",
            name=t.get("hpo.running_best", "Running best"),
            line={"color": COLORS["accent"], "width": 3},
        )
    )
    holdout_fig.update_layout(
        title=t.get("hpo.holdout_progression", "Holdout Sharpe progression"),
        xaxis_title=t.get("hpo.trial_number", "Trial"),
        yaxis_title=t.get("hpo.holdout_sharpe", "Holdout Sharpe"),
    )
    st.plotly_chart(chart_layout(holdout_fig, height=430), width="stretch")

    payout_fig = px.scatter(
        history.dropna(subset=["payout_score"]),
        x="trial",
        y="payout_score",
        color="ensemble_method",
        hover_name="recipe",
        title=t.get("hpo.payout_progression", "Validation payout progression"),
        labels={
            "trial": t.get("hpo.trial_number", "Trial"),
            "payout_score": t.get("hpo.validation_payout", "Validation payout"),
            "ensemble_method": t.get("hpo.ensemble_method", "Ensemble"),
        },
    )
    payout_fig.add_trace(
        go.Scatter(
            x=history["trial"],
            y=history["best_payout_so_far"],
            mode="lines",
            name=t.get("hpo.running_best", "Running best"),
            line={"color": COLORS["warning"], "width": 3},
        )
    )
    st.plotly_chart(chart_layout(payout_fig, height=390), width="stretch")

with leaderboard_tab:
    max_rows = min(100, len(ranked))
    top_n = st.slider(
        t.get("hpo.num_trials_display", "Rows to display"),
        min_value=1,
        max_value=max_rows,
        value=min(20, max_rows),
    )
    display_columns = list(
        dict.fromkeys(
            [
                "trial",
                rank_metric,
                "holdout_corr_sharpe",
                "payout_score",
                "val_corr_sharpe",
                "mean_era_corr",
                "max_drawdown",
                "pct_positive_eras",
                "recipe",
                "elapsed_seconds",
            ]
        )
    )
    leaderboard = ranked.head(top_n)[display_columns].copy()
    st.dataframe(
        leaderboard,
        hide_index=True,
        width="stretch",
        height=min(720, 36 * (top_n + 1)),
        column_config={
            "trial": st.column_config.NumberColumn("Trial", format="%d"),
            rank_metric: st.column_config.NumberColumn(
                t.get("hpo.selection_score", "Selection score"), format="%.4f"
            ),
            "holdout_corr_sharpe": st.column_config.NumberColumn(
                t.get("hpo.holdout_sharpe", "Holdout Sharpe"), format="%.4f"
            ),
            "payout_score": st.column_config.NumberColumn(
                t.get("hpo.validation_payout", "Validation payout"), format="%.4f"
            ),
            "pct_positive_eras": st.column_config.NumberColumn(
                t.get("hpo.positive_eras", "Positive eras"), format="percent"
            ),
            "elapsed_seconds": st.column_config.NumberColumn(
                t.get("hpo.elapsed_seconds", "Runtime (s)"), format="%.0f"
            ),
        },
    )
    st.download_button(
        t.get("hpo.download_leaderboard", "Download leaderboard"),
        data=leaderboard.to_csv(index=False).encode("utf-8"),
        file_name="alphapulse_hpo_leaderboard.csv",
        mime="text/csv",
    )

    st.subheader(
        t.get("hpo.best_config", "Configuration for selected trial #{trial}").format(
            trial=int(best["trial"])
        )
    )
    st.json(best["params"], expanded=False)
