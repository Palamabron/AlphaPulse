from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
TRIALS_DB = ROOT / "artifacts" / "hpo_v53_12h_stable_cores_20260824" / "trials.db"
PROTOCOL_PATH = TRIALS_DB.parent / "protocol.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

NAVY = "#16324F"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#D97706"
RED = "#B91C1C"
PURPLE = "#7C3AED"
GRAY = "#64748B"
LIGHT_GRAY = "#CBD5E1"
PALE_BLUE = "#DBEAFE"
PALE_TEAL = "#CCFBF1"
PALE_ORANGE = "#FFEDD5"

MODEL_COLORS = {
    "LightGBM": TEAL,
    "CatBoost": ORANGE,
    "XGBoost": BLUE,
    "TabICL": PURPLE,
    "TabPFN": RED,
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def load_trials() -> tuple[pd.DataFrame, dict[str, int], dict[str, Any]]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    statuses: dict[str, int] = {}
    with sqlite3.connect(TRIALS_DB) as connection:
        records = connection.execute(
            "SELECT trial_number, status, flat_config, metrics, elapsed_seconds "
            "FROM trials ORDER BY trial_number"
        ).fetchall()

    for trial_number, status, flat_config, metrics, elapsed_seconds in records:
        statuses[status] = statuses.get(status, 0) + 1
        if status != "completed" or not metrics:
            continue
        config = json.loads(flat_config)
        score = json.loads(metrics)
        holdout = score.get("holdout_numerai_corr_sharpe")
        validation = score.get("val_numerai_corr_sharpe")
        mmc = score.get("numerai_mmc_sharpe")
        if not all(_is_finite(value) for value in (holdout, validation, mmc)):
            continue
        rows.append(
            {
                "trial": int(trial_number),
                "model": str(config.get("model_1_type", "?")),
                "holdout": float(holdout),
                "holdout_mean": float(score["holdout_numerai_corr_mean"]),
                "holdout_drawdown": float(score["holdout_numerai_corr_max_drawdown"]),
                "holdout_positive": float(
                    score["holdout_numerai_corr_pct_positive_eras"]
                ),
                "validation": float(validation),
                "validation_mean": float(score["val_numerai_corr_mean"]),
                "mmc": float(mmc),
                "mmc_mean": float(score["numerai_mmc_mean"]),
                "seconds": float(elapsed_seconds),
                "features": int(config.get("routed_feature_count", 780)),
            }
        )

    trials = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    return trials, statuses, protocol


def polish_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def polish_axis(value: float, _position: float) -> str:
    return f"{value:g}".replace(".", ",")


def style_axis(axis: Axes, *, x_grid: bool = False) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(
        axis="x" if x_grid else "y",
        color=LIGHT_GRAY,
        linewidth=0.6,
        alpha=0.65,
    )
    axis.set_axisbelow(True)
    axis.yaxis.set_major_formatter(FuncFormatter(polish_axis))


def save_figure(figure: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_architecture() -> None:
    figure, axis = plt.subplots(figsize=(8.3, 4.9))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 8.0)
    axis.axis("off")
    layers = [
        (7.25, "Dane i cechy", "NumeraiDataLoader · katalog cech · cele", PALE_BLUE),
        (
            5.8,
            "Konfiguracja i automatyzacja",
            "YAML/Pydantic · Optuna/Ray · AutoResearch · TrialDB",
            "#FEF3C7",
        ),
        (
            4.35,
            "Potok predykcyjny",
            "routing cech · preprocessing · modele · ensemble",
            "#EDE9FE",
        ),
        (
            2.9,
            "Walidacja i diagnostyka",
            "purged holdout · CORR/MMC · W&B · EDA",
            PALE_TEAL,
        ),
        (
            1.45,
            "Eksport i inferencja",
            "predict.pkl · test dymny · submisja live",
            PALE_ORANGE,
        ),
    ]
    for index, (y, title, detail, color) in enumerate(layers):
        axis.add_patch(
            plt.Rectangle(
                (0.75, y - 0.5),
                8.5,
                1.0,
                facecolor=color,
                edgecolor=NAVY,
                linewidth=1.0,
            )
        )
        axis.text(
            5.0,
            y + 0.16,
            title,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
        )
        axis.text(5.0, y - 0.2, detail, ha="center", va="center", fontsize=10)
        if index < len(layers) - 1:
            next_y = layers[index + 1][0]
            axis.annotate(
                "",
                xy=(5.0, next_y + 0.54),
                xytext=(5.0, y - 0.54),
                arrowprops={"arrowstyle": "-|>", "color": NAVY, "linewidth": 1.2},
            )
    save_figure(figure, "architecture_print")


def plot_validation_design() -> None:
    figure, axis = plt.subplots(figsize=(9.2, 4.15))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7.4)
    axis.axis("off")
    segments = [
        (0.25, PALE_BLUE, NAVY, "UCZENIE", "506 er", "238 769 wierszy"),
        (4.2, PALE_ORANGE, ORANGE, "PURGE", "16 er", "7 814 wierszy"),
        (8.15, PALE_TEAL, TEAL, "HOLDOUT", "52 ery", "28 044 wiersze"),
    ]
    for x, fill, edge, title, eras, rows in segments:
        axis.add_patch(
            plt.Rectangle(
                (x, 4.4), 3.6, 1.75, facecolor=fill, edgecolor=edge, linewidth=1.3
            )
        )
        axis.text(x + 1.8, 5.65, title, ha="center", weight="bold", color=edge)
        axis.text(x + 1.8, 5.15, eras, ha="center", fontsize=10, weight="bold")
        axis.text(x + 1.8, 4.72, rows, ha="center", fontsize=9.5)

    for start in (3.86, 7.81):
        axis.annotate(
            "",
            xy=(start + 0.27, 5.27),
            xytext=(start, 5.27),
            arrowprops={"arrowstyle": "-|>", "color": NAVY, "linewidth": 1.2},
        )

    axis.text(
        0.25,
        6.85,
        "Próba ucząca v5.3: 274 627 wierszy, 574 ery, 780 cech bazowych",
        fontsize=11.5,
        weight="bold",
        color=NAVY,
    )
    axis.text(
        6.0,
        3.83,
        "Purge usuwa obserwacje z nakładającego się horyzontu celu",
        ha="center",
        fontsize=9.5,
        color=ORANGE,
    )
    axis.add_patch(
        plt.Rectangle(
            (1.35, 0.55),
            9.3,
            1.65,
            facecolor="#F8FAFC",
            edgecolor=PURPLE,
            linewidth=1.3,
        )
    )
    axis.text(
        6.0,
        1.65,
        "NIEZALEŻNY ZBIÓR VALIDATION · 88 ER",
        ha="center",
        weight="bold",
        color=PURPLE,
    )
    axis.text(
        6.0,
        1.05,
        "wyrównana predykcja Meta Modelu · ocena CORR oraz MMC",
        ha="center",
        fontsize=9.5,
    )
    axis.annotate(
        "",
        xy=(6.0, 2.25),
        xytext=(6.0, 3.42),
        arrowprops={"arrowstyle": "-|>", "color": PURPLE, "linewidth": 1.2},
    )
    save_figure(figure, "validation_design")


def plot_hpo_progression(trials: pd.DataFrame) -> None:
    history = trials.copy()
    history["best_holdout"] = history["holdout"].cummax()
    history["best_validation"] = history["validation"].cummax()
    figure, axes = plt.subplots(2, 1, figsize=(8.3, 6.4), sharex=True)

    for axis, metric, best, ylabel in (
        (axes[0], "holdout", "best_holdout", "CORR Sharpe · holdout"),
        (axes[1], "validation", "best_validation", "CORR Sharpe · validation"),
    ):
        for model, group in history.groupby("model"):
            axis.scatter(
                group["trial"],
                group[metric],
                s=34,
                color=MODEL_COLORS[model],
                alpha=0.8,
                label=model,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        axis.plot(history["trial"], history[best], color=NAVY, linewidth=1.9)
        axis.axvline(19.5, color=GRAY, linewidth=1.0, linestyle="--")
        axis.axhline(0.0, color=GRAY, linewidth=0.7)
        axis.set_ylabel(ylabel)
        style_axis(axis)

    axes[0].set_title("Przebieg optymalizacji HPO")
    axes[0].annotate(
        "próba 11 · 1,044",
        (11, history.loc[history["trial"] == 11, "holdout"].iloc[0]),
        xytext=(15.2, 0.9),
        arrowprops={"arrowstyle": "->", "color": TEAL},
        fontsize=9,
        color=TEAL,
    )
    axes[1].annotate(
        "próba 22 · 0,425",
        (22, history.loc[history["trial"] == 22, "validation"].iloc[0]),
        xytext=(24.5, 0.34),
        arrowprops={"arrowstyle": "->", "color": TEAL},
        fontsize=9,
        color=TEAL,
    )
    axes[1].text(9.5, -0.27, "faza inicjalna", ha="center", color=GRAY, fontsize=9)
    axes[1].text(25.2, -0.27, "adaptacja TPE", ha="center", color=GRAY, fontsize=9)
    axes[1].set_xlabel("Numer próby")
    axes[1].set_xlim(-0.8, 31)
    axes[1].set_ylim(-0.31, 0.51)
    axes[0].set_ylim(-0.38, 1.16)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncols=5, loc="lower center")
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(figure, "official_hpo_progression")


def plot_holdout_validation(trials: pd.DataFrame) -> None:
    pearson = trials["holdout"].corr(trials["validation"])
    spearman = trials["holdout"].corr(trials["validation"], method="spearman")
    figure, axis = plt.subplots(figsize=(7.3, 5.0))
    for model, group in trials.groupby("model"):
        axis.scatter(
            group["holdout"],
            group["validation"],
            s=48,
            color=MODEL_COLORS[model],
            alpha=0.82,
            label=model,
            edgecolor="white",
            linewidth=0.5,
        )
    for trial_number, offset in ((11, (8, -20)), (22, (8, 8)), (4, (-60, 10))):
        row = trials.loc[trials["trial"] == trial_number].iloc[0]
        axis.annotate(
            f"próba {trial_number}",
            (row["holdout"], row["validation"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=NAVY,
            arrowprops={"arrowstyle": "-", "color": GRAY, "linewidth": 0.8},
        )
    axis.axhline(0.0, color=GRAY, linewidth=0.7)
    axis.axvline(0.0, color=GRAY, linewidth=0.7)
    statistics = (
        f"n = {len(trials)}\n"
        f"Pearson r = {polish_number(pearson)}\n"
        f"Spearman ρ = {polish_number(spearman)}"
    )
    axis.text(
        0.03,
        0.96,
        statistics,
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "pad": 4},
    )
    axis.set_xlabel("CORR Sharpe na holdoucie (52 ery)")
    axis.set_ylabel("CORR Sharpe na validation (88 er)")
    axis.set_title("Zgodność jakości między dwoma przedziałami czasowymi")
    axis.legend(frameon=False, ncols=2, loc="lower right")
    axis.xaxis.set_major_formatter(FuncFormatter(polish_axis))
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, "holdout_validation_scatter")


def pareto_trials(trials: pd.DataFrame) -> pd.DataFrame:
    efficient: list[bool] = []
    for _, candidate in trials.iterrows():
        dominated = (
            (trials["validation"] >= candidate["validation"])
            & (trials["mmc"] >= candidate["mmc"])
            & (
                (trials["validation"] > candidate["validation"])
                | (trials["mmc"] > candidate["mmc"])
            )
        ).any()
        efficient.append(not dominated)
    return trials.loc[efficient].sort_values("validation")


def plot_pareto(trials: pd.DataFrame) -> None:
    frontier = pareto_trials(trials)
    figure, axis = plt.subplots(figsize=(7.3, 5.0))
    for model, group in trials.groupby("model"):
        axis.scatter(
            group["validation"],
            group["mmc"],
            s=44,
            color=MODEL_COLORS[model],
            alpha=0.62,
            label=model,
            edgecolor="white",
            linewidth=0.5,
        )
    axis.plot(
        frontier["validation"],
        frontier["mmc"],
        color=NAVY,
        linewidth=1.6,
        linestyle="--",
        zorder=2,
    )
    axis.scatter(
        frontier["validation"],
        frontier["mmc"],
        s=105,
        facecolor="none",
        edgecolor=NAVY,
        linewidth=1.6,
        zorder=3,
    )
    for _, row in frontier.iterrows():
        offset = (-82, 8) if row["trial"] == 4 else (10, -20)
        axis.annotate(
            f"próba {int(row['trial'])} · {row['model']}",
            (row["validation"], row["mmc"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=NAVY,
            arrowprops={"arrowstyle": "-", "color": NAVY, "linewidth": 0.8},
        )
    axis.axhline(0.0, color=GRAY, linewidth=0.7)
    axis.axvline(0.0, color=GRAY, linewidth=0.7)
    axis.set_xlabel("CORR Sharpe na validation")
    axis.set_ylabel("MMC Sharpe na validation")
    axis.set_title("Front Pareto jakości sygnału i jego unikalności")
    axis.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    axis.xaxis.set_major_formatter(FuncFormatter(polish_axis))
    style_axis(axis)
    figure.tight_layout(rect=(0, 0, 0.77, 1))
    save_figure(figure, "validation_pareto")


def plot_model_family_comparison(trials: pd.DataFrame) -> None:
    order = ["LightGBM", "CatBoost", "XGBoost", "TabICL", "TabPFN"]
    figure, axis = plt.subplots(figsize=(8.2, 4.7))
    rng = np.random.default_rng(20260824)
    values = [
        trials.loc[trials["model"] == model, "holdout"].to_numpy() for model in order
    ]
    boxes = axis.boxplot(
        values,
        positions=np.arange(len(order)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": NAVY, "linewidth": 1.5},
        whiskerprops={"color": GRAY},
        capprops={"color": GRAY},
    )
    for box, model in zip(boxes["boxes"], order, strict=True):
        box.set_facecolor(MODEL_COLORS[model])
        box.set_alpha(0.22)
        box.set_edgecolor(MODEL_COLORS[model])
    for index, (model, model_values) in enumerate(zip(order, values, strict=True)):
        jitter = rng.normal(0.0, 0.055, len(model_values))
        axis.scatter(
            np.full(len(model_values), index) + jitter,
            model_values,
            s=38,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        label_y = min(float(model_values.max()) + 0.07, 1.1)
        axis.text(
            index,
            label_y,
            f"n={len(model_values)}",
            ha="center",
            fontsize=9,
            color=GRAY,
        )
    axis.axhline(0.0, color=GRAY, linewidth=0.8)
    axis.set_xticks(np.arange(len(order)), order)
    axis.set_ylabel("Oficjalny CORR Sharpe na holdoucie")
    axis.set_title("Rozkład wyników według rodziny modelu")
    axis.set_ylim(-0.42, 1.16)
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, "model_family_comparison")


def write_metrics(
    trials: pd.DataFrame, statuses: dict[str, int], protocol: dict[str, Any]
) -> None:
    winner = trials.loc[trials["holdout"].idxmax()]
    validation_winner = trials.loc[trials["validation"].idxmax()]
    mmc_winner = trials.loc[trials["mmc"].idxmax()]
    launched = sum(statuses.values())
    commands = {
        "FinalTrialsLaunched": str(launched),
        "FinalTrialsCompleted": str(len(trials)),
        "FinalTrialsFailed": str(statuses.get("failed", 0)),
        "FinalTrialsInterrupted": str(statuses.get("running", 0)),
        "FinalCompletionRate": polish_number(100 * len(trials) / launched, 1),
        "FinalBestTrial": str(int(winner["trial"])),
        "FinalBestModel": str(winner["model"]),
        "FinalBestCorrSharpe": polish_number(winner["holdout"]),
        "FinalBestMeanCorr": polish_number(winner["holdout_mean"], 4),
        "FinalBestDrawdown": polish_number(winner["holdout_drawdown"], 4),
        "FinalBestPositiveEras": polish_number(100 * winner["holdout_positive"], 1),
        "FinalBestValidationCorrSharpe": polish_number(winner["validation"]),
        "FinalBestMmcSharpe": polish_number(winner["mmc"]),
        "FinalValidationWinnerTrial": str(int(validation_winner["trial"])),
        "FinalValidationBest": polish_number(validation_winner["validation"]),
        "FinalMmcWinnerTrial": str(int(mmc_winner["trial"])),
        "FinalMmcBest": polish_number(mmc_winner["mmc"]),
        "FinalMedianRuntime": polish_number(trials["seconds"].median(), 1),
        "FinalCompletedComputeHours": polish_number(trials["seconds"].sum() / 3600, 2),
        "FinalSampleRows": f"{int(protocol['sampled_rows']):,}".replace(",", "~"),
    }
    content = "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in commands.items()
    )
    (OUTPUT_DIR / "generated_metrics.tex").write_text(content + "\n", encoding="utf-8")


def main() -> None:
    trials, statuses, protocol = load_trials()
    plot_architecture()
    plot_validation_design()
    plot_hpo_progression(trials)
    plot_holdout_validation(trials)
    plot_pareto(trials)
    plot_model_family_comparison(trials)
    write_metrics(trials, statuses, protocol)


if __name__ == "__main__":
    main()
