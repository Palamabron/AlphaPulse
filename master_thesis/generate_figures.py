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
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parents[1]
TRIALS_DB = ROOT / "artifacts" / "hpo_v53_12h_stable_cores_20260824" / "trials.db"
PROTOCOL_PATH = TRIALS_DB.parent / "protocol.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
REPRODUCIBILITY_DIR = Path(__file__).resolve().parent / "reproducibility"

EXTRAPOLATION_SEED = 20260830
EXTRAPOLATION_REPLICATES = 200_000
EXTRAPOLATION_BUDGETS = (50, 100, 200)
EXTRAPOLATION_CHUNK_SIZE = 2_000

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
    figure.savefig(
        OUTPUT_DIR / f"{name}.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "AlphaPulse master_thesis/generate_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_dataset_eda() -> None:
    """Summarize the full v5.3 training set used by the EDA discussion."""
    import pyarrow.parquet as pq

    data_path = ROOT / "data" / "v5.3" / "train.parquet"
    feature_path = ROOT / "data" / "v5.3" / "features.json"
    metadata = json.loads(feature_path.read_text(encoding="utf-8"))
    features = list(metadata["feature_sets"]["small"])
    target = "target_ender_60"
    era_count = 574
    feature_count = len(features)

    counts = np.zeros(era_count)
    target_sum = np.zeros(era_count)
    target_sq_sum = np.zeros(era_count)
    feature_sum = np.zeros((era_count, feature_count))
    feature_sq_sum = np.zeros((era_count, feature_count))
    cross_sum = np.zeros((era_count, feature_count))

    parquet = pq.ParquetFile(data_path)
    columns = ["era", target, *features]
    for batch in parquet.iter_batches(columns=columns, batch_size=65_536):
        frame = batch.to_pandas()
        era = frame.pop("era").astype(str).astype(int).to_numpy() - 1
        y = frame.pop(target).to_numpy(dtype=float)
        x = frame.to_numpy(dtype=float)
        counts += np.bincount(era, minlength=era_count)
        target_sum += np.bincount(era, weights=y, minlength=era_count)
        target_sq_sum += np.bincount(era, weights=y * y, minlength=era_count)
        for index, values in enumerate(x.T):
            feature_sum[:, index] += np.bincount(
                era, weights=values, minlength=era_count
            )
            feature_sq_sum[:, index] += np.bincount(
                era, weights=values * values, minlength=era_count
            )
            cross_sum[:, index] += np.bincount(
                era, weights=values * y, minlength=era_count
            )

    target_mean = target_sum / counts
    target_variance = (target_sq_sum - target_sum**2 / counts) / (counts - 1)
    feature_variance = (feature_sq_sum - feature_sum**2 / counts[:, None]) / (
        counts[:, None] - 1
    )
    covariance = (cross_sum - feature_sum * target_sum[:, None] / counts[:, None]) / (
        counts[:, None] - 1
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        era_correlations = covariance / np.sqrt(
            feature_variance * target_variance[:, None]
        )

    total_count = counts.sum()
    global_target_sum = target_sum.sum()
    global_target_variance = (
        target_sq_sum.sum() - global_target_sum**2 / total_count
    ) / (total_count - 1)
    global_feature_sum = feature_sum.sum(axis=0)
    global_feature_variance = (
        feature_sq_sum.sum(axis=0) - global_feature_sum**2 / total_count
    ) / (total_count - 1)
    global_covariance = (
        cross_sum.sum(axis=0) - global_feature_sum * global_target_sum / total_count
    ) / (total_count - 1)
    global_correlations = global_covariance / np.sqrt(
        global_feature_variance * global_target_variance
    )

    eras = np.arange(1, era_count + 1)
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.5))

    axes[0, 0].plot(eras, counts, color=BLUE, linewidth=1.0)
    axes[0, 0].axhline(np.median(counts), color=GRAY, linewidth=0.9, linestyle="--")
    axes[0, 0].set_title("Liczba obserwacji w erach")
    axes[0, 0].set_xlabel("Era")
    axes[0, 0].set_ylabel("Liczba wierszy")
    style_axis(axes[0, 0])

    target_delta = (target_mean - 0.5) * 100_000
    axes[0, 1].plot(eras, target_delta, color=TEAL, linewidth=1.0)
    axes[0, 1].axhline(0, color=GRAY, linewidth=0.9)
    axes[0, 1].set_title("Średnia celu w kolejnych erach")
    axes[0, 1].set_xlabel("Era")
    axes[0, 1].set_ylabel("Odchylenie od 0,5 · 100 000")
    style_axis(axes[0, 1])

    ordered = np.sort(global_correlations)
    axes[1, 0].bar(
        np.arange(feature_count),
        ordered,
        color=np.where(ordered >= 0, TEAL, ORANGE),
        width=0.82,
    )
    axes[1, 0].axhline(0, color=GRAY, linewidth=0.9)
    axes[1, 0].set_title("Korelacja cech z celem")
    axes[1, 0].set_xlabel("42 cechy zestawu small")
    axes[1, 0].set_ylabel("Korelacja Pearsona")
    style_axis(axes[1, 0])

    finite = np.abs(era_correlations[np.isfinite(era_correlations)])
    axes[1, 1].hist(finite, bins=30, color=BLUE, alpha=0.82, edgecolor="white")
    axes[1, 1].axvline(np.median(finite), color=RED, linewidth=1.2)
    axes[1, 1].set_title("Siła zależności w pojedynczych erach")
    axes[1, 1].set_xlabel("Wartość bezwzględna korelacji")
    axes[1, 1].set_ylabel("Liczba obserwacji")
    style_axis(axes[1, 1])

    figure.suptitle(
        "Analiza eksploracyjna zbioru treningowego Numerai v5.3",
        fontsize=13,
        weight="bold",
        color=NAVY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, "dataset_eda_summary")


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
        "Próba ucząca v5.3: 274 627 wierszy, 574 ery · routing 256-989 z 3555 cech",
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
        "PÓŹNIEJSZY PRZEDZIAŁ SELEKCYJNY · 88 ER",
        ha="center",
        weight="bold",
        color=PURPLE,
        fontsize=10,
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
        (
            axes[1],
            "validation",
            "best_validation",
            "CORR Sharpe · późniejszy przedział",
        ),
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
        # Optuna counts completed startup trials rather than raw trial numbers.
        # Trial 22 was the twentieth completed startup trial, so trial 23 was
        # the first proposal informed by TPE.
        axis.axvline(22.5, color=GRAY, linewidth=1.0, linestyle="--")
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
    axes[1].text(
        10.7,
        -0.27,
        "losowy start · 20 ukończonych",
        ha="center",
        color=GRAY,
        fontsize=8.5,
    )
    axes[1].text(
        27.0,
        -0.27,
        "TPE · od próby 23",
        ha="center",
        color=GRAY,
        fontsize=8.5,
    )
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
    axis.set_ylabel("CORR Sharpe - późniejszy przedział (88 er)")
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
    axis.set_xlabel("CORR Sharpe - późniejszy przedział")
    axis.set_ylabel("MMC Sharpe - późniejszy przedział")
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


def _simulate_empirical_best(
    values: np.ndarray,
    *,
    future_trials: int,
    incumbent: float,
) -> np.ndarray:
    """Bootstrap future maxima without extrapolating beyond observed support."""
    rng = np.random.default_rng(EXTRAPOLATION_SEED)
    output = np.empty(EXTRAPOLATION_REPLICATES)
    for start in range(0, EXTRAPOLATION_REPLICATES, EXTRAPOLATION_CHUNK_SIZE):
        size = min(
            EXTRAPOLATION_CHUNK_SIZE,
            EXTRAPOLATION_REPLICATES - start,
        )
        sampled = values[rng.integers(0, len(values), size=(size, future_trials))]
        output[start : start + size] = np.maximum(
            incumbent,
            sampled.max(axis=1),
        )
    return output


def _simulate_normal_bootstrap_best(
    values: np.ndarray,
    *,
    future_trials: int,
    incumbent: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Scenario bootstrap with a Normal local tail smoother.

    For every Monte Carlo replication, the source observations are sampled
    with replacement, their mean and sample standard deviation are estimated,
    and the maximum of ``future_trials`` Normal predictive draws is generated
    through its inverse CDF.  The returned value is always best-so-far, so it
    cannot fall below the observed incumbent.
    """
    output = np.empty(EXTRAPOLATION_REPLICATES)
    for start in range(0, EXTRAPOLATION_REPLICATES, EXTRAPOLATION_CHUNK_SIZE):
        size = min(
            EXTRAPOLATION_CHUNK_SIZE,
            EXTRAPOLATION_REPLICATES - start,
        )
        bootstrap = values[rng.integers(0, len(values), size=(size, len(values)))]
        means = bootstrap.mean(axis=1)
        standard_deviations = bootstrap.std(axis=1, ddof=1)
        maximum_quantiles = rng.random(size) ** (1.0 / future_trials)
        future_maxima = means + standard_deviations * ndtri(maximum_quantiles)
        output[start : start + size] = np.maximum(incumbent, future_maxima)
    return output


def _extrapolation_row(
    *,
    scenario: str,
    label: str,
    budget: int,
    source: np.ndarray,
    incumbent: float,
    simulated_best: np.ndarray,
) -> dict[str, Any]:
    lower, upper = np.quantile(simulated_best, (0.05, 0.95))
    return {
        "scenario": scenario,
        "scenario_label_pl": label,
        "total_completed_trials": budget,
        "additional_completed_trials": budget - 27,
        "source_observations": len(source),
        "source_mean": float(source.mean()),
        "source_sample_std": float(source.std(ddof=1)),
        "incumbent_holdout_corr_sharpe": incumbent,
        "expected_best": float(simulated_best.mean()),
        "median_best": float(np.median(simulated_best)),
        "predictive_p05": float(lower),
        "predictive_p95": float(upper),
        "replication_fraction_improving": float(np.mean(simulated_best > incumbent)),
        "monte_carlo_replicates": EXTRAPOLATION_REPLICATES,
        "random_seed": EXTRAPOLATION_SEED,
    }


def generate_hpo_budget_extrapolation(trials: pd.DataFrame) -> pd.DataFrame:
    """Generate a reproducible, explicitly scenario-based HPO extrapolation.

    Only completed trials are present in ``trials``.  Trial 22 was still part
    of Optuna's 20-completion random startup phase; therefore the cautious and
    empirical scenarios use the seven completed adaptive trials 23--30.  The
    optimistic sensitivity scenario assumes that future proposals concentrate
    on the 16 observed LightGBM and CatBoost configurations.

    The later 88-era selection interval and MMC are deliberately not
    extrapolated: they were inspected during candidate selection and their
    rankings transfer only weakly from the optimized holdout metric.
    """
    if len(trials) != 27:
        raise ValueError(
            "The published extrapolation requires exactly 27 completed trials; "
            f"found {len(trials)}. Re-audit the phase boundary and methodology."
        )

    incumbent = float(trials["holdout"].max())
    adaptive = trials.loc[trials["trial"] >= 23, "holdout"].to_numpy()
    tree_families = trials.loc[
        trials["model"].isin(("LightGBM", "CatBoost")), "holdout"
    ].to_numpy()
    if len(adaptive) != 7 or len(tree_families) != 16:
        raise ValueError(
            "The published scenarios require 7 adaptive TPE and 16 tree-family "
            f"observations; found {len(adaptive)} and {len(tree_families)}."
        )

    rows: list[dict[str, Any]] = []
    for budget in EXTRAPOLATION_BUDGETS:
        empirical = _simulate_empirical_best(
            adaptive,
            future_trials=budget - len(trials),
            incumbent=incumbent,
        )
        rows.append(
            _extrapolation_row(
                scenario="empirical_no_unseen_tail",
                label="empiryczny · bez niewidocznego ogona",
                budget=budget,
                source=adaptive,
                incumbent=incumbent,
                simulated_best=empirical,
            )
        )

    # One deterministic stream, with scenarios and budgets evaluated in this
    # fixed order, reproduces the values reported in the thesis analysis.
    rng = np.random.default_rng(EXTRAPOLATION_SEED)
    for scenario, label, source in (
        ("cautious_tpe7", "ostrożny · kontynuacja TPE", adaptive),
        (
            "optimistic_tree16",
            "optymistyczny · koncentracja na drzewach",
            tree_families,
        ),
    ):
        for budget in EXTRAPOLATION_BUDGETS:
            simulated = _simulate_normal_bootstrap_best(
                source,
                future_trials=budget - len(trials),
                incumbent=incumbent,
                rng=rng,
            )
            rows.append(
                _extrapolation_row(
                    scenario=scenario,
                    label=label,
                    budget=budget,
                    source=source,
                    incumbent=incumbent,
                    simulated_best=simulated,
                )
            )

    extrapolation = pd.DataFrame(rows)
    scenario_order = {
        "empirical_no_unseen_tail": 0,
        "cautious_tpe7": 1,
        "optimistic_tree16": 2,
    }
    extrapolation["_scenario_order"] = extrapolation["scenario"].map(scenario_order)
    extrapolation = extrapolation.sort_values(
        ["_scenario_order", "total_completed_trials"]
    ).drop(columns="_scenario_order")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extrapolation.to_csv(
        OUTPUT_DIR / "hpo_budget_extrapolation.csv",
        index=False,
        float_format="%.9f",
    )
    return extrapolation


def plot_hpo_budget_extrapolation(extrapolation: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    incumbent = float(extrapolation["incumbent_holdout_corr_sharpe"].iloc[0])
    visual = {
        "empirical_no_unseen_tail": (GRAY, "empiryczny"),
        "cautious_tpe7": (TEAL, "ostrożny TPE"),
        "optimistic_tree16": (ORANGE, "optymistyczny - drzewa"),
    }

    for scenario, (color, label) in visual.items():
        group = extrapolation.loc[extrapolation["scenario"] == scenario]
        x_values = np.r_[27, group["total_completed_trials"].to_numpy()]
        expected = np.r_[incumbent, group["expected_best"].to_numpy()]
        lower = np.r_[incumbent, group["predictive_p05"].to_numpy()]
        upper = np.r_[incumbent, group["predictive_p95"].to_numpy()]
        improvement_fraction = np.r_[
            0.0, group["replication_fraction_improving"].to_numpy()
        ]
        axes[0].plot(
            x_values,
            expected,
            marker="o",
            color=color,
            linewidth=1.8,
            label=label,
        )
        axes[0].fill_between(x_values, lower, upper, color=color, alpha=0.12)
        axes[1].plot(
            x_values,
            improvement_fraction,
            marker="o",
            color=color,
            linewidth=1.8,
            label=label,
        )

    axes[0].axhline(
        incumbent,
        color=NAVY,
        linewidth=1.0,
        linestyle="--",
        label="obecny lider · 1,044",
    )
    axes[0].set_title("Najlepszy CORR Sharpe na holdoucie")
    axes[0].set_ylabel("Średnia maksimum · pasmo 5-95%")
    axes[0].set_xlabel("Łączna liczba ukończonych prób")
    axes[0].set_xticks((27, *EXTRAPOLATION_BUDGETS))
    axes[0].set_ylim(1.025, 1.315)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(axes[0])

    axes[1].set_title("Replikacje poprawiające obecnego lidera")
    axes[1].set_ylabel("Odsetek replikacji scenariusza")
    axes[1].set_xlabel("Łączna liczba ukończonych prób")
    axes[1].set_xticks((27, *EXTRAPOLATION_BUDGETS))
    axes[1].set_ylim(-0.02, 0.70)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(axes[1])

    figure.suptitle("Scenariuszowa ekstrapolacja budżetu HPO", weight="bold")
    figure.text(
        0.5,
        0.01,
        "200 000 replikacji · seed 20260830 · przedziały scenariuszowe, "
        "nie prognoza jakości rynkowej",
        ha="center",
        fontsize=8.5,
        color=GRAY,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(figure, "hpo_budget_extrapolation")


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
        "FinalBestCorrSharpe": polish_number(winner["holdout"], 4),
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


def export_historical_candidate_records(protocol: dict[str, Any]) -> None:
    """Freeze the two published candidate DB records and protocol for audit."""
    REPRODUCIBILITY_DIR.mkdir(parents=True, exist_ok=True)
    protocol_export = {
        "source_artifact": str(PROTOCOL_PATH.relative_to(ROOT)),
        "historical_record": protocol,
        "audit_notes": [
            "The source tree hash does not identify a reconstructible Git snapshot.",
            (
                "The current code adds an inner target-aware purge that was "
                "absent historically."
            ),
        ],
    }
    (REPRODUCIBILITY_DIR / "historical_protocol.json").write_text(
        json.dumps(protocol_export, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with sqlite3.connect(TRIALS_DB) as connection:
        rows = connection.execute(
            "SELECT trial_number, status, flat_config, metrics, error, "
            "elapsed_seconds, created_at FROM trials "
            "WHERE trial_number IN (4, 22) ORDER BY trial_number"
        ).fetchall()
    if [row[0] for row in rows] != [4, 22]:
        raise ValueError("Historical candidate records 4 and 22 are required")

    notes = {
        4: [
            (
                "catboost_colsample_bylevel was recorded but not passed by the "
                "historical GPU builder; it was ineffective."
            ),
            (
                "catboost_min_data_in_leaf=500 was recorded, but grow_policy "
                "was not set; the default SymmetricTree policy does not use "
                "min_data_in_leaf, so the parameter was ineffective."
            ),
            (
                "catboost_iterations and catboost_early_stopping were absent from "
                "the flat record, so resolver defaults were 2000 and 100."
            ),
            (
                "Early stopping was ineffective inside the historical EraEnsemble; "
                "all 2000 base rounds were used."
            ),
        ],
        22: [
            (
                "lgbm_early_stopping=50 was recorded but ineffective inside the "
                "historical EraEnsemble; the fixed 200-round budget was used."
            ),
        ],
    }
    for (
        trial_number,
        status,
        flat_config,
        metrics,
        error,
        elapsed_seconds,
        created_at,
    ) in rows:
        export = {
            "source_artifact": str(TRIALS_DB.relative_to(ROOT)),
            "trial_number": trial_number,
            "status": status,
            "flat_config": json.loads(flat_config),
            "metrics": json.loads(metrics),
            "error": error,
            "elapsed_seconds": elapsed_seconds,
            "created_at": created_at,
            "audit_notes": notes[trial_number],
        }
        (REPRODUCIBILITY_DIR / f"trial_{trial_number:02d}_record.json").write_text(
            json.dumps(export, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    trials, statuses, protocol = load_trials()
    plot_dataset_eda()
    plot_architecture()
    plot_validation_design()
    plot_hpo_progression(trials)
    plot_holdout_validation(trials)
    plot_pareto(trials)
    plot_model_family_comparison(trials)
    extrapolation = generate_hpo_budget_extrapolation(trials)
    plot_hpo_budget_extrapolation(extrapolation)
    write_metrics(trials, statuses, protocol)
    export_historical_candidate_records(protocol)


if __name__ == "__main__":
    main()
