from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..pipeline.model_access import (
    PipelineLike,
    model_prediction_map,
    multitarget_blend_weights,
)
from ..pipeline.multi_target import MultiTargetPipeline
from ..pipeline.pipeline import Pipeline
from ..utils.alignment import align_series_to_frame
from .backtester import predict_with_optional_eras
from .metrics import per_era_correlation, rank_normalize

MAX_HEXBIN_POINTS = 10_000
FEATURE_EXPOSURE_TOP_N = 15
MAX_FNC_FEATURES = 200
_ERA_IMPORTANCE_MIN_ROWS = 10
_PRED_PLOT_BINS = 20
_PRED_PLOT_DPI = 150


def _diag_key(split: str, name: str) -> str:
    return f"diagnostics/{split}/{name}"


def _wandb_active() -> bool:
    try:
        import wandb

        return wandb.run is not None
    except ImportError:
        return False


def _log_wandb_figure(wandb: Any, key: str, fig: Any) -> None:
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_PRED_PLOT_DPI, bbox_inches="tight")
    buf.seek(0)
    wandb.log({key: wandb.Image(Image.open(buf))})


def _new_figure(*, figsize: tuple[float, float]) -> Any:
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    mpl.use("Agg", force=True)
    fig = Figure(figsize=figsize, dpi=_PRED_PLOT_DPI)
    FigureCanvasAgg(fig)
    return fig


def _log_horizontal_bar_chart(
    wandb: Any,
    *,
    labels: list[str],
    values: list[float],
    key: str,
    title: str,
    xlabel: str,
) -> None:
    if not labels:
        return

    n = len(labels)
    fig_h = max(3.5, min(14.0, n * 0.32))
    fig = _new_figure(figsize=(9, fig_h))
    ax = fig.add_subplot(111)
    y = np.arange(n)
    ax.barh(y, values, color="steelblue", edgecolor="white", height=0.75)
    ax.set_yticks(y, labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    _log_wandb_figure(wandb, key, fig)


def _log_correlation_heatmap(
    wandb: Any,
    names: list[str],
    corr: dict[str, dict[str, float]],
    key: str,
    *,
    title: str,
) -> None:
    if len(names) < 2:
        return

    mat = np.array([[float(corr[a][b]) for b in names] for a in names])
    size = max(5.0, min(12.0, len(names) * 0.65))
    fig = _new_figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(mat[i, j]) > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    _log_wandb_figure(wandb, key, fig)


def _log_mmc_metrics_from_dict(
    wandb: Any, metrics: dict[str, float], *, split: str = "validation"
) -> None:
    logged: dict[str, float] = {}
    scalar_names = {
        "mmc": "ValidationMmc",
        "mmc_sharpe": "ValidationMmcSharpe",
        "payout_score": "LegacyPayoutProxy",
        "corr_sharpe": "ValidationSharpe",
        "mean_per_era_correlation": "ValidationMeanCorr",
    }
    for key, wandb_name in scalar_names.items():
        value = metrics.get(key)
        if key == "corr_sharpe" and metrics.get("val_corr_sharpe") is not None:
            value = metrics.get("val_corr_sharpe")
        if (
            key == "mean_per_era_correlation"
            and metrics.get("val_mean_per_era_correlation") is not None
        ):
            value = metrics.get("val_mean_per_era_correlation")
        if value is not None and np.isfinite(value):
            logged[_diag_key(split, wandb_name)] = float(value)
    if logged:
        wandb.log(logged)


def _subsample_finite_pairs(
    y: np.ndarray, p: np.ndarray, *, max_points: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) <= max_points:
        return y, p
    idx = np.random.default_rng(seed).choice(len(y), size=max_points, replace=False)
    return y[idx], p[idx]


def _collect_model_predictions(
    pipeline: PipelineLike,
    X_val: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, np.ndarray]:
    return model_prediction_map(pipeline, X_val, feature_cols)


def _feature_exposure_summary(
    preds: np.ndarray,
    features: pd.DataFrame,
    eras: pd.Series,
    *,
    top_n: int = FEATURE_EXPOSURE_TOP_N,
) -> dict[str, Any]:
    e_arr = np.asarray(eras.to_numpy())
    unique_eras = sorted(pd.unique(e_arr), key=str)
    per_feature: dict[str, list[float]] = {c: [] for c in features.columns}

    for era in unique_eras:
        mask = e_arr == era
        if mask.sum() < 3:
            continue
        p = preds[mask]
        for col in features.columns:
            f = features[col].to_numpy()[mask]
            if np.std(p) == 0 or np.std(f) == 0:
                continue
            corr = float(np.corrcoef(p, f)[0, 1])
            if np.isfinite(corr):
                per_feature[col].append(abs(corr))

    mean_abs = {col: float(np.mean(vals)) for col, vals in per_feature.items() if vals}
    if not mean_abs:
        return {
            "max_mean_abs_corr": float("nan"),
            "mean_abs_corr": float("nan"),
            "top": [],
        }

    ranked = sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)
    top = [{"feature": k, "mean_abs_corr": v} for k, v in ranked[:top_n]]
    all_vals = list(mean_abs.values())
    return {
        "max_mean_abs_corr": max(all_vals),
        "mean_abs_corr": float(np.mean(all_vals)),
        "top": top,
    }


def _aligned_meta_model_predictions(
    X: pd.DataFrame,
    meta_model_preds: np.ndarray | pd.Series | None,
) -> np.ndarray | None:
    if isinstance(meta_model_preds, pd.Series):
        meta_model_preds = align_series_to_frame(
            X,
            meta_model_preds,
            name="meta_model_preds",
        )
    if meta_model_preds is None:
        return None
    meta_array = np.asarray(meta_model_preds, dtype=np.float64).reshape(-1)
    if len(meta_array) != len(X):
        raise ValueError("meta_model_preds length must match the validation frame")
    if not np.isfinite(meta_array).all():
        raise ValueError("meta_model_preds contains missing or non-finite predictions")
    return meta_array


def log_experiment_diagnostics(
    *,
    pipeline: PipelineLike,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    era_val: pd.Series,
    feature_cols: list[str],
    metrics: dict[str, float],
    meta_model_preds: np.ndarray | pd.Series | None = None,
    log_shap: bool = True,
    log_feature_report: bool = True,
    log_era_importance: bool = False,
    compute_fnc: bool | None = None,
    split: str = "holdout",
) -> None:
    """Log comprehensive XAI and backtest diagnostics to the active WandB run.

    Args:
        pipeline: Trained pipeline.
        X_val: Evaluation features (may include era column).
        y_val: Evaluation targets.
        era_val: Era labels aligned with X_val.
        feature_cols: Feature column names (must not include "era").
        metrics: Backtest metrics dict for this *split* (holdout or validation).
        meta_model_preds: Optional meta-model predictions for MMC logging and
            prediction neutralization. Series inputs are aligned to X_val by row ID.
        log_shap: If True, log universal feature importance (all model types).
        log_feature_report: If True, log per-era stability report via LightGBM proxy.
        log_era_importance: If True, log era-stratified importance from pipeline models
            (expensive — recommended only for best-trial diagnostics).
        compute_fnc: Whether to log FNC. Auto-detected from feature count when None.
        split: Data split label used in W&B keys (`holdout` or `validation`).
    """
    if not _wandb_active():
        return

    import wandb

    X_use = X_val[feature_cols] if feature_cols else X_val
    y_aligned = align_series_to_frame(X_use, y_val, name="target")
    era_aligned = align_series_to_frame(X_use, era_val, name="era")
    meta_array = _aligned_meta_model_predictions(X_use, meta_model_preds)
    preds = predict_with_optional_eras(
        pipeline,
        X_use,
        era_aligned,
        meta_model_preds=meta_array,
    )

    _log_per_era_correlation(y_aligned, preds, era_aligned, split=split)
    _log_prediction_diagnostics(y_aligned, preds, split=split)
    _log_feature_exposure(preds, X_use, era_aligned, split=split)

    if isinstance(pipeline, Pipeline) and len(pipeline.models) > 1:
        _log_ensemble_diagnostics(
            pipeline, X_val, feature_cols, y_aligned, era_aligned, split=split
        )
    elif isinstance(pipeline, MultiTargetPipeline) and len(pipeline._models) > 1:
        _log_ensemble_diagnostics(
            pipeline, X_val, feature_cols, y_aligned, era_aligned, split=split
        )

    if split == "validation" and (
        meta_array is not None
        or any(k in metrics for k in ("mmc", "mmc_sharpe", "payout_score"))
    ):
        _log_mmc_metrics_from_dict(wandb, metrics, split=split)

    use_fnc = compute_fnc
    if use_fnc is None:
        use_fnc = len(feature_cols) <= MAX_FNC_FEATURES
    if use_fnc and "fnc_sharpe" in metrics:
        wandb.log({_diag_key(split, "fnc_sharpe"): metrics["fnc_sharpe"]})

    if log_shap:
        from ..evaluation.shap_report import log_universal_feature_importance

        log_universal_feature_importance(
            pipeline,
            X_use,
            feature_cols=feature_cols,
            top_n=20,
            diagnostics_prefix=f"diagnostics/{split}",
        )

    if log_feature_report:
        _log_feature_report(
            X_use,
            y_aligned,
            era_aligned,
            feature_cols,
            split=split,
        )

    if log_era_importance:
        _log_era_stratified_importance(
            pipeline, X_use, feature_cols, era_aligned, split=split
        )


def _log_per_era_correlation(
    y_val: pd.Series, preds: np.ndarray, era_val: pd.Series, *, split: str
) -> None:
    import wandb

    per_era = per_era_correlation(y_val, preds, era_val, method="spearman").dropna()
    if per_era.empty:
        return

    cumulative = per_era.cumsum()
    cum_arr = cumulative.to_numpy(dtype=np.float64)
    peak_arr = np.maximum.accumulate(cum_arr)
    drawdown = pd.Series(peak_arr - cum_arr, index=per_era.index)

    table = wandb.Table(
        columns=[
            "era_index",
            "era",
            "correlation",
            "cumulative_correlation",
            "drawdown",
        ]
    )
    for idx, (era, corr) in enumerate(per_era.items()):
        table.add_data(
            idx,
            str(era),
            float(corr),
            float(cumulative.loc[era]),
            float(drawdown.loc[era]),
        )

    split_label = "train holdout" if split == "holdout" else "validation"
    wandb.log(
        {
            _diag_key(split, "per_era_correlation"): wandb.plot.line(
                table,
                "era_index",
                "correlation",
                title=f"Per-era Spearman correlation ({split_label})",
            ),
            _diag_key(split, "cumulative_correlation"): wandb.plot.line(
                table,
                "era_index",
                "cumulative_correlation",
                title=f"Cumulative per-era correlation ({split_label})",
            ),
            _diag_key(split, "drawdown_curve"): wandb.plot.line(
                table,
                "era_index",
                "drawdown",
                title=f"Drawdown from peak cumulative correlation ({split_label})",
            ),
        }
    )

    valid_corrs = per_era.to_numpy(dtype=np.float64)
    valid_corrs = valid_corrs[np.isfinite(valid_corrs)]
    if len(valid_corrs) >= 5:
        counts, edges = np.histogram(valid_corrs, bins=30, range=(-0.1, 0.1))
        mid = 0.5 * (edges[:-1] + edges[1:])
        dist_table = wandb.Table(columns=["bin_center", "count"])
        for m, c in zip(mid, counts, strict=False):
            dist_table.add_data(float(m), int(c))
        wandb.log(
            {
                _diag_key(split, "corr_distribution"): wandb.plot.bar(
                    dist_table,
                    "bin_center",
                    "count",
                    title=f"Distribution of per-era correlations ({split_label})",
                )
            }
        )


def _log_prediction_diagnostics(
    y_val: pd.Series, preds: np.ndarray, *, split: str
) -> None:
    import wandb

    split_label = "train holdout" if split == "holdout" else "validation"
    ranked = rank_normalize(preds)
    finite_ranked = ranked[np.isfinite(ranked)]
    if len(finite_ranked):
        fig = _new_figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.hist(
            finite_ranked,
            bins=_PRED_PLOT_BINS,
            range=(0.0, 1.0),
            color="steelblue",
            edgecolor="white",
            alpha=0.9,
        )
        uniform_ref = len(finite_ranked) / _PRED_PLOT_BINS
        ax.axhline(
            uniform_ref,
            color="tomato",
            linestyle="--",
            linewidth=1.2,
            label="uniform reference",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Rank-normalized prediction")
        ax.set_ylabel("Count")
        ax.set_title(f"Prediction distribution ({split_label})")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        _log_wandb_figure(wandb, _diag_key(split, "prediction_histogram"), fig)

    y_arr = y_val.to_numpy(dtype=np.float64)
    p_arr = np.asarray(preds, dtype=np.float64)
    y_plot, p_plot = _subsample_finite_pairs(
        y_arr, p_arr, max_points=MAX_HEXBIN_POINTS, seed=0
    )
    if len(y_plot):
        y_lo, y_hi = np.percentile(y_plot, [1, 99])
        p_lo, p_hi = np.percentile(p_plot, [1, 99])
        if y_lo == y_hi:
            y_lo, y_hi = y_lo - 1e-6, y_hi + 1e-6
        if p_lo == p_hi:
            p_lo, p_hi = p_lo - 1e-6, p_hi + 1e-6

        fig = _new_figure(figsize=(12, 4.5))
        axes = [fig.add_subplot(1, 2, i + 1) for i in range(2)]
        hb = axes[0].hexbin(
            y_plot,
            p_plot,
            gridsize=35,
            cmap="Blues",
            mincnt=1,
            extent=(y_lo, y_hi, p_lo, p_hi),
        )
        fig.colorbar(hb, ax=axes[0], label="count")
        axes[0].set_xlabel("Target (1–99 pct)")
        axes[0].set_ylabel("Raw prediction (1–99 pct)")
        axes[0].set_title(f"Pred vs target — hexbin (n={len(y_plot):,})")

        step = max(1, len(y_plot) // 2500)
        axes[1].scatter(
            y_plot[::step],
            p_plot[::step],
            alpha=0.2,
            s=6,
            c="steelblue",
            edgecolors="none",
            rasterized=True,
        )
        axes[1].set_xlim(y_lo, y_hi)
        axes[1].set_ylim(p_lo, p_hi)
        axes[1].set_xlabel("Target (1–99 pct)")
        axes[1].set_ylabel("Raw prediction (1–99 pct)")
        axes[1].set_title("Pred vs target — subsampled scatter")
        fig.tight_layout()
        _log_wandb_figure(wandb, _diag_key(split, "pred_vs_target_scatter"), fig)

    residuals = y_arr - p_arr
    finite = residuals[np.isfinite(residuals)]
    if len(finite):
        wandb.log(
            {
                _diag_key(split, "residual_mean"): float(np.mean(finite)),
                _diag_key(split, "residual_std"): float(np.std(finite, ddof=0)),
                _diag_key(split, "residual_mae"): float(np.mean(np.abs(finite))),
            }
        )


def _log_feature_exposure(
    preds: np.ndarray, features: pd.DataFrame, eras: pd.Series, *, split: str
) -> None:
    import wandb

    split_label = "train holdout" if split == "holdout" else "validation"
    summary = _feature_exposure_summary(preds, features, eras)
    wandb.log(
        {
            _diag_key(split, "feature_exposure_max"): summary["max_mean_abs_corr"],
            _diag_key(split, "feature_exposure_mean"): summary["mean_abs_corr"],
        }
    )
    if summary["top"]:
        _log_horizontal_bar_chart(
            wandb,
            labels=[row["feature"] for row in summary["top"]],
            values=[row["mean_abs_corr"] for row in summary["top"]],
            key=_diag_key(split, "feature_exposure_bar"),
            title=f"Feature exposure ({split_label}, top 15)",
            xlabel="Mean |corr| with predictions",
        )


def _log_ensemble_diagnostics(
    pipeline: PipelineLike,
    X_val: pd.DataFrame,
    feature_cols: list[str],
    y_val: pd.Series,
    era_val: pd.Series,
    *,
    split: str,
) -> None:
    import wandb

    from ..evaluation.ensemble_diagnostics import compute_ensemble_diagnostics

    split_label = "train holdout" if split == "holdout" else "validation"
    oof = _collect_model_predictions(pipeline, X_val, feature_cols)
    weights = None
    if isinstance(pipeline, Pipeline) and pipeline.ensemble_method == "weighted":
        w = pipeline._ensemble.params.get("weights")
        if w is not None:
            weights = np.asarray(w, dtype=np.float64)
    elif isinstance(pipeline, MultiTargetPipeline):
        weights = multitarget_blend_weights(pipeline)

    diag = compute_ensemble_diagnostics(
        oof,
        y_val.to_numpy(dtype=np.float64),
        era_val,
        weights=weights,
    )
    wandb.log(
        {
            _diag_key(split, "effective_model_count"): diag["effective_model_count"],
            _diag_key(split, "mean_pairwise_correlation"): diag[
                "mean_pairwise_correlation"
            ],
        }
    )
    names = diag["model_names"]
    corr = diag["correlation_matrix"]
    _log_correlation_heatmap(
        wandb,
        names,
        corr,
        _diag_key(split, "ensemble_correlation_heatmap"),
        title=f"Model prediction correlations ({split_label})",
    )
    if len(names) > 1:
        pairs: list[str] = []
        pair_corrs: list[float] = []
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if j > i:
                    pairs.append(f"{a} → {b}")
                    pair_corrs.append(float(corr[a][b]))
        _log_horizontal_bar_chart(
            wandb,
            labels=pairs,
            values=pair_corrs,
            key=_diag_key(split, "ensemble_correlation_bar"),
            title=f"Model pair correlations ({split_label})",
            xlabel="Spearman correlation",
        )


def _log_feature_report(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    era_val: pd.Series,
    feature_cols: list[str],
    *,
    split: str,
    top_n: int = 20,
) -> None:
    """Log per-era feature stability report (LightGBM proxy) to WandB.

    Calls compute_feature_report and logs horizontal bar charts for mean
    importance, most stable features, and least stable features.
    Silently skips if lightgbm is not installed.
    """
    if not _wandb_active():
        return

    import wandb

    try:
        from ..evaluation.feature_report import compute_feature_report
    except ImportError:
        return

    try:
        X_feat = X_val[feature_cols] if feature_cols else X_val
        report = compute_feature_report(X_feat, y_val, era_val, top_n=top_n)
    except Exception:
        return

    split_label = "train holdout" if split == "holdout" else "validation"
    wandb.log({_diag_key(split, "feature_n_eras_used"): report["n_eras_used"]})

    if report["top_by_mean"]:
        _log_horizontal_bar_chart(
            wandb,
            labels=[row["feature"] for row in report["top_by_mean"]],
            values=[row["mean_importance"] for row in report["top_by_mean"]],
            key=_diag_key(split, "feature_importance_mean_bar"),
            title=f"Top features by mean importance ({split_label})",
            xlabel="Mean importance",
        )

    if report["top_by_stability"]:
        _log_horizontal_bar_chart(
            wandb,
            labels=[row["feature"] for row in report["top_by_stability"]],
            values=[row["stability"] for row in report["top_by_stability"]],
            key=_diag_key(split, "feature_stability_bar"),
            title=f"Most stable features ({split_label})",
            xlabel="Stability",
        )

    if report["bottom_by_stability"]:
        _log_horizontal_bar_chart(
            wandb,
            labels=[row["feature"] for row in report["bottom_by_stability"]],
            values=[row["stability"] for row in report["bottom_by_stability"]],
            key=_diag_key(split, "feature_worst_stability_bar"),
            title=f"Least stable features ({split_label})",
            xlabel="Stability",
        )


def _log_era_stratified_importance(
    pipeline: PipelineLike,
    X_val: pd.DataFrame,
    feature_cols: list[str],
    era_val: pd.Series,
    *,
    split: str,
    top_n: int = 20,
    max_eras: int = 30,
) -> None:
    """Log era-stratified feature importance from the actual trained pipeline models.

    Samples up to max_eras eras, computes universal feature importance on each
    era slice, then summarizes stability (mean/std ratio) and logs a heatmap table.

    Args:
        pipeline: Trained pipeline.
        X_val: Validation features (pre-selected to feature_cols).
        feature_cols: Feature column names.
        era_val: Era labels aligned with X_val.
        top_n: Number of top features to include in the heatmap.
        max_eras: Maximum eras to sample (keeps runtime bounded).
    """
    if not _wandb_active():
        return

    import wandb

    from ..evaluation.shap_report import compute_universal_feature_importance

    e_arr = np.asarray(era_val.to_numpy())
    unique_eras = sorted(pd.unique(e_arr), key=str)

    if len(unique_eras) > max_eras:
        rng = np.random.default_rng(42)
        unique_eras = list(rng.choice(unique_eras, size=max_eras, replace=False))

    era_imps: list[dict[str, float]] = []
    era_labels: list[str] = []

    for era in unique_eras:
        mask = e_arr == era
        if mask.sum() < _ERA_IMPORTANCE_MIN_ROWS:
            continue
        X_era = X_val[mask]
        imp, _ = compute_universal_feature_importance(
            pipeline, X_era, feature_cols=feature_cols, top_n=top_n
        )
        if imp:
            era_imps.append(imp)
            era_labels.append(str(era))

    if not era_imps:
        return

    all_features = sorted(
        {f for imp in era_imps for f in imp},
        key=lambda f: -float(np.mean([imp.get(f, 0.0) for imp in era_imps])),
    )[:top_n]

    imp_matrix = np.array([[imp.get(f, 0.0) for f in all_features] for imp in era_imps])
    mean_imp = imp_matrix.mean(axis=0)
    std_imp = imp_matrix.std(axis=0, ddof=0)
    stability = mean_imp / (std_imp + 1e-10)

    split_label = "train holdout" if split == "holdout" else "validation"
    _log_horizontal_bar_chart(
        wandb,
        labels=all_features,
        values=[float(v) for v in stability],
        key=_diag_key(split, "era_importance_stability_bar"),
        title=f"Era-stratified importance stability ({split_label})",
        xlabel="Stability",
    )

    xs = list(range(len(era_labels)))
    ys = [[float(imp.get(f, 0.0)) for imp in era_imps] for f in all_features]
    wandb.log(
        {
            _diag_key(split, "era_importance_over_time"): wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=all_features,
                title=f"Feature importance across eras ({split_label})",
                xname="era_index",
            ),
        }
    )
