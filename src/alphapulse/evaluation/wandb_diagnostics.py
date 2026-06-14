from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.metrics import per_era_correlation, rank_normalize
from ..pipeline.multihead import MultiHeadPipeline
from ..pipeline.pipeline import Pipeline
from ..pipeline.row_utils import protected_metadata_frame

MAX_SCATTER_POINTS = 5000
MAX_HEXBIN_POINTS = 10_000
FEATURE_EXPOSURE_TOP_N = 15
MAX_FNC_FEATURES = 200
_ERA_IMPORTANCE_MIN_ROWS = 10


def _wandb_active() -> bool:
    try:
        import wandb

        return wandb.run is not None
    except ImportError:
        return False


def _collect_model_predictions(
    pipeline: Pipeline | MultiHeadPipeline,
    X_val: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, np.ndarray]:
    if isinstance(pipeline, MultiHeadPipeline):
        preds = pipeline.predict(X_val[feature_cols] if feature_cols else X_val)
        return {"ensemble": preds}

    X_feat = X_val[feature_cols] if feature_cols else X_val
    era_meta = protected_metadata_frame(X_feat)
    X_t = pipeline._preprocess(X_feat, era_meta)
    X_numeric = X_t.select_dtypes(include=[np.number])

    if len(pipeline.models) == 1:
        return {pipeline.models[0].name: pipeline.models[0].predict(X_numeric)}
    return {m.name: m.predict(X_numeric) for m in pipeline.models}


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


def log_experiment_diagnostics(
    *,
    pipeline: Pipeline | MultiHeadPipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    era_val: pd.Series,
    feature_cols: list[str],
    metrics: dict[str, float],
    meta_model_preds: np.ndarray | None = None,
    log_shap: bool = True,
    log_feature_report: bool = True,
    log_era_importance: bool = False,
    compute_fnc: bool | None = None,
) -> None:
    """Log comprehensive XAI and backtest diagnostics to the active WandB run.

    Args:
        pipeline: Trained pipeline.
        X_val: Validation features (may include era column).
        y_val: Validation targets.
        era_val: Era labels aligned with X_val.
        feature_cols: Feature column names (must not include "era").
        metrics: Backtest metrics dict.
        meta_model_preds: Optional meta-model predictions for MMC logging.
        log_shap: If True, log universal feature importance (all model types).
        log_feature_report: If True, log per-era stability report via LightGBM proxy.
        log_era_importance: If True, log era-stratified importance from pipeline models
            (expensive — recommended only for best-trial diagnostics).
        compute_fnc: Whether to log FNC. Auto-detected from feature count when None.
    """
    if not _wandb_active():
        return

    import wandb

    X_use = X_val[feature_cols] if feature_cols else X_val
    neutralize = getattr(pipeline, "neutralize_proportion", 0)
    eras_for_predict = era_val if neutralize > 0 else None
    if isinstance(pipeline, Pipeline):
        preds = pipeline.predict(X_use, eras=eras_for_predict)
    else:
        preds = pipeline.predict(X_val[feature_cols] if feature_cols else X_val)

    _log_per_era_correlation(y_val, preds, era_val)
    _log_prediction_diagnostics(y_val, preds)
    _log_feature_exposure(preds, X_use, era_val)

    if isinstance(pipeline, Pipeline) and len(pipeline.models) > 1:
        _log_ensemble_diagnostics(pipeline, X_val, feature_cols, y_val, era_val)

    if meta_model_preds is not None:
        wandb.log(
            {
                "diagnostics/mmc_sharpe": metrics.get("mmc_sharpe"),
                "diagnostics/payout_score": metrics.get("payout_score"),
            }
        )

    use_fnc = compute_fnc
    if use_fnc is None:
        use_fnc = len(feature_cols) <= MAX_FNC_FEATURES
    if use_fnc and "fnc_sharpe" in metrics:
        wandb.log({"diagnostics/fnc_sharpe": metrics["fnc_sharpe"]})

    if log_shap:
        from ..evaluation.shap_report import log_universal_feature_importance

        log_universal_feature_importance(
            pipeline, X_use, feature_cols=feature_cols, top_n=20
        )

    if log_feature_report:
        _log_feature_report(X_use, y_val, era_val, feature_cols)

    if log_era_importance:
        _log_era_stratified_importance(pipeline, X_use, feature_cols, era_val)


def _log_per_era_correlation(
    y_val: pd.Series, preds: np.ndarray, era_val: pd.Series
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
            "era_index", "era", "correlation", "cumulative_correlation", "drawdown"
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

    wandb.log(
        {
            "diagnostics/per_era_correlation_table": table,
            "diagnostics/per_era_correlation": wandb.plot.line(
                table, "era_index", "correlation", title="Per-era Spearman correlation"
            ),
            "diagnostics/cumulative_correlation": wandb.plot.line(
                table,
                "era_index",
                "cumulative_correlation",
                title="Cumulative per-era correlation",
            ),
            "diagnostics/drawdown_curve": wandb.plot.line(
                table,
                "era_index",
                "drawdown",
                title="Drawdown from peak cumulative correlation",
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
                "diagnostics/corr_distribution": wandb.plot.bar(
                    dist_table,
                    "bin_center",
                    "count",
                    title="Distribution of per-era correlations",
                )
            }
        )


def _log_prediction_diagnostics(y_val: pd.Series, preds: np.ndarray) -> None:
    import wandb

    ranked = rank_normalize(preds)
    finite_ranked = ranked[np.isfinite(ranked)]
    counts, edges = np.histogram(finite_ranked, bins=50)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    hist_table = wandb.Table(columns=["bin_center", "count"])
    for mid, cnt in zip(midpoints, counts, strict=False):
        hist_table.add_data(float(mid), int(cnt))
    wandb.log(
        {
            "diagnostics/prediction_histogram": wandb.plot.bar(
                hist_table,
                "bin_center",
                "count",
                title="Rank-normalized prediction distribution (50 bins)",
            )
        }
    )

    n = min(len(y_val), MAX_SCATTER_POINTS)
    if n < len(y_val):
        idx = np.random.default_rng(0).choice(len(y_val), size=n, replace=False)
        y_sample = y_val.iloc[idx]
        p_sample = preds[idx]
    else:
        y_sample = y_val
        p_sample = preds

    scatter = wandb.Table(columns=["target", "prediction"])
    for yt, pp in zip(y_sample, p_sample, strict=False):
        if np.isfinite(yt) and np.isfinite(pp):
            scatter.add_data(float(yt), float(pp))
    wandb.log(
        {
            "diagnostics/pred_vs_target_scatter": wandb.plot.scatter(
                scatter,
                "target",
                "prediction",
                title="Predictions vs target (sampled)",
            )
        }
    )

    residuals = y_val.to_numpy(dtype=np.float64) - preds
    finite = residuals[np.isfinite(residuals)]
    if len(finite):
        wandb.log(
            {
                "diagnostics/residual_mean": float(np.mean(finite)),
                "diagnostics/residual_std": float(np.std(finite, ddof=0)),
                "diagnostics/residual_mae": float(np.mean(np.abs(finite))),
            }
        )


def _log_feature_exposure(
    preds: np.ndarray, features: pd.DataFrame, eras: pd.Series
) -> None:
    import wandb

    summary = _feature_exposure_summary(preds, features, eras)
    wandb.log(
        {
            "diagnostics/feature_exposure_max": summary["max_mean_abs_corr"],
            "diagnostics/feature_exposure_mean": summary["mean_abs_corr"],
        }
    )
    if summary["top"]:
        table = wandb.Table(columns=["feature", "mean_abs_corr"])
        for row in summary["top"]:
            table.add_data(row["feature"], row["mean_abs_corr"])
        wandb.log(
            {
                "diagnostics/feature_exposure_top": table,
                "diagnostics/feature_exposure_bar": wandb.plot.bar(
                    table,
                    "feature",
                    "mean_abs_corr",
                    title="Feature exposure (top 15 by mean |corr| with predictions)",
                ),
            }
        )


def _log_ensemble_diagnostics(
    pipeline: Pipeline | MultiHeadPipeline,
    X_val: pd.DataFrame,
    feature_cols: list[str],
    y_val: pd.Series,
    era_val: pd.Series,
) -> None:
    import wandb

    from ..evaluation.ensemble_diagnostics import compute_ensemble_diagnostics

    oof = _collect_model_predictions(pipeline, X_val, feature_cols)
    weights = None
    if isinstance(pipeline, Pipeline) and pipeline.ensemble_method == "weighted":
        w = pipeline._ensemble.params.get("weights")
        if w is not None:
            weights = np.asarray(w, dtype=np.float64)

    diag = compute_ensemble_diagnostics(
        oof,
        y_val.to_numpy(dtype=np.float64),
        era_val,
        weights=weights,
    )
    wandb.log(
        {
            "diagnostics/effective_model_count": diag["effective_model_count"],
            "diagnostics/mean_pairwise_correlation": diag["mean_pairwise_correlation"],
        }
    )
    names = diag["model_names"]
    corr = diag["correlation_matrix"]
    table = wandb.Table(columns=["model_a", "model_b", "correlation"])
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j >= i:
                table.add_data(a, b, corr[a][b])

    pair_table = wandb.Table(columns=["pair", "correlation"])
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j > i:
                pair_table.add_data(f"{a}→{b}", corr[a][b])

    logged: dict[str, Any] = {"diagnostics/ensemble_correlation_matrix": table}
    if len(names) > 1:
        logged["diagnostics/ensemble_correlation_bar"] = wandb.plot.bar(
            pair_table,
            "pair",
            "correlation",
            title="Model pair correlations (lower = more diverse ensemble)",
        )
    wandb.log(logged)


def _log_feature_report(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    era_val: pd.Series,
    feature_cols: list[str],
    *,
    top_n: int = 20,
) -> None:
    """Log per-era feature stability report (LightGBM proxy) to WandB.

    Calls compute_feature_report and logs three tables: top features by mean
    importance, top features by era stability, and worst features by stability.
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

    wandb.log({"diagnostics/feature_n_eras_used": report["n_eras_used"]})

    if report["top_by_mean"]:
        table_mean = wandb.Table(columns=["feature", "mean_importance"])
        for row in report["top_by_mean"]:
            table_mean.add_data(row["feature"], row["mean_importance"])
        wandb.log(
            {
                "diagnostics/feature_top_by_mean": table_mean,
                "diagnostics/feature_importance_mean_bar": wandb.plot.bar(
                    table_mean,
                    "feature",
                    "mean_importance",
                    title="Top features by mean importance (LightGBM proxy, per era)",
                ),
            }
        )

    if report["top_by_stability"]:
        table_stab = wandb.Table(columns=["feature", "stability", "mean_importance"])
        for row in report["top_by_stability"]:
            table_stab.add_data(
                row["feature"], row["stability"], row["mean_importance"]
            )
        wandb.log(
            {
                "diagnostics/feature_top_by_stability": table_stab,
                "diagnostics/feature_stability_bar": wandb.plot.bar(
                    table_stab,
                    "feature",
                    "stability",
                    title="Most stable features across eras (mean/std ratio)",
                ),
            }
        )

    if report["bottom_by_stability"]:
        table_worst = wandb.Table(columns=["feature", "stability", "mean_importance"])
        for row in report["bottom_by_stability"]:
            table_worst.add_data(
                row["feature"], row["stability"], row["mean_importance"]
            )
        wandb.log(
            {
                "diagnostics/feature_worst_stability": table_worst,
                "diagnostics/feature_worst_stability_bar": wandb.plot.bar(
                    table_worst,
                    "feature",
                    "stability",
                    title="Least stable features across eras (worst to prune)",
                ),
            }
        )


def _log_era_stratified_importance(
    pipeline: Pipeline | MultiHeadPipeline,
    X_val: pd.DataFrame,
    feature_cols: list[str],
    era_val: pd.Series,
    *,
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

    imp_matrix = np.array(
        [[imp.get(f, 0.0) for f in all_features] for imp in era_imps]
    )
    mean_imp = imp_matrix.mean(axis=0)
    std_imp = imp_matrix.std(axis=0, ddof=0)
    stability = mean_imp / (std_imp + 1e-10)

    stab_table = wandb.Table(
        columns=["feature", "mean_importance", "std_importance", "stability"]
    )
    for feat, mean_v, std_v, stab_v in zip(
        all_features, mean_imp, std_imp, stability, strict=False
    ):
        stab_table.add_data(feat, float(mean_v), float(std_v), float(stab_v))
    wandb.log(
        {
            "diagnostics/era_importance_stability": stab_table,
            "diagnostics/era_importance_stability_bar": wandb.plot.bar(
                stab_table,
                "feature",
                "stability",
                title="Era-stratified importance stability (mean/std)",
            ),
        }
    )

    xs = list(range(len(era_labels)))
    ys = [[float(imp.get(f, 0.0)) for imp in era_imps] for f in all_features]
    wandb.log(
        {
            "diagnostics/era_importance_over_time": wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=all_features,
                title="Feature importance across eras (each line = one feature)",
                xname="era_index",
            ),
        }
    )
