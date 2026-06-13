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
    compute_fnc: bool | None = None,
) -> None:
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
        from ..evaluation.shap_report import log_xgboost_shap_importance

        log_xgboost_shap_importance(pipeline, X_use, top_n=20)


def _log_per_era_correlation(
    y_val: pd.Series, preds: np.ndarray, era_val: pd.Series
) -> None:
    import wandb

    per_era = per_era_correlation(y_val, preds, era_val, method="spearman").dropna()
    if per_era.empty:
        return

    cumulative = per_era.cumsum()
    table = wandb.Table(columns=["era", "correlation", "cumulative_correlation"])
    for era, corr in per_era.items():
        table.add_data(str(era), float(corr), float(cumulative.loc[era]))

    wandb.log(
        {
            "diagnostics/per_era_correlation_table": table,
            "diagnostics/per_era_correlation": wandb.plot.line(
                table, "era", "correlation", title="Per-era correlation"
            ),
            "diagnostics/cumulative_correlation": wandb.plot.line(
                table,
                "era",
                "cumulative_correlation",
                title="Cumulative per-era correlation",
            ),
        }
    )


def _log_prediction_diagnostics(y_val: pd.Series, preds: np.ndarray) -> None:
    import wandb

    ranked = rank_normalize(preds)
    hist_table = wandb.Table(columns=["rank_normalized_prediction"])
    for value in ranked[np.isfinite(ranked)]:
        hist_table.add_data(float(value))
    wandb.log(
        {
            "diagnostics/prediction_histogram": wandb.plot.histogram(
                hist_table,
                "rank_normalized_prediction",
                title="Rank-normalized predictions",
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
        wandb.log({"diagnostics/feature_exposure_top": table})


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
    wandb.log({"diagnostics/ensemble_correlation_matrix": table})
