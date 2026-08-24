import gc
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..evaluation.backtester import Backtester
from ..evaluation.era_split import (
    HPO_FAST_HOLDOUT_ERAS,
    HPO_FAST_MAX_TRAIN_ERAS,
    HPO_FAST_WF_N_SPLITS,
    WF_MIN_TRAIN_ERAS,
    WF_N_PURGE,
    WF_N_SPLITS,
    EraSplitEvaluator,
)
from ..experiments.split import internal_val_split
from ..features.catalog import FeatureCatalog, load_feature_catalog
from ..models.diffusion_augmenter import SyntheticDataAugmenter
from ..pipeline.ensemble import needs_internal_val_for_ensemble
from ..utils import set_global_seed
from ..validation.purge import effective_purge_eras
from .builder import (
    build_multi_target_from_config,
    build_pipeline_or_multi,
)
from .feature_routing import (
    merge_routing_into_pipeline_config,
    resolve_feature_routing,
)
from .search_space import get_train_kwargs_from_flat, resolve_flat_config
from .target_strategy import strategy_from_flat


@dataclass(frozen=True)
class TrialResult:
    trial_number: int
    sharpe: float
    metrics: dict[str, float]
    model_type: str
    elapsed_seconds: float
    params: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    corr_sharpe: float = float("-inf")
    mmc_sharpe: float | None = None
    payout_score: float | None = None


def _effective_purge_eras(configured: int, strategy: Any) -> int:
    targets = [strategy.primary_target, *strategy.auxiliary_targets]
    return effective_purge_eras(configured, targets)


def ray_model_seed(config: dict[str, Any], data_seed: int) -> int:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return data_seed + int.from_bytes(digest[:4], byteorder="big")


def ray_trainable(
    config: dict[str, Any], *, data_seed: int = 42, **kwargs: Any
) -> dict[str, float]:
    model_seed = ray_model_seed(config, data_seed)
    set_global_seed(model_seed)
    trial_config = dict(config)
    trial_config["data_seed"] = data_seed
    trial_config["model_seed"] = model_seed
    return run_trial(
        trial_config,
        seed=model_seed,
        data_seed=data_seed,
        **kwargs,
    )


def _resolve_pipeline_cfg(
    config: dict[str, Any],
    catalog: FeatureCatalog | None,
) -> tuple[dict[str, Any], dict[str, list[str]] | None, list[str] | None]:
    pipeline_cfg = resolve_flat_config(config)
    if not config.get("use_feature_routing"):
        return pipeline_cfg, None, None

    cat = catalog
    if cat is None:
        data_dir = config.get("_data_dir")
        if not data_dir:
            raise ValueError("_data_dir required when use_feature_routing is enabled")
        cat = load_feature_catalog(data_dir)

    routing = resolve_feature_routing(config, cat)
    pipeline_cfg = merge_routing_into_pipeline_config(pipeline_cfg, routing)
    feature_groups = routing.feature_groups or None
    feature_columns = routing.feature_columns or None
    return pipeline_cfg, feature_groups, feature_columns


def _apply_synthetic_augmentation(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    flat_config: dict[str, Any],
    feature_cols: list[str],
    seed: int | None,
) -> tuple[pd.DataFrame, pd.Series]:
    aug = SyntheticDataAugmenter(
        top_fraction=float(flat_config.get("augmenter_top_fraction", 0.10)),
        n_synthetic=int(flat_config.get("augmenter_n_synthetic", 500)),
        backend=str(flat_config.get("augmenter_backend", "auto")),
        seed=int(seed or 42),
    )
    feat_cols = [c for c in feature_cols if c in X_tr.columns]
    feat = X_tr[feat_cols]
    aug.fit(feat, y_tr)
    X_syn, y_syn = aug.generate()
    syn_df = X_syn.copy()
    if "era" in X_tr.columns:
        rng = np.random.default_rng(seed)
        train_eras = X_tr["era"].unique()
        syn_df["era"] = rng.choice(train_eras, size=len(syn_df))
    aligned = syn_df.reindex(columns=X_tr.columns, fill_value=0.0)
    X_out = pd.concat([X_tr.reset_index(drop=True), aligned], ignore_index=True)
    y_out = pd.concat(
        [y_tr.reset_index(drop=True), y_syn.reset_index(drop=True)],
        ignore_index=True,
    )
    y_out.name = y_tr.name or y_syn.name or "target"
    if len(X_out) != len(y_out):
        raise ValueError(
            f"Augmented X/y length mismatch: {len(X_out)} rows vs {len(y_out)} labels"
        )
    return X_out, y_out


def _is_multi_target(flat_config: dict[str, Any] | None) -> bool:
    if not flat_config:
        return False
    return flat_config.get("target_mode") == "multi_blend" and bool(
        flat_config.get("auxiliary_targets")
    )


def _fit_pipeline(
    pipeline_cfg: dict[str, Any],
    feature_cols: list[str],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    train_kwargs: dict[str, Any],
    flat_config: dict[str, Any] | None = None,
    seed: int | None = None,
    feature_groups: dict[str, list[str]] | None = None,
    targets_df: pd.DataFrame | None = None,
) -> Any:
    logger.info("Building trial pipeline")
    X_fit_src = X_tr
    y_fit_src = y_tr
    targets_fit = targets_df
    if (
        flat_config
        and flat_config.get("use_augmentation")
        and not _is_multi_target(flat_config)
    ):
        X_fit_src, y_fit_src = _apply_synthetic_augmentation(
            X_tr, y_tr, flat_config, feature_cols, seed
        )
        if targets_fit is not None:
            targets_fit = targets_fit.loc[X_tr.index].reset_index(drop=True)
            aug_targets = targets_fit.copy()
            targets_fit = pd.concat(
                [aug_targets, aug_targets.iloc[:0]], ignore_index=True
            )

    era_col = X_fit_src["era"] if "era" in X_fit_src.columns else None
    force_internal = needs_internal_val_for_ensemble(pipeline_cfg)

    if _is_multi_target(flat_config):
        assert flat_config is not None
        pipeline: Any = build_multi_target_from_config(
            pipeline_cfg,
            flat_config,
            feature_columns=feature_cols,
            feature_groups=feature_groups,
        )
        X_fit, _, X_val_inner, _ = internal_val_split(
            X_fit_src,
            y_fit_src,
            era_train=era_col,
            force_internal=force_internal,
        )
        targets_split = (
            targets_fit.loc[X_fit_src.index] if targets_fit is not None else None
        )
        targets_train = (
            targets_split.loc[X_fit.index] if targets_split is not None else None
        )
        targets_val = (
            targets_split.loc[X_val_inner.index]
            if targets_split is not None and X_val_inner is not None
            else None
        )
        era_train_fit = era_col.loc[X_fit.index] if era_col is not None else None
        era_val_fit = (
            era_col.loc[X_val_inner.index]
            if era_col is not None and X_val_inner is not None
            else None
        )
        pipeline.fit(
            X_fit.drop(columns=["era"], errors="ignore"),
            targets_train,
            X_val=X_val_inner.drop(columns=["era"], errors="ignore")
            if X_val_inner is not None
            else None,
            targets_val=targets_val,
            era_train=era_train_fit,
            era_val=era_val_fit,
            **train_kwargs,
        )
        return pipeline

    pipeline = build_pipeline_or_multi(
        pipeline_cfg,
        feature_columns=feature_cols,
        feature_groups=feature_groups,
    )
    logger.info("Trial pipeline built; preparing internal split")
    X_fit, y_fit, X_val_inner, y_val_inner = internal_val_split(
        X_fit_src, y_fit_src, era_train=era_col, force_internal=force_internal
    )
    logger.info("Internal split ready; fitting trial pipeline")
    era_val_fit = (
        era_col.loc[X_val_inner.index]
        if era_col is not None and X_val_inner is not None
        else None
    )
    pipeline.fit(
        X_fit,
        y_fit,
        X_val=X_val_inner,
        y_val=y_val_inner,
        era_val=era_val_fit,
        **train_kwargs,
    )
    if flat_config is not None and hasattr(pipeline, "ensemble_weights"):
        weights = pipeline.ensemble_weights
        if weights is not None:
            flat_config["ensemble_weights"] = weights
    return pipeline


_HOLDOUT_METRIC_KEYS = (
    "corr_sharpe",
    "numerai_corr_mean",
    "numerai_corr_std",
    "numerai_corr_sharpe",
    "numerai_corr_max_drawdown",
    "numerai_corr_pct_positive_eras",
    "mean_per_era_correlation",
    "std_per_era_correlation",
    "max_drawdown",
    "pct_positive_eras",
    "n_valid_eras",
)


def _merge_validation_mmc_metrics(
    metrics: dict[str, float],
    *,
    pipeline: Any,
    data_dir: Path,
    feature_cols: list[str],
    target_col: str,
    train_subsample: float,
    seed: int | None,
    preloaded_frame: tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray]
    | None = None,
    frame_preloaded: bool = False,
) -> tuple[
    dict[str, float],
    tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray] | None,
]:
    frame = preloaded_frame
    if not frame_preloaded:
        from ..experiments.data import load_mmc_validation_frame

        frame = load_mmc_validation_frame(
            data_dir,
            feature_cols=feature_cols,
            target_col=target_col,
            train_subsample=train_subsample,
            seed=seed or 42,
        )
    if frame is None:
        return metrics, None

    X_val, y_val, era_val, meta_preds = frame
    mmc_metrics = Backtester(pipeline, feature_columns=feature_cols).evaluate(
        X_val,
        y_val,
        era_val,
        meta_model_preds=meta_preds,
    )
    merged = dict(metrics)
    for key in _HOLDOUT_METRIC_KEYS:
        if key in merged:
            merged[f"holdout_{key}"] = merged[key]
    for key in (
        "mmc",
        "mmc_sharpe",
        "payout_score",
        "numerai_mmc_mean",
        "numerai_mmc_std",
        "numerai_mmc_sharpe",
        "weighted_corr_mmc_mean",
        "weighted_corr_mmc_std",
        "weighted_corr_mmc_sharpe",
    ):
        value = mmc_metrics.get(key)
        if value is not None and np.isfinite(value):
            merged[key] = float(value)
    for key in _HOLDOUT_METRIC_KEYS:
        value = mmc_metrics.get(key)
        if value is not None and np.isfinite(value):
            merged[f"val_{key}"] = float(value)
    return merged, frame


def _holdout_metrics_for_diagnostics(metrics: dict[str, float]) -> dict[str, float]:
    scoped: dict[str, float] = {}
    for key in _HOLDOUT_METRIC_KEYS:
        holdout_key = f"holdout_{key}"
        if holdout_key in metrics:
            scoped[key] = float(metrics[holdout_key])
        elif key in metrics:
            scoped[key] = float(metrics[key])
    return scoped


def _validation_metrics_for_diagnostics(metrics: dict[str, float]) -> dict[str, float]:
    scoped: dict[str, float] = {}
    for key in _HOLDOUT_METRIC_KEYS:
        val_key = f"val_{key}"
        if val_key in metrics:
            scoped[key] = float(metrics[val_key])
    for key in (
        "mmc",
        "mmc_sharpe",
        "payout_score",
        "numerai_mmc_mean",
        "numerai_mmc_std",
        "numerai_mmc_sharpe",
        "weighted_corr_mmc_mean",
        "weighted_corr_mmc_std",
        "weighted_corr_mmc_sharpe",
    ):
        if key in metrics:
            scoped[key] = float(metrics[key])
    return scoped


def _evaluate_holdout(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    era_train: pd.Series,
    feature_cols: list[str],
    pipeline_cfg: dict[str, Any],
    train_kwargs: dict[str, Any],
    holdout_eras: int = HPO_FAST_HOLDOUT_ERAS,
    purge_eras: int = WF_N_PURGE,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
    data_seed: int | None = None,
    feature_groups: dict[str, list[str]] | None = None,
    targets_df: pd.DataFrame | None = None,
    mmc_frame: tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray] | None = None,
    mmc_frame_preloaded: bool = False,
) -> dict[str, float]:
    logger.info("Preparing purged holdout evaluation")
    eras_sorted = sorted(era_train.unique(), key=str)
    min_train = WF_MIN_TRAIN_ERAS
    n_holdout = min(holdout_eras, max(5, len(eras_sorted) // 5))
    if len(eras_sorted) <= min_train + n_holdout:
        n_holdout = max(1, len(eras_sorted) // 5)

    holdout_start = len(eras_sorted) - n_holdout
    train_end = max(0, holdout_start - purge_eras)
    holdout_set = set(eras_sorted[holdout_start:])
    train_set = set(eras_sorted[:train_end])
    train_mask = era_train.isin(train_set)
    if not train_mask.any():
        return {"corr_sharpe": float("nan"), "mean_per_era_correlation": float("nan")}

    logger.info("Purged holdout split ready; fitting model")
    pipeline = _fit_pipeline(
        pipeline_cfg,
        feature_cols,
        X_train.loc[train_mask],
        y_train.loc[train_mask],
        train_kwargs,
        flat_config=config,
        seed=seed,
        feature_groups=feature_groups,
        targets_df=targets_df.loc[train_mask] if targets_df is not None else None,
    )
    ho_mask = era_train.isin(holdout_set)
    X_ho = X_train.loc[ho_mask]
    y_ho = y_train.loc[ho_mask]
    era_ho = era_train.loc[ho_mask]

    metrics = Backtester(pipeline, feature_columns=feature_cols).evaluate(
        X_ho,
        y_ho,
        era_ho,
    )
    if config and config.get("_data_dir"):
        data_dir = Path(str(config["_data_dir"]))
        target_col = str(config.get("primary_target", "target"))
        train_subsample = float(config.get("_train_subsample", 1.0))
        metrics, mmc_frame = _merge_validation_mmc_metrics(
            metrics,
            pipeline=pipeline,
            data_dir=data_dir,
            feature_cols=feature_cols,
            target_col=target_col,
            train_subsample=train_subsample,
            seed=data_seed if data_seed is not None else seed,
            preloaded_frame=mmc_frame,
            frame_preloaded=mmc_frame_preloaded,
        )
    if config and config.get("log_wandb_diagnostics"):
        from ..evaluation.wandb_diagnostics import log_experiment_diagnostics

        log_experiment_diagnostics(
            pipeline=pipeline,
            X_val=X_ho,
            y_val=y_ho,
            era_val=era_ho,
            feature_cols=feature_cols,
            metrics=_holdout_metrics_for_diagnostics(metrics),
            log_shap=bool(config.get("wandb_log_shap", True)),
            compute_fnc=False,
            split="holdout",
        )
        if mmc_frame is not None:
            X_val, y_val, era_val, meta_preds = mmc_frame
            log_experiment_diagnostics(
                pipeline=pipeline,
                X_val=X_val,
                y_val=y_val,
                era_val=era_val,
                feature_cols=feature_cols,
                metrics=_validation_metrics_for_diagnostics(metrics),
                meta_model_preds=meta_preds,
                log_shap=False,
                log_feature_report=False,
                log_era_importance=False,
                compute_fnc=False,
                split="validation",
            )
    del pipeline
    gc.collect()
    return metrics


def run_trial(
    config: dict[str, Any],
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    era_train: pd.Series,
    feature_cols: list[str],
    seed: int | None = None,
    data_seed: int | None = None,
    fast_eval: bool | None = None,
    targets_df: pd.DataFrame | None = None,
    catalog: FeatureCatalog | None = None,
    mmc_frame: tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray] | None = None,
    mmc_frame_preloaded: bool = False,
) -> dict[str, float]:
    """Train a single HPO trial and return backtest metrics."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        np.random.seed(rng.integers(0, 2**31))

    use_fast = fast_eval if fast_eval is not None else bool(config.get("hpo_fast"))
    purge_eras = int(config.get("purge_eras", WF_N_PURGE))
    if purge_eras < 0:
        raise ValueError("purge_eras must be >= 0")

    pipeline_cfg, feature_groups, routed_columns = _resolve_pipeline_cfg(
        config, catalog
    )
    if routed_columns:
        feature_cols = [c for c in routed_columns if c in X_train.columns]

    if config.get("use_gpu"):
        from .search_space import apply_gpu_pipeline_config

        pipeline_cfg = apply_gpu_pipeline_config(pipeline_cfg)
    train_kwargs = get_train_kwargs_from_flat(config)

    strategy = strategy_from_flat(config)
    purge_eras = _effective_purge_eras(purge_eras, strategy)
    config["effective_purge_eras"] = purge_eras
    y_eval = (
        targets_df[strategy.primary_target]
        if targets_df is not None and strategy.primary_target in targets_df.columns
        else y_train
    )

    if use_fast:
        return _evaluate_holdout(
            X_train=X_train,
            y_train=y_eval,
            era_train=era_train,
            feature_cols=feature_cols,
            pipeline_cfg=pipeline_cfg,
            train_kwargs=train_kwargs,
            purge_eras=purge_eras,
            config=config,
            seed=seed,
            data_seed=data_seed,
            feature_groups=feature_groups,
            targets_df=targets_df,
            mmc_frame=mmc_frame,
            mmc_frame_preloaded=mmc_frame_preloaded,
        )

    def train_fn(X_tr: pd.DataFrame, y_tr: pd.Series) -> Any:
        local_targets = None
        if targets_df is not None:
            local_targets = targets_df.loc[X_tr.index]
        return _fit_pipeline(
            pipeline_cfg,
            feature_cols,
            X_tr,
            y_tr,
            train_kwargs,
            flat_config=config,
            seed=seed,
            feature_groups=feature_groups,
            targets_df=local_targets,
        )

    diagnostics_state: dict[str, Any] = {}

    def last_fold_callback(
        predictor: Any, X_te: pd.DataFrame, y_te: pd.Series, era_te: pd.Series
    ) -> None:
        diagnostics_state["pipeline"] = predictor
        diagnostics_state["X_val"] = X_te
        diagnostics_state["y_val"] = y_te
        diagnostics_state["era_val"] = era_te

    metrics = EraSplitEvaluator(
        feature_columns=feature_cols,
        n_splits=WF_N_SPLITS,
        n_purge=purge_eras,
        min_train_eras=WF_MIN_TRAIN_ERAS,
    ).evaluate_walk_forward(
        X_train,
        y_eval,
        era_train,
        train_fn,
        last_fold_callback=last_fold_callback
        if config.get("log_wandb_diagnostics")
        else None,
    )
    if config.get("log_wandb_diagnostics") and diagnostics_state:
        from ..evaluation.wandb_diagnostics import log_experiment_diagnostics

        log_experiment_diagnostics(
            pipeline=diagnostics_state["pipeline"],
            X_val=diagnostics_state["X_val"],
            y_val=diagnostics_state["y_val"],
            era_val=diagnostics_state["era_val"],
            feature_cols=feature_cols,
            metrics=metrics,
            log_shap=bool(config.get("wandb_log_shap", True)),
            compute_fnc=False,
        )
    gc.collect()
    return metrics


def run_trial_fast_walk_forward(
    config: dict[str, Any],
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    era_train: pd.Series,
    feature_cols: list[str],
    seed: int | None = None,
    targets_df: pd.DataFrame | None = None,
    catalog: FeatureCatalog | None = None,
) -> dict[str, float]:
    """Walk-forward with reduced folds and capped train window for HPO."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        np.random.seed(rng.integers(0, 2**31))

    pipeline_cfg, feature_groups, routed_columns = _resolve_pipeline_cfg(
        config, catalog
    )
    if routed_columns:
        feature_cols = [c for c in routed_columns if c in X_train.columns]
    if config.get("use_gpu"):
        from .search_space import apply_gpu_pipeline_config

        pipeline_cfg = apply_gpu_pipeline_config(pipeline_cfg)
    train_kwargs = get_train_kwargs_from_flat(config)
    purge_eras = int(config.get("purge_eras", WF_N_PURGE))
    if purge_eras < 0:
        raise ValueError("purge_eras must be >= 0")

    strategy = strategy_from_flat(config)
    purge_eras = _effective_purge_eras(purge_eras, strategy)
    config["effective_purge_eras"] = purge_eras
    y_eval = (
        targets_df[strategy.primary_target]
        if targets_df is not None and strategy.primary_target in targets_df.columns
        else y_train
    )

    def train_fn(X_tr: pd.DataFrame, y_tr: pd.Series) -> Any:
        local_targets = None
        if targets_df is not None:
            local_targets = targets_df.loc[X_tr.index]
        return _fit_pipeline(
            pipeline_cfg,
            feature_cols,
            X_tr,
            y_tr,
            train_kwargs,
            flat_config=config,
            seed=seed,
            feature_groups=feature_groups,
            targets_df=local_targets,
        )

    metrics = EraSplitEvaluator(
        feature_columns=feature_cols,
        n_splits=HPO_FAST_WF_N_SPLITS,
        n_purge=purge_eras,
        min_train_eras=WF_MIN_TRAIN_ERAS,
        max_train_eras=HPO_FAST_MAX_TRAIN_ERAS,
    ).evaluate_walk_forward(X_train, y_eval, era_train, train_fn)
    gc.collect()
    return metrics
