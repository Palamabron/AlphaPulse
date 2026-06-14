from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

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
from ..models.diffusion_augmenter import SyntheticDataAugmenter
from .builder import build_pipeline_or_multi
from .search_space import get_train_kwargs_from_flat, resolve_flat_config


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


def ray_trainable(config: dict[str, Any], **kwargs: Any) -> dict[str, float]:
    return run_trial(config, **kwargs)


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


def _fit_pipeline(
    pipeline_cfg: dict[str, Any],
    feature_cols: list[str],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    train_kwargs: dict[str, Any],
    flat_config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> Any:
    X_fit_src = X_tr
    y_fit_src = y_tr
    if flat_config and flat_config.get("use_augmentation"):
        X_fit_src, y_fit_src = _apply_synthetic_augmentation(
            X_tr, y_tr, flat_config, feature_cols, seed
        )
    pipeline = build_pipeline_or_multi(
        pipeline_cfg, feature_columns=feature_cols, feature_groups=None
    )
    era_col = X_fit_src["era"] if "era" in X_fit_src.columns else None
    stacking_needs_val = (
        pipeline_cfg.get("ensemble_method") == "stacking"
        and len(pipeline_cfg.get("models", [])) > 1
    )
    X_fit, y_fit, X_val_inner, y_val_inner = internal_val_split(
        X_fit_src, y_fit_src, era_train=era_col, force_internal=stacking_needs_val
    )
    pipeline.fit(X_fit, y_fit, X_val=X_val_inner, y_val=y_val_inner, **train_kwargs)
    return pipeline


def _evaluate_holdout(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    era_train: pd.Series,
    feature_cols: list[str],
    pipeline_cfg: dict[str, Any],
    train_kwargs: dict[str, Any],
    holdout_eras: int = HPO_FAST_HOLDOUT_ERAS,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    eras_sorted = sorted(era_train.unique(), key=str)
    min_train = WF_MIN_TRAIN_ERAS
    n_holdout = min(holdout_eras, max(5, len(eras_sorted) // 5))
    if len(eras_sorted) <= min_train + n_holdout:
        n_holdout = max(1, len(eras_sorted) // 5)

    holdout_set = set(eras_sorted[-n_holdout:])
    train_mask = ~era_train.isin(holdout_set)
    if not train_mask.any():
        return {"corr_sharpe": float("nan"), "mean_per_era_correlation": float("nan")}

    pipeline = _fit_pipeline(
        pipeline_cfg,
        feature_cols,
        X_train.loc[train_mask],
        y_train.loc[train_mask],
        train_kwargs,
        flat_config=config,
        seed=seed,
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
    if config and config.get("log_wandb_diagnostics"):
        from ..evaluation.wandb_diagnostics import log_experiment_diagnostics

        log_experiment_diagnostics(
            pipeline=pipeline,
            X_val=X_ho,
            y_val=y_ho,
            era_val=era_ho,
            feature_cols=feature_cols,
            metrics=metrics,
            log_shap=bool(config.get("wandb_log_shap", True)),
            compute_fnc=False,
        )
    return metrics


def run_trial(
    config: dict[str, Any],
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    era_train: pd.Series,
    feature_cols: list[str],
    seed: int | None = None,
    fast_eval: bool | None = None,
) -> dict[str, float]:
    """Train a single HPO trial and return backtest metrics.

    When ``fast_eval`` is True (default for ``hpo_fast`` configs), scores a
    single holdout on the last ``HPO_FAST_HOLDOUT_ERAS`` eras instead of full
    walk-forward CV. This keeps trials under the 30-minute budget on full data.

    Args:
        config: Flat parameter dictionary (as produced by
            ``sample_random_config`` or Ray Tune).
        X_train: Training feature DataFrame (may include an "era" column).
        y_train: Training target Series.
        era_train: Era labels aligned to X_train.
        feature_cols: Feature column names (must not include "era").
        seed: Optional integer seed for reproducibility.
        fast_eval: Override fast holdout mode. When None, reads ``hpo_fast``
            from *config* (defaults to False).

    Returns:
        Dictionary of backtest metrics (keys include ``corr_sharpe``,
        ``mean_per_era_correlation``, ``max_drawdown``, ``pct_positive_eras``,
        ``n_valid_eras``).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        np.random.seed(rng.integers(0, 2**31))

    use_fast = fast_eval if fast_eval is not None else bool(config.get("hpo_fast"))

    pipeline_cfg = resolve_flat_config(config)
    if config.get("use_gpu"):
        from .search_space import apply_gpu_pipeline_config

        pipeline_cfg = apply_gpu_pipeline_config(pipeline_cfg)
    train_kwargs = get_train_kwargs_from_flat(config)

    if use_fast:
        return _evaluate_holdout(
            X_train=X_train,
            y_train=y_train,
            era_train=era_train,
            feature_cols=feature_cols,
            pipeline_cfg=pipeline_cfg,
            train_kwargs=train_kwargs,
            config=config,
            seed=seed,
        )

    def train_fn(X_tr: pd.DataFrame, y_tr: pd.Series) -> Any:
        return _fit_pipeline(
            pipeline_cfg,
            feature_cols,
            X_tr,
            y_tr,
            train_kwargs,
            flat_config=config,
            seed=seed,
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
        n_purge=WF_N_PURGE,
        min_train_eras=WF_MIN_TRAIN_ERAS,
    ).evaluate_walk_forward(
        X_train,
        y_train,
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
    return metrics


def run_trial_fast_walk_forward(
    config: dict[str, Any],
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    era_train: pd.Series,
    feature_cols: list[str],
    seed: int | None = None,
) -> dict[str, float]:
    """Walk-forward with reduced folds and capped train window for HPO."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        np.random.seed(rng.integers(0, 2**31))

    pipeline_cfg = resolve_flat_config(config)
    if config.get("use_gpu"):
        from .search_space import apply_gpu_pipeline_config

        pipeline_cfg = apply_gpu_pipeline_config(pipeline_cfg)
    train_kwargs = get_train_kwargs_from_flat(config)

    def train_fn(X_tr: pd.DataFrame, y_tr: pd.Series) -> Any:
        return _fit_pipeline(
            pipeline_cfg,
            feature_cols,
            X_tr,
            y_tr,
            train_kwargs,
            flat_config=config,
            seed=seed,
        )

    return EraSplitEvaluator(
        feature_columns=feature_cols,
        n_splits=HPO_FAST_WF_N_SPLITS,
        n_purge=WF_N_PURGE,
        min_train_eras=WF_MIN_TRAIN_ERAS,
        max_train_eras=HPO_FAST_MAX_TRAIN_ERAS,
    ).evaluate_walk_forward(X_train, y_train, era_train, train_fn)
