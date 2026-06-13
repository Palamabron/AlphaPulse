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


def _fit_pipeline(
    pipeline_cfg: dict[str, Any],
    feature_cols: list[str],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    train_kwargs: dict[str, Any],
) -> Any:
    pipeline = build_pipeline_or_multi(
        pipeline_cfg, feature_columns=feature_cols, feature_groups=None
    )
    era_col = X_tr["era"] if "era" in X_tr.columns else None
    X_fit, y_fit, X_val_inner, y_val_inner = internal_val_split(
        X_tr, y_tr, era_train=era_col
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
    )
    ho_mask = era_train.isin(holdout_set)
    return Backtester(pipeline, feature_columns=feature_cols).evaluate(
        X_train.loc[ho_mask],
        y_train.loc[ho_mask],
        era_train.loc[ho_mask],
    )


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
        )

    def train_fn(X_tr: pd.DataFrame, y_tr: pd.Series) -> Any:
        return _fit_pipeline(pipeline_cfg, feature_cols, X_tr, y_tr, train_kwargs)

    return EraSplitEvaluator(
        feature_columns=feature_cols,
        n_splits=WF_N_SPLITS,
        n_purge=WF_N_PURGE,
        min_train_eras=WF_MIN_TRAIN_ERAS,
    ).evaluate_walk_forward(X_train, y_train, era_train, train_fn)


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
        return _fit_pipeline(pipeline_cfg, feature_cols, X_tr, y_tr, train_kwargs)

    return EraSplitEvaluator(
        feature_columns=feature_cols,
        n_splits=HPO_FAST_WF_N_SPLITS,
        n_purge=WF_N_PURGE,
        min_train_eras=WF_MIN_TRAIN_ERAS,
        max_train_eras=HPO_FAST_MAX_TRAIN_ERAS,
    ).evaluate_walk_forward(X_train, y_train, era_train, train_fn)
