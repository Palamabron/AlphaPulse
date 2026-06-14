from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.era_split import (
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


def run_trial(
    config: dict[str, Any],
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    era_train: pd.Series,
    feature_cols: list[str],
    seed: int | None = None,
) -> dict[str, float]:
    """Train a single HPO trial and return walk-forward backtest metrics.

    Each trial retrains the model on expanding era windows (n_splits=3) so
    that the returned metrics reflect out-of-sample temporal performance rather
    than a fixed holdout split.

    Args:
        config: Flat parameter dictionary (as produced by
            ``sample_random_config`` or Ray Tune).
        X_train: Training feature DataFrame (may include an "era" column).
        y_train: Training target Series.
        era_train: Era labels aligned to X_train.
        feature_cols: Feature column names (must not include "era").
        seed: Optional integer seed for reproducibility. Pass a per-trial
            value (e.g. trial number) rather than a fixed constant so that
            parallel Ray workers do not share the same RNG state.

    Returns:
        Dictionary of walk-forward metrics (keys include ``corr_sharpe``,
        ``mean_per_era_correlation``, ``max_drawdown``, ``pct_positive_eras``,
        ``n_valid_eras``).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        np.random.seed(rng.integers(0, 2**31))

    pipeline_cfg = resolve_flat_config(config)
    train_kwargs = get_train_kwargs_from_flat(config)

    def train_fn(X_tr: pd.DataFrame, y_tr: pd.Series) -> Any:
        pipeline = build_pipeline_or_multi(
            pipeline_cfg, feature_columns=feature_cols, feature_groups=None
        )
        era_col = X_tr["era"] if "era" in X_tr.columns else None
        stacking_needs_val = (
            pipeline_cfg.get("ensemble_method") == "stacking"
            and len(pipeline_cfg.get("models", [])) > 1
        )
        X_fit, y_fit, X_val_inner, y_val_inner = internal_val_split(
            X_tr, y_tr, era_train=era_col, force_internal=stacking_needs_val
        )
        pipeline.fit(X_fit, y_fit, X_val=X_val_inner, y_val=y_val_inner, **train_kwargs)
        return pipeline

    return EraSplitEvaluator(
        feature_columns=feature_cols,
        n_splits=WF_N_SPLITS,
        n_purge=WF_N_PURGE,
        min_train_eras=WF_MIN_TRAIN_ERAS,
    ).evaluate_walk_forward(X_train, y_train, era_train, train_fn)
