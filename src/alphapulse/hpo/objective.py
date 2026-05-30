from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation import Backtester
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
    X_val: pd.DataFrame,
    y_val: pd.Series,
    era_val: pd.Series,
    feature_cols: list[str],
    seed: int | None = None,
    meta_model_preds: np.ndarray | None = None,
    corr_weight: float = 0.75,
    mmc_weight: float = 2.25,
) -> dict[str, float]:
    """Train a single HPO trial and return backtest metrics.

    Args:
        config: Flat parameter dictionary (as produced by
            ``sample_random_config`` or Ray Tune).
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_val: Validation feature DataFrame.
        y_val: Validation target Series.
        era_val: Era labels for the validation set.
        feature_cols: Feature column names.
        seed: Optional integer seed for reproducibility. Pass a per-trial
            value (e.g. trial number) rather than a fixed constant so that
            parallel Ray workers do not share the same RNG state.
        meta_model_preds: Optional Numerai meta model predictions for the
            validation set rows. When provided, ``mmc_sharpe`` and
            ``payout_score`` are included in the returned metrics.
        corr_weight: Weight for CORR Sharpe in payout formula. Default 0.75.
        mmc_weight: Weight for MMC Sharpe in payout formula. Default 2.25.

    Returns:
        Dictionary of backtest metrics (keys include ``sharpe``,
        ``mean_per_era_correlation``, ``corr_sharpe``, ``max_drawdown``,
        and optionally ``mmc_sharpe``, ``payout_score``).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        np.random.seed(rng.integers(0, 2**31))

    pipeline_cfg = resolve_flat_config(config)
    pipeline = build_pipeline_or_multi(
        pipeline_cfg, feature_columns=feature_cols, feature_groups=None
    )
    train_kwargs = get_train_kwargs_from_flat(config)

    pipeline.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        **train_kwargs,
    )

    bt = Backtester(pipeline, feature_columns=feature_cols)
    return bt.evaluate(
        X_val,
        y_val,
        era_val,
        meta_model_preds=meta_model_preds,
        corr_weight=corr_weight,
        mmc_weight=mmc_weight,
    )
