import random
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

    Returns:
        Dictionary of backtest metrics (keys include ``sharpe``,
        ``mean_per_era_correlation``, ``correlation``, etc.).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

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
    return bt.evaluate(X_val, y_val, era_val)
