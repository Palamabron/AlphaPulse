from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphapulse.logging_.leaderboard import compute_robust_payout_score

RANKING_METRICS = {
    "holdout": "holdout_corr_sharpe",
    "validation": "payout_score",
    "robust": "robust_payout_score",
}


def discover_latest_trials(root: Path = Path("artifacts")) -> Path | None:
    candidates = list(root.glob("**/all_trials.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_hpo_trials(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("all_trials.json must contain a JSON list")

    rows = [
        row
        for trial in raw
        if isinstance(trial, dict)
        if (row := _normalize_trial(trial)) is not None
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)


def rank_trials(
    trials: pd.DataFrame,
    ranking: str,
    *,
    min_holdout_sharpe: float | None = None,
) -> pd.DataFrame:
    metric = RANKING_METRICS[ranking]
    ranked = trials.dropna(subset=[metric]).copy()
    if min_holdout_sharpe is not None:
        ranked = ranked[ranked["holdout_corr_sharpe"] >= min_holdout_sharpe]
    return ranked.sort_values(
        [metric, "holdout_corr_sharpe", "trial"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def recipe_summary(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()
    return (
        trials.groupby("recipe", dropna=False)
        .agg(
            trials=("trial", "count"),
            median_holdout=("holdout_corr_sharpe", "median"),
            best_holdout=("holdout_corr_sharpe", "max"),
            median_payout=("payout_score", "median"),
        )
        .reset_index()
        .sort_values(["median_holdout", "best_holdout"], ascending=False)
    )


def _normalize_trial(trial: dict[str, Any]) -> dict[str, Any] | None:
    if trial.get("error"):
        return None

    params = trial.get("params")
    metrics = trial.get("metrics")
    if not isinstance(params, dict) or not isinstance(metrics, dict):
        return None

    trial_number = _finite_number(trial.get("trial"))
    holdout = _first_finite(
        metrics.get("holdout_corr_sharpe"),
        metrics.get("corr_sharpe"),
        trial.get("sharpe"),
    )
    if trial_number is None or holdout is None:
        return None

    num_models = _safe_model_count(params.get("num_models", 1))
    model_types = [
        str(params.get(f"model_{index}_type", "?"))
        for index in range(1, num_models + 1)
    ]
    payout = _finite_number(metrics.get("payout_score"))
    val_sharpe = _finite_number(metrics.get("val_corr_sharpe"))
    robust_payout = compute_robust_payout_score(payout, val_sharpe, holdout)

    row: dict[str, Any] = {
        "trial": int(trial_number),
        "holdout_corr_sharpe": holdout,
        "sharpe": holdout,
        "model_types": "+".join(model_types),
        "recipe": (
            f"{'+'.join(model_types)} · {params.get('ensemble_method', 'single')}"
        ),
        "model_1_type": model_types[0],
        "num_models": num_models,
        "scaler_type": params.get("scaler_type", "?"),
        "use_packboost": bool(params.get("use_packboost", False)),
        "use_augmentation": bool(params.get("use_augmentation", False)),
        "n_subs": _finite_number(params.get("n_subs")),
        "ensemble_method": params.get("ensemble_method", "single"),
        "use_neutralization": bool(params.get("use_neutralization", False)),
        "use_meta_neutralization": bool(params.get("use_meta_neutralization", False)),
        "neutralization_proportion": _finite_number(
            params.get("neutralization_proportion")
        ),
        "target_mode": params.get("target_mode", "single"),
        "routed_feature_count": _finite_number(params.get("routed_feature_count")),
        "elapsed_seconds": _finite_number(trial.get("elapsed_seconds")),
        "mean_era_corr": _first_finite(
            metrics.get("holdout_mean_per_era_correlation"),
            metrics.get("mean_per_era_correlation"),
        ),
        "std_era_corr": _first_finite(
            metrics.get("holdout_std_per_era_correlation"),
            metrics.get("std_per_era_correlation"),
        ),
        "max_drawdown": _first_finite(
            metrics.get("holdout_max_drawdown"),
            metrics.get("max_drawdown"),
        ),
        "pct_positive_eras": _first_finite(
            metrics.get("holdout_pct_positive_eras"),
            metrics.get("pct_positive_eras"),
        ),
        "val_corr_sharpe": val_sharpe,
        "val_mean_era_corr": _finite_number(
            metrics.get("val_mean_per_era_correlation")
        ),
        "mmc_sharpe": _finite_number(metrics.get("mmc_sharpe")),
        "payout_score": payout,
        "robust_payout_score": robust_payout,
        "params": params,
    }

    for key in (
        "xgb_max_depth",
        "xgb_learning_rate",
        "xgb_n_rounds",
        "lgbm_num_leaves",
        "lgbm_learning_rate",
        "lgbm_n_rounds",
    ):
        row[key] = _finite_number(params.get(key))
    return row


def _safe_model_count(raw: Any) -> int:
    try:
        return min(3, max(1, int(raw)))
    except (TypeError, ValueError):
        return 1


def _finite_number(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _first_finite(*values: Any) -> float | None:
    for value in values:
        parsed = _finite_number(value)
        if parsed is not None:
            return parsed
    return None
