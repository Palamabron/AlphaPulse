from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from ..hpo.objective import TrialResult


def resolve_wandb_project(
    base: str,
    *,
    output_dir: "Path | None" = None,
    stamp_file: str = "wandb_project.txt",
) -> str:
    """Return a timestamped W&B project name, persisted for resume in *output_dir*."""
    from datetime import datetime
    from pathlib import Path

    if output_dir is not None:
        path = Path(output_dir) / stamp_file
        if path.exists():
            return path.read_text(encoding="utf-8").strip()

    stamped = f"{base}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path.write_text(stamped, encoding="utf-8")
    return stamped


def init_wandb(
    project: str,
    config: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> None:
    import wandb

    wandb.init(project=project, name=name, config=config or {}, **kwargs)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    import wandb

    wandb.log(metrics, step=step)


def log_backtest_results(metrics: dict[str, float], prefix: str = "backtest") -> None:
    import wandb

    prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()}
    wandb.log(prefixed)


def init_wandb_run(
    project: str, name: str | None = None, config: dict[str, Any] | None = None
) -> None:
    """Start a long-lived WandB run (e.g. for the full AutoResearch loop)."""
    import wandb

    wandb.init(
        project=project, name=name, config=config or {}, reinit="finish_previous"
    )


def finish_wandb_run() -> None:
    import wandb

    from .wandb_logging import detach_wandb_loguru

    detach_wandb_loguru()
    wandb.finish()


def log_research_step(
    trial_number: int,
    metrics: dict[str, float],
    model_types: list[str],
    action_taken: str,
    elapsed_seconds: float,
    sharpe: float,
    mmc_sharpe: float | None = None,
    payout_score: float | None = None,
) -> None:
    """Log a single AutoResearch trial step to the active WandB run.

    All trials are steps in one run so that sharpe-over-time and
    mutation-type trajectory charts are generated automatically.
    """
    import wandb

    logged: dict[str, Any] = {
        "sharpe": sharpe,
        "elapsed_seconds": elapsed_seconds,
        "model_types": "+".join(model_types),
        "action_taken": action_taken,
    }
    if mmc_sharpe is not None:
        logged["mmc_sharpe"] = mmc_sharpe
    if payout_score is not None:
        logged["payout_score"] = payout_score
    for k, v in metrics.items():
        logged[f"metric/{k}"] = v

    wandb.log(logged, step=trial_number)


def log_hpo_summary_table(
    results: list[Any],
    project: str,
    group: str,
) -> None:
    """Log all successful HPO trials as a WandB Table.

    Creates a 'hpo-summary' run in the same group, uploading a Table artifact
    that enables custom scatter, violin, and grouped bar charts inside WandB.
    """
    import wandb

    columns = [
        "trial",
        "HoldoutSharpe",
        "ValidationSharpe",
        "ValidationMmcSharpe",
        "LegacyPayoutProxy",
        "HoldoutMeanCorr",
        "ValidationMeanCorr",
        "HoldoutStdCorr",
        "HoldoutMaxDrawdown",
        "HoldoutPctPositiveEras",
        "model_types",
        "model_1_type",
        "model_2_type",
        "model_3_type",
        "scaler_type",
        "use_packboost",
        "num_models",
        "n_subs",
        "ensemble_method",
        "use_neutralization",
        "neutralization_proportion",
        "xgb_max_depth",
        "xgb_learning_rate",
        "lgbm_num_leaves",
        "lgbm_learning_rate",
        "lgbm_min_child_samples",
        "use_noise_injection",
        "feature_selection_type",
        "use_feature_selection",
        "use_augmentation",
        "active_groups",
        "active_groups_count",
        "routed_feature_count",
        "elapsed_seconds",
    ]
    table = wandb.Table(columns=columns)
    for r in results:
        if r.error:
            continue
        num = r.params.get("num_models", 1)
        model_types = "+".join(
            str(r.params.get(f"model_{i}_type", "?")) for i in range(1, num + 1)
        )
        table.add_data(
            r.trial_number,
            r.corr_sharpe
            if r.corr_sharpe not in (float("-inf"), float("inf"))
            else None,
            r.metrics.get("val_corr_sharpe"),
            r.mmc_sharpe,
            r.payout_score,
            r.metrics.get("holdout_mean_per_era_correlation")
            if r.metrics.get("holdout_mean_per_era_correlation") is not None
            else r.metrics.get("mean_per_era_correlation"),
            r.metrics.get("val_mean_per_era_correlation"),
            r.metrics.get("holdout_std_per_era_correlation")
            if r.metrics.get("holdout_std_per_era_correlation") is not None
            else r.metrics.get("std_per_era_correlation"),
            r.metrics.get("holdout_max_drawdown")
            if r.metrics.get("holdout_max_drawdown") is not None
            else r.metrics.get("max_drawdown"),
            r.metrics.get("holdout_pct_positive_eras")
            if r.metrics.get("holdout_pct_positive_eras") is not None
            else r.metrics.get("pct_positive_eras"),
            model_types,
            r.params.get("model_1_type"),
            r.params.get("model_2_type"),
            r.params.get("model_3_type"),
            r.params.get("scaler_type"),
            r.params.get("use_packboost"),
            r.params.get("num_models", 1),
            r.params.get("n_subs"),
            r.params.get("ensemble_method"),
            r.params.get("use_neutralization"),
            r.params.get("neutralization_proportion"),
            r.params.get("xgb_max_depth"),
            r.params.get("xgb_learning_rate"),
            r.params.get("lgbm_num_leaves"),
            r.params.get("lgbm_learning_rate"),
            r.params.get("lgbm_min_child_samples"),
            r.params.get("use_noise_injection"),
            r.params.get("feature_selection_type"),
            r.params.get("use_feature_selection"),
            r.params.get("use_augmentation"),
            "+".join(r.params.get("active_groups", [])),
            r.params.get("active_groups_count"),
            r.params.get("routed_feature_count"),
            r.elapsed_seconds,
        )

    wandb.init(
        project=project,
        group=group,
        name="hpo-summary",
        job_type="summary",
        reinit="finish_previous",
    )
    summary_charts: dict[str, Any] = {}
    if any(r.error is None for r in results):
        summary_charts["hpo/trial_LegacyPayoutProxy"] = wandb.plot.scatter(
            table,
            "trial",
            "LegacyPayoutProxy",
            title="Trial validation legacy payout proxy",
        )
        summary_charts["hpo/trial_HoldoutSharpe"] = wandb.plot.scatter(
            table,
            "trial",
            "HoldoutSharpe",
            title="Trial holdout HoldoutSharpe",
        )
        summary_charts["hpo/trial_ValidationSharpe"] = wandb.plot.scatter(
            table,
            "trial",
            "ValidationSharpe",
            title="Trial validation ValidationSharpe",
        )
        summary_charts["hpo/trial_ValidationMmcSharpe"] = wandb.plot.scatter(
            table,
            "trial",
            "ValidationMmcSharpe",
            title="Trial validation ValidationMmcSharpe",
        )
        summary_charts["hpo/trial_elapsed"] = wandb.plot.scatter(
            table,
            "trial",
            "elapsed_seconds",
            title="Trial runtime (seconds)",
        )
    wandb.log({"trials_summary": table, **summary_charts})
    wandb.finish()


def _finite_metric(value: Any) -> float | None:
    import numpy as np

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _log_split_metrics(
    logged: dict[str, Any],
    metrics: dict[str, Any],
    result: "TrialResult",
) -> None:
    holdout_corr = _finite_metric(metrics.get("holdout_corr_sharpe"))
    if holdout_corr is None:
        holdout_corr = _finite_metric(result.corr_sharpe)
    if holdout_corr is not None:
        logged["holdout/HoldoutSharpe"] = holdout_corr
    holdout_mean = _finite_metric(metrics.get("holdout_mean_per_era_correlation"))
    if holdout_mean is None:
        holdout_mean = _finite_metric(metrics.get("mean_per_era_correlation"))
    if holdout_mean is not None:
        logged["holdout/HoldoutMeanCorr"] = holdout_mean
    holdout_dd = _finite_metric(metrics.get("holdout_max_drawdown"))
    if holdout_dd is None:
        holdout_dd = _finite_metric(metrics.get("max_drawdown"))
    if holdout_dd is not None:
        logged["holdout/HoldoutMaxDrawdown"] = holdout_dd

    val_corr = _finite_metric(metrics.get("val_corr_sharpe"))
    if val_corr is not None:
        logged["validation/ValidationSharpe"] = val_corr
    val_mean = _finite_metric(metrics.get("val_mean_per_era_correlation"))
    if val_mean is not None:
        logged["validation/ValidationMeanCorr"] = val_mean
    mmc = _finite_metric(result.mmc_sharpe)
    if mmc is None:
        mmc = _finite_metric(metrics.get("mmc_sharpe"))
    if mmc is not None:
        logged["validation/ValidationMmcSharpe"] = mmc
    payout = _finite_metric(result.payout_score)
    if payout is None:
        payout = _finite_metric(metrics.get("payout_score"))
    if payout is not None:
        logged["validation/LegacyPayoutProxy"] = payout
    mmc_mean = _finite_metric(metrics.get("mmc"))
    if mmc_mean is not None:
        logged["validation/ValidationMmc"] = mmc_mean


def log_hpo_trial_metrics(
    result: "TrialResult",
    objective: float,
    *,
    model_types: str | None = None,
    preprocessors: str | None = None,
) -> None:
    import numpy as np
    import wandb

    logged: dict[str, Any] = {
        "objective": objective,
        "elapsed_seconds": result.elapsed_seconds,
    }
    if model_types is not None:
        logged["model_types"] = model_types
    if preprocessors is not None:
        logged["preprocessors"] = preprocessors
    active_groups = result.params.get("active_groups", [])
    if isinstance(active_groups, list):
        logged["active_groups"] = "+".join(str(g) for g in active_groups)
        logged["active_groups_count"] = len(active_groups)
    routed_feature_count = result.params.get("routed_feature_count")
    if routed_feature_count is not None:
        logged["routed_feature_count"] = routed_feature_count
    metrics = result.metrics or {}
    _log_split_metrics(logged, metrics, result)
    skip_metric_keys = {
        "corr_sharpe",
        "mean_per_era_correlation",
        "std_per_era_correlation",
        "max_drawdown",
        "pct_positive_eras",
        "n_valid_eras",
        "mmc",
        "mmc_sharpe",
        "payout_score",
    }
    for key, value in metrics.items():
        if key in skip_metric_keys or key.startswith(("holdout_", "val_")):
            continue
        if value is None:
            continue
        if isinstance(value, float) and not np.isfinite(value):
            continue
        logged[f"metric/{key}"] = value
    wandb.log(logged)


def log_importance_artifact(
    importance: dict[str, float],
    *,
    name: str = "feature-importance",
) -> None:
    """Log a feature importance dict as a WandB CSV Artifact on the active run.

    Args:
        importance: Mapping of feature name to importance score, sorted descending.
        name: Artifact name (used as the WandB artifact identifier).
    """
    import io

    import wandb

    if wandb.run is None:
        return

    rows = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    buf = io.StringIO()
    buf.write("feature,importance\n")
    for feat, score in rows:
        buf.write(f"{feat},{score}\n")

    artifact = wandb.Artifact(name=name, type="dataset")
    with artifact.new_file("feature_importance.csv", mode="w") as f:
        f.write(buf.getvalue())
    wandb.run.log_artifact(artifact)


def log_hpo_trial(
    result: "TrialResult",
    flat_config: dict[str, Any],
    project: str,
    group: str,
    objective: float,
) -> None:
    """Log a single HPO trial as its own WandB run.

    Each trial gets a separate run within ``group`` so WandB's sweep
    analysis tools (parallel coordinates, parameter importance) work
    correctly across all trials.
    """
    import wandb

    num = flat_config.get("num_models", 1)
    model_types = "+".join(
        str(flat_config.get(f"model_{i}_type", "?")) for i in range(1, num + 1)
    )
    preprocessors = flat_config.get("scaler_type", "StandardScaler")
    if flat_config.get("use_packboost"):
        preprocessors += "+Packboost"

    config_for_wandb = {
        **flat_config,
        "model_types": model_types,
        "preprocessors": preprocessors,
    }

    wandb.init(
        project=project,
        group=group,
        name=f"trial_{result.trial_number:03d}",
        config=config_for_wandb,
        reinit="finish_previous",
    )

    log_hpo_trial_metrics(
        result, objective, model_types=model_types, preprocessors=preprocessors
    )
    wandb.finish()


def log_hpo_convergence(
    results: list[Any],
    *,
    project: str,
    group: str,
    objective: str = "payout_score",
) -> None:
    """Log per-trial holdout/validation metrics and running best in one WandB run."""
    import numpy as np
    import wandb

    from ..hpo.optimization import (
        is_better_optimization_score,
        worst_optimization_score,
    )

    wandb.init(
        project=project,
        group=group,
        name="search-convergence",
        job_type="convergence",
        reinit="finish_previous",
    )
    best_so_far = worst_optimization_score(objective)
    for r in results:
        if r.error:
            continue
        holdout_corr = r.corr_sharpe
        if holdout_corr in (float("-inf"), float("inf")):
            holdout_corr = None
        payout = r.payout_score
        if payout is not None and not np.isfinite(payout):
            payout = None
        mmc = r.mmc_sharpe
        if mmc is not None and not np.isfinite(mmc):
            mmc = None
        val_corr = r.metrics.get("val_corr_sharpe") if r.metrics else None
        if val_corr is not None and not np.isfinite(float(val_corr)):
            val_corr = None

        trial_objective = r.metrics.get(objective) if r.metrics else None
        if trial_objective is not None:
            trial_objective = float(trial_objective)
            if not np.isfinite(trial_objective):
                trial_objective = None
        if trial_objective is not None and is_better_optimization_score(
            trial_objective,
            best_so_far,
            objective,
        ):
            best_so_far = trial_objective

        logged: dict[str, Any] = {}
        if holdout_corr is not None:
            logged["holdout/HoldoutSharpe"] = holdout_corr
        if val_corr is not None:
            logged["validation/ValidationSharpe"] = float(val_corr)
        if mmc is not None:
            logged["validation/ValidationMmcSharpe"] = mmc
        if payout is not None:
            logged["validation/LegacyPayoutProxy"] = payout
        if trial_objective is not None:
            logged[f"best_{objective}_so_far"] = best_so_far
        if logged:
            wandb.log(logged, step=r.trial_number)
    wandb.finish()
