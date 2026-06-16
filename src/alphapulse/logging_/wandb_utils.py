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

    wandb.init(project=project, name=name, config=config or {}, reinit=True)


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
        "sharpe",
        "corr_sharpe",
        "mmc_sharpe",
        "payout_score",
        "mean_era_corr",
        "std_era_corr",
        "max_drawdown",
        "pct_positive_eras",
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
            r.sharpe,
            r.corr_sharpe
            if r.corr_sharpe not in (float("-inf"), float("inf"))
            else None,
            r.mmc_sharpe,
            r.payout_score,
            r.metrics.get("mean_per_era_correlation"),
            r.metrics.get("std_per_era_correlation"),
            r.metrics.get("max_drawdown"),
            r.metrics.get("pct_positive_eras"),
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
            r.params.get("model_1_max_depth") or r.params.get("xgb_max_depth"),
            r.params.get("model_1_learning_rate") or r.params.get("xgb_learning_rate"),
            r.params.get("model_1_num_leaves") or r.params.get("lgbm_num_leaves"),
            r.params.get("model_1_learning_rate_lgbm")
            or r.params.get("lgbm_learning_rate"),
            r.params.get("model_1_min_child_samples")
            or r.params.get("lgbm_min_child_samples"),
            r.params.get("use_noise_injection"),
            r.params.get("feature_selection_type"),
            r.params.get("use_feature_selection"),
            r.params.get("use_augmentation"),
            r.elapsed_seconds,
        )

    wandb.init(
        project=project,
        group=group,
        name="hpo-summary",
        job_type="summary",
        reinit=True,
    )
    summary_charts: dict[str, Any] = {}
    if any(r.error is None for r in results):
        summary_charts["hpo/trial_corr_sharpe"] = wandb.plot.scatter(
            table,
            "trial",
            "corr_sharpe",
            title="Trial corr Sharpe",
        )
        summary_charts["hpo/trial_mmc_sharpe"] = wandb.plot.scatter(
            table,
            "trial",
            "mmc_sharpe",
            title="Trial MMC Sharpe",
        )
        summary_charts["hpo/trial_elapsed"] = wandb.plot.scatter(
            table,
            "trial",
            "elapsed_seconds",
            title="Trial runtime (seconds)",
        )
    wandb.log({"trials_summary": table, **summary_charts})
    wandb.finish()


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
        "sharpe": result.sharpe,
        "objective": objective,
        "elapsed_seconds": result.elapsed_seconds,
    }
    if model_types is not None:
        logged["model_types"] = model_types
    if preprocessors is not None:
        logged["preprocessors"] = preprocessors
    if result.corr_sharpe not in (float("-inf"), float("inf")):
        logged["corr_sharpe"] = result.corr_sharpe
    mmc_sharpe = result.mmc_sharpe
    if mmc_sharpe is None and result.metrics:
        raw_mmc = result.metrics.get("mmc_sharpe")
        if raw_mmc is not None and np.isfinite(raw_mmc):
            mmc_sharpe = float(raw_mmc)
    payout_score = result.payout_score
    if payout_score is None and result.metrics:
        raw_payout = result.metrics.get("payout_score")
        if raw_payout is not None and np.isfinite(raw_payout):
            payout_score = float(raw_payout)
    if mmc_sharpe is not None:
        logged["mmc_sharpe"] = mmc_sharpe
    if payout_score is not None:
        logged["payout_score"] = payout_score
    top_level_keys = {"sharpe", "corr_sharpe", "mmc_sharpe", "payout_score"}
    for k, v in (result.metrics or {}).items():
        if k not in top_level_keys and v is not None:
            if isinstance(v, float) and not np.isfinite(v):
                continue
            logged[f"metric/{k}"] = v
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
        reinit=True,
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
) -> None:
    """Log per-trial corr_sharpe and running best in a single WandB convergence run.

    All trials are logged as ordered steps within one run so that WandB renders
    a proper convergence curve (trial scores + running maximum line).

    Args:
        results: All TrialResult objects from the HPO search, in trial order.
        project: WandB project name.
        group: WandB group name (same as HPO run group).
    """
    import wandb

    wandb.init(
        project=project,
        group=group,
        name="search-convergence",
        job_type="convergence",
        reinit=True,
    )
    best_so_far = float("-inf")
    for r in results:
        if r.error:
            continue
        trial_corr = (
            r.corr_sharpe
            if r.corr_sharpe not in (float("-inf"), float("inf"))
            else r.sharpe
        )
        if trial_corr > best_so_far:
            best_so_far = trial_corr
        wandb.log(
            {
                "trial_corr_sharpe": trial_corr,
                "best_corr_sharpe_so_far": best_so_far,
            },
            step=r.trial_number,
        )
    wandb.finish()
