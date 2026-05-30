from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..hpo.objective import TrialResult


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

    wandb.finish(quiet=True)


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
        "scaler_type",
        "use_packboost",
        "num_models",
        "n_subs",
        "ensemble_method",
        "use_neutralization",
        "neutralization_proportion",
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
            r.params.get("scaler_type"),
            r.params.get("use_packboost"),
            r.params.get("num_models", 1),
            r.params.get("n_subs"),
            r.params.get("ensemble_method"),
            r.params.get("use_neutralization"),
            r.params.get("neutralization_proportion"),
            r.elapsed_seconds,
        )

    wandb.init(
        project=project,
        group=group,
        name="hpo-summary",
        job_type="summary",
        reinit=True,
    )
    wandb.log({"trials_summary": table})
    wandb.finish(quiet=True)


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

    logged: dict[str, Any] = {
        "sharpe": result.sharpe,
        "objective": objective,
        "elapsed_seconds": result.elapsed_seconds,
        "model_types": model_types,
        "preprocessors": preprocessors,
    }
    if result.corr_sharpe not in (float("-inf"), float("inf")):
        logged["corr_sharpe"] = result.corr_sharpe
    if result.mmc_sharpe is not None:
        logged["mmc_sharpe"] = result.mmc_sharpe
    if result.payout_score is not None:
        logged["payout_score"] = result.payout_score
    for k, v in (result.metrics or {}).items():
        logged[f"metric/{k}"] = v

    wandb.log(logged)
    wandb.finish(quiet=True)
