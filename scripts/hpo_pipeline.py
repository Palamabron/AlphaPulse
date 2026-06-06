"""HPO pipeline: search over preprocessing, models, and ensembles.

Supports two modes:
  --local   Random search (no extra dependencies).
  (default) Ray Tune distributed search (requires ``pip install 'alphapulse[hpo]'``).

Each trial is scored via walk-forward backtesting (3 folds, n_purge=4) so
that the objective reflects temporal out-of-sample performance rather than a
fixed holdout split. The default objective is corr_sharpe from walk-forward.

Pass --wandb-project <name> to log every trial to Weights & Biases.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

from alphapulse.experiments.data import load_train_only_frame
from alphapulse.hpo.objective import TrialResult, run_trial
from alphapulse.hpo.search_space import sample_random_config
from alphapulse.logging_.leaderboard import (
    entry_from_hpo_result,
    print_leaderboard,
    save_leaderboard,
)
from alphapulse.utils import set_global_seed


def _run_local(
    *,
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    num_trials: int,
    output_dir: Path,
    objective: str = "corr_sharpe",
    wandb_project: str | None = None,
) -> None:
    """Local random-search HPO (no Ray dependency)."""

    X_train, y_train, feature_cols = load_train_only_frame(
        data_dir,
        train_subsample=train_subsample,
        target_col=target_col,
        seed=seed,
        feature_columns=None,
        need_era=True,
    )
    era_train = X_train["era"]
    logger.info(
        "Data loaded: train={}, features={}",
        X_train.shape,
        len(feature_cols),
    )

    wandb_group = f"hpo-{uuid.uuid4().hex[:8]}" if wandb_project else None
    if wandb_project:
        logger.info(
            "WandB logging enabled: project={} group={}", wandb_project, wandb_group
        )

    results: list[TrialResult] = []
    best_score = float("-inf")
    best_config: dict = {}

    for i in range(num_trials):
        flat_config = sample_random_config(seed=seed + i)
        t0 = time.perf_counter()
        try:
            metrics = run_trial(
                flat_config,
                X_train=X_train,
                y_train=y_train,
                era_train=era_train,
                feature_cols=feature_cols,
                seed=seed + i,
            )
            elapsed = time.perf_counter() - t0
            corr_sharpe = metrics.get("corr_sharpe", float("-inf"))
            trial_score = float(metrics.get(objective, corr_sharpe))
            result = TrialResult(
                trial_number=i,
                sharpe=corr_sharpe,
                metrics=metrics,
                model_type=flat_config.get("model_1_type", "XGBoost"),
                elapsed_seconds=elapsed,
                params=flat_config,
                corr_sharpe=corr_sharpe,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.warning("Trial {} failed: {}", i, e)
            trial_score = float("-inf")
            result = TrialResult(
                trial_number=i,
                sharpe=float("-inf"),
                metrics={},
                model_type=flat_config.get("model_1_type", "XGBoost"),
                elapsed_seconds=elapsed,
                params=flat_config,
                error=str(e),
            )

        results.append(result)
        logger.info(
            "Trial {}/{}: corr_sharpe={:.4f} ({:.1f}s){}",
            i + 1,
            num_trials,
            result.sharpe,
            result.elapsed_seconds,
            f" [ERROR: {result.error}]" if result.error else "",
        )
        print_leaderboard(
            logger,
            [entry_from_hpo_result(r) for r in results],
            current_trial=i,
        )

        if wandb_project and wandb_group and not result.error:
            from alphapulse.logging_.wandb_utils import log_hpo_trial

            log_hpo_trial(
                result=result,
                flat_config=flat_config,
                project=wandb_project,
                group=wandb_group,
                objective=trial_score,
            )

        if trial_score > best_score:
            best_score = trial_score
            best_config = flat_config

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_config.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    logger.info("Best {} score: {:.4f}", objective, best_score)
    logger.info("Best config saved to: {}", best_path)

    all_results_path = output_dir / "all_trials.json"
    serializable = [
        {
            "trial": r.trial_number,
            "sharpe": r.sharpe,
            "metrics": r.metrics,
            "model_type": r.model_type,
            "params": r.params,
            "elapsed_seconds": r.elapsed_seconds,
            "error": r.error,
        }
        for r in results
    ]
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    logger.info("All trial results saved to: {}", all_results_path)
    save_leaderboard(
        output_dir / "leaderboard.json",
        [entry_from_hpo_result(r) for r in results],
    )
    logger.info("Leaderboard saved to: {}", output_dir / "leaderboard.json")

    if wandb_project and wandb_group:
        from alphapulse.logging_.wandb_utils import log_hpo_summary_table

        log_hpo_summary_table(results, project=wandb_project, group=wandb_group)
        logger.info("WandB summary table logged to project={}", wandb_project)


def _run_ray(
    *,
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    num_trials: int,
    output_dir: Path,
    objective: str = "corr_sharpe",
    wandb_project: str | None = None,
) -> None:
    """Ray Tune distributed HPO."""
    try:
        import ray
        from ray import tune
        from ray.tune import CLIReporter
    except ImportError as exc:
        logger.error(
            "ray[tune] is required for non-local HPO. "
            "Install with: pip install 'ray[tune]' or use --local."
        )
        raise SystemExit(1) from exc

    from alphapulse.hpo.objective import ray_trainable
    from alphapulse.hpo.search_space import get_full_param_space

    X_train, y_train, feature_cols = load_train_only_frame(
        data_dir,
        train_subsample=train_subsample,
        target_col=target_col,
        seed=seed,
        feature_columns=None,
        need_era=True,
    )
    era_train = X_train["era"]

    ray.init(ignore_reinit_error=True)

    trainable = tune.with_parameters(
        ray_trainable,
        X_train=X_train,
        y_train=y_train,
        era_train=era_train,
        feature_cols=feature_cols,
    )

    param_space = get_full_param_space()

    reporter = CLIReporter(
        metric_columns=["corr_sharpe", "mean_per_era_correlation"],
        max_report_frequency=30,
    )

    callbacks = []
    if wandb_project:
        try:
            from ray.air.integrations.wandb import WandbLoggerCallback

            wandb_group = f"hpo-ray-{uuid.uuid4().hex[:8]}"
            callbacks.append(
                WandbLoggerCallback(
                    project=wandb_project,
                    group=wandb_group,
                    log_config=True,
                )
            )
            logger.info(
                "WandB Ray callback enabled: project={} group={}",
                wandb_project,
                wandb_group,
            )
        except ImportError:
            logger.warning(
                "ray.air.integrations.wandb not available; "
                "skipping WandB logging for Ray mode."
            )

    analysis = tune.run(
        trainable,
        config=param_space,
        num_samples=num_trials,
        metric=objective,
        mode="max",
        progress_reporter=reporter,
        local_dir=str(output_dir / "ray_results"),
        verbose=1,
        callbacks=callbacks,
    )

    best_trial = analysis.best_trial
    best_config = best_trial.config if best_trial else {}

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_config.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, default=str)
    logger.info("Best config saved to: {}", best_path)
    best_score = analysis.best_result.get(objective) if best_trial else "N/A"
    logger.info("Best {}: {}", objective, best_score)

    ray.shutdown()


def main(
    data_dir: Path = Path("data/v5.2"),
    train_subsample: float = 0.125,
    target_col: str = "target",
    seed: int = 42,
    num_trials: int = 30,
    output_dir: Path = Path("artifacts/hpo"),
    local: bool = False,
    objective: Literal[
        "corr_sharpe", "mean_per_era_correlation", "max_drawdown"
    ] = "corr_sharpe",
    wandb_project: str | None = None,
) -> None:
    """Run HPO search over preprocessing, models, and ensemble strategies.

    Use --local for random search without Ray, or omit for Ray Tune.
    Use --objective to choose the optimization target (default: corr_sharpe).
    Pass --wandb-project <name> to log every trial to Weights & Biases.
    """
    set_global_seed(seed)
    if local:
        _run_local(
            data_dir=data_dir,
            train_subsample=train_subsample,
            target_col=target_col,
            seed=seed,
            num_trials=num_trials,
            output_dir=output_dir,
            objective=objective,
            wandb_project=wandb_project,
        )
    else:
        _run_ray(
            data_dir=data_dir,
            train_subsample=train_subsample,
            target_col=target_col,
            seed=seed,
            num_trials=num_trials,
            output_dir=output_dir,
            objective=objective,
            wandb_project=wandb_project,
        )


if __name__ == "__main__":
    tyro.cli(main)
