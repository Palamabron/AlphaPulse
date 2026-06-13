"""HPO pipeline: search over preprocessing, models, and ensembles.

Supports two modes:
  --local   Random search (no extra dependencies).
  (default) Ray Tune distributed search (requires ``pip install 'alphapulse[hpo]'``).

Each trial is scored via walk-forward backtesting (3 folds, n_purge=4) so
that the objective reflects temporal out-of-sample performance rather than a
fixed holdout split. The default objective is corr_sharpe from walk-forward.

Pass --wandb-project <name> to log every trial to Weights & Biases.
Pass --resume to skip already-completed trials recorded in the trial database.
"""

import json
import multiprocessing
import time
import uuid
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

from alphapulse.experiments.data import load_train_only_frame
from alphapulse.hpo.objective import TrialResult, run_trial
from alphapulse.hpo.search_space import sample_random_config
from alphapulse.hpo.trial_db import TrialDB
from alphapulse.logging_.leaderboard import (
    entry_from_hpo_result,
    print_leaderboard,
    save_leaderboard,
)
from alphapulse.utils import set_global_seed

_MP_CTX = multiprocessing.get_context("spawn")


def _trial_worker(
    flat_config: dict,
    data_dir: str,
    train_subsample: float,
    target_col: str,
    seed: int,
    result_queue: "multiprocessing.Queue[dict]",
) -> None:
    """Run a single trial inside a subprocess and push the result to the queue."""
    try:
        X_train, y_train, feature_cols = load_train_only_frame(
            Path(data_dir),
            train_subsample=train_subsample,
            target_col=target_col,
            seed=seed,
            feature_columns=None,
            need_era=True,
        )
        era_train = X_train["era"]
        metrics = run_trial(
            flat_config,
            X_train=X_train,
            y_train=y_train,
            era_train=era_train,
            feature_cols=feature_cols,
            seed=seed,
        )
        result_queue.put({"ok": True, "metrics": metrics})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})


def _best_from_db(db: TrialDB, objective: str) -> tuple[float, dict]:
    best_score = float("-inf")
    best_config: dict = {}
    for row in db.load_all_trials():
        if row["status"] != "completed" or not row["metrics"]:
            continue
        metrics = row["metrics"]
        score = float(metrics.get(objective, metrics.get("corr_sharpe", float("-inf"))))
        if score > best_score:
            best_score = score
            best_config = row["flat_config"]
    return best_score, best_config


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
    resume: bool = False,
    trial_timeout: int = 1800,
    gpu: bool = False,
) -> None:
    """Local random-search HPO with subprocess isolation and SQLite trial DB."""

    X_train, y_train, feature_cols = load_train_only_frame(
        data_dir,
        train_subsample=train_subsample,
        target_col=target_col,
        seed=seed,
        feature_columns=None,
        need_era=True,
    )
    logger.info(
        "Data loaded: train={}, features={}",
        X_train.shape,
        len(feature_cols),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "trials.db"
    wandb_group = f"hpo-{uuid.uuid4().hex[:8]}" if wandb_project else None
    if wandb_project:
        logger.info(
            "WandB logging enabled: project={} group={}", wandb_project, wandb_group
        )

    trial_kwargs = {
        "data_dir": str(data_dir),
        "train_subsample": train_subsample,
        "target_col": target_col,
    }

    results: list[TrialResult] = []
    best_score = float("-inf")
    best_config: dict = {}

    with TrialDB(db_path) as db:
        already_done = db.completed_trials() if resume else set()
        if resume:
            best_score, best_config = _best_from_db(db, objective)
            if best_config:
                logger.info(
                    "Resuming: global best {}={:.4f} from {} completed trial(s)",
                    objective,
                    best_score,
                    len(already_done),
                )
        if resume and already_done:
            logger.info(
                "Resuming: {} trial(s) already completed, skipping.", len(already_done)
            )

        for i in range(num_trials):
            if i in already_done:
                logger.info(
                    "Trial {}/{}: skipped (already completed)", i + 1, num_trials
                )
                continue

            flat_config = sample_random_config(seed=seed + i)
            if gpu:
                flat_config["use_gpu"] = True
            db.insert_trial(i, flat_config)

            t0 = time.perf_counter()
            result_queue: multiprocessing.Queue = _MP_CTX.Queue()
            p = _MP_CTX.Process(
                target=_trial_worker,
                args=(
                    flat_config,
                    trial_kwargs["data_dir"],
                    trial_kwargs["train_subsample"],
                    trial_kwargs["target_col"],
                    seed + i,
                    result_queue,
                ),
            )
            p.start()
            p.join(timeout=trial_timeout)
            elapsed = time.perf_counter() - t0

            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
                error_msg = f"timeout after {trial_timeout}s"
                result = TrialResult(
                    trial_number=i,
                    sharpe=float("-inf"),
                    metrics={},
                    model_type=flat_config.get("model_1_type", "XGBoost"),
                    elapsed_seconds=elapsed,
                    params=flat_config,
                    error=error_msg,
                )
                db.update_trial(
                    i, status="failed", error=error_msg, elapsed_seconds=elapsed
                )
                trial_score = float("-inf")
            elif p.exitcode != 0:
                error_msg = f"subprocess exit code {p.exitcode}"
                result = TrialResult(
                    trial_number=i,
                    sharpe=float("-inf"),
                    metrics={},
                    model_type=flat_config.get("model_1_type", "XGBoost"),
                    elapsed_seconds=elapsed,
                    params=flat_config,
                    error=error_msg,
                )
                db.update_trial(
                    i, status="failed", error=error_msg, elapsed_seconds=elapsed
                )
                trial_score = float("-inf")
            else:
                payload = (
                    result_queue.get_nowait()
                    if not result_queue.empty()
                    else {"ok": False, "error": "no result"}
                )
                if payload.get("ok"):
                    metrics = payload["metrics"]
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
                    db.update_trial(
                        i,
                        status="completed",
                        metrics=metrics,
                        elapsed_seconds=elapsed,
                    )
                else:
                    error_msg = payload.get("error", "unknown")
                    trial_score = float("-inf")
                    result = TrialResult(
                        trial_number=i,
                        sharpe=float("-inf"),
                        metrics={},
                        model_type=flat_config.get("model_1_type", "XGBoost"),
                        elapsed_seconds=elapsed,
                        params=flat_config,
                        error=error_msg,
                    )
                    db.update_trial(
                        i, status="failed", error=error_msg, elapsed_seconds=elapsed
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

        best_score, best_config = _best_from_db(db, objective)

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
    resume: bool = False,
    objective: Literal[
        "corr_sharpe", "mean_per_era_correlation", "max_drawdown"
    ] = "corr_sharpe",
    wandb_project: str | None = None,
    trial_timeout: int = 1800,
    gpu: bool = False,
) -> None:
    """Run HPO search over preprocessing, models, and ensemble strategies.

    Use --local for random search without Ray, or omit for Ray Tune.
    Use --objective to choose the optimization target (default: corr_sharpe).
    Pass --wandb-project <name> to log every trial to Weights & Biases.
    Pass --resume to continue an interrupted sweep (requires --local).
    Pass --trial-timeout N to cap each subprocess trial at N seconds (default: 1800).
    Pass --gpu to enable CUDA for XGBoost and GPU task type for CatBoost.
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
            resume=resume,
            trial_timeout=trial_timeout,
            gpu=gpu,
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
