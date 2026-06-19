"""HPO pipeline: search over preprocessing, models, and ensembles.

Supports two modes:
  --local   Optuna TPE Bayesian search (default sampler) with subprocess isolation.
  (default) Ray Tune distributed search (requires ``pip install 'alphapulse[hpo]'``).

Each trial is scored via era holdout (fast mode, default) or walk-forward backtesting
(3 folds with --no-fast) so the objective reflects temporal out-of-sample performance.
Fast mode targets sub-30-minute trials on full data.

Pass --wandb-project <name> to log every trial to Weights & Biases.
Pass --resume to skip already-completed trials recorded in the trial database.
"""

import gc
import json
import multiprocessing
import os
import random
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tyro
from dotenv import load_dotenv
from loguru import logger

from alphapulse.experiments.data import load_train_only_frame, load_train_targets_frame
from alphapulse.features.catalog import load_feature_catalog, load_target_catalog
from alphapulse.hpo.feature_routing import resolve_feature_routing
from alphapulse.hpo.objective import TrialResult, run_trial
from alphapulse.hpo.optuna_search import (
    DEFAULT_N_STARTUP_TRIALS,
    SamplerName,
    create_hpo_study,
    suggest_flat_config,
    tell_trial_result,
)
from alphapulse.hpo.target_strategy import (
    apply_target_strategy_to_flat,
    strategy_from_flat,
    validate_target_strategy_early,
)
from alphapulse.hpo.trial_db import TrialDB
from alphapulse.logging_.leaderboard import (
    entry_from_hpo_result,
    print_leaderboard,
    save_leaderboard,
)
from alphapulse.utils import set_global_seed
from alphapulse.utils.gpu_cleanup import (
    cleanup_after_trial_subprocess,
    cleanup_stale_gpu_processes,
    release_cuda_memory,
)

_MP_CTX = multiprocessing.get_context("spawn")
_WANDB_GROUP_FILE = "wandb_group.txt"


def _load_or_create_wandb_group(
    output_dir: Path, wandb_project: str | None
) -> str | None:
    if not wandb_project:
        return None
    path = output_dir / _WANDB_GROUP_FILE
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    group = f"hpo-{uuid.uuid4().hex[:8]}"
    path.write_text(group, encoding="utf-8")
    return group


def _trial_result_from_db_row(row: dict) -> TrialResult:
    metrics = row["metrics"] or {}
    corr_sharpe = float(metrics.get("corr_sharpe", float("-inf")))
    flat_config = row["flat_config"]
    mmc_sharpe = metrics.get("mmc_sharpe")
    payout_score = metrics.get("payout_score")
    return TrialResult(
        trial_number=int(row["trial_number"]),
        sharpe=corr_sharpe,
        metrics=metrics,
        model_type=str(flat_config.get("model_1_type", "XGBoost")),
        elapsed_seconds=float(row["elapsed_seconds"] or 0.0),
        params=flat_config,
        error=row["error"],
        corr_sharpe=corr_sharpe,
        mmc_sharpe=float(mmc_sharpe)
        if mmc_sharpe is not None and np.isfinite(mmc_sharpe)
        else None,
        payout_score=float(payout_score)
        if payout_score is not None and np.isfinite(payout_score)
        else None,
    )


def _all_results_from_db(db: TrialDB) -> list[TrialResult]:
    return [_trial_result_from_db_row(row) for row in db.load_all_trials()]


def _trial_worker(
    flat_config: dict,
    data_dir: str,
    train_subsample: float,
    target_col: str,
    seed: int,
    result_queue: "multiprocessing.Queue[dict]",
    *,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    trial_number: int | None = None,
    wandb_diagnostics: bool = True,
) -> None:
    """Run a single trial inside a subprocess and push the result to the queue."""
    wandb_initialized = False
    X_train = None
    y_train = None
    targets_df = None
    era_train = None
    metrics: dict = {}
    try:
        load_dotenv()
        worker_config = dict(flat_config)
        worker_config["_data_dir"] = data_dir
        worker_config["_train_subsample"] = train_subsample
        worker_config.setdefault("primary_target", target_col)
        if "hpo_objective" not in worker_config:
            worker_config.setdefault("hpo_objective", "payout_score")

        feature_catalog = load_feature_catalog(data_dir)
        target_catalog = load_target_catalog(data_dir)
        strategy = strategy_from_flat(worker_config)
        routing = resolve_feature_routing(worker_config, feature_catalog)
        feature_columns = routing.feature_columns or None
        active_groups = list(worker_config.get("active_groups") or [])
        worker_config["active_groups"] = active_groups
        worker_config["active_groups_str"] = "+".join(active_groups)
        worker_config["active_groups_count"] = len(active_groups)
        worker_config["routed_feature_count"] = len(feature_columns or [])

        if strategy.target_mode == "multi_blend" and strategy.auxiliary_targets:
            X_train, y_train, targets_df, feature_cols = load_train_targets_frame(
                Path(data_dir),
                train_subsample=train_subsample,
                primary_target=strategy.primary_target,
                auxiliary_targets=strategy.auxiliary_targets,
                seed=seed,
                feature_columns=feature_columns,
                need_era=True,
            )
        else:
            X_train, y_train, feature_cols = load_train_only_frame(
                Path(data_dir),
                train_subsample=train_subsample,
                target_col=strategy.primary_target,
                seed=seed,
                feature_columns=feature_columns,
                need_era=True,
            )
            targets_df = None

        era_train = X_train["era"]
        targets_for_validation = (
            targets_df
            if targets_df is not None
            else pd.DataFrame({strategy.primary_target: y_train})
        )
        validation = validate_target_strategy_early(
            targets_for_validation,
            strategy,
            catalog=target_catalog,
            rng=random.Random(seed),
        )
        if not validation.ok:
            raise ValueError(validation.reason or "target strategy validation failed")
        worker_config = apply_target_strategy_to_flat(
            worker_config, validation.strategy
        )
        if validation.strategy.target_mode == "single":
            targets_df = None

        wandb_active = (
            wandb_project is not None
            and wandb_group is not None
            and trial_number is not None
        )
        if wandb_active:
            worker_config["log_wandb_diagnostics"] = wandb_diagnostics
            worker_config["wandb_log_shap"] = wandb_diagnostics
            import wandb

            from alphapulse.hpo.objective import TrialResult
            from alphapulse.logging_.wandb_logging import attach_wandb_loguru
            from alphapulse.logging_.wandb_utils import log_hpo_trial_metrics

            num = worker_config.get("num_models", 1)
            model_types = "+".join(
                str(worker_config.get(f"model_{i}_type", "?"))
                for i in range(1, num + 1)
            )
            preprocessors = worker_config.get("scaler_type", "StandardScaler")
            if worker_config.get("use_packboost"):
                preprocessors += "+Packboost"
            wandb.init(
                project=wandb_project,
                group=wandb_group,
                name=f"trial_{trial_number:03d}",
                config={
                    **worker_config,
                    "model_types": model_types,
                    "preprocessors": preprocessors,
                },
                reinit="finish_previous",
                settings=wandb.Settings(console="wrap"),
            )
            attach_wandb_loguru()
            wandb_initialized = True

        t0 = time.perf_counter()
        metrics = run_trial(
            worker_config,
            X_train=X_train,
            y_train=y_train,
            era_train=era_train,
            feature_cols=feature_cols,
            seed=seed,
            targets_df=targets_df,
            catalog=feature_catalog,
        )
        elapsed = time.perf_counter() - t0
        if wandb_initialized:
            assert trial_number is not None
            corr_sharpe = float(metrics.get("corr_sharpe", float("-inf")))
            mmc_sharpe = metrics.get("mmc_sharpe")
            payout_score = metrics.get("payout_score")
            trial_result = TrialResult(
                trial_number=trial_number,
                sharpe=corr_sharpe,
                metrics=metrics,
                model_type=str(worker_config.get("model_1_type", "XGBoost")),
                elapsed_seconds=elapsed,
                params=worker_config,
                corr_sharpe=corr_sharpe,
                mmc_sharpe=float(mmc_sharpe)
                if mmc_sharpe is not None and np.isfinite(mmc_sharpe)
                else None,
                payout_score=float(payout_score)
                if payout_score is not None and np.isfinite(payout_score)
                else None,
            )
            log_hpo_trial_metrics(
                trial_result,
                corr_sharpe,
                model_types=model_types,
                preprocessors=preprocessors,
            )

        result_queue.put(
            {
                "ok": True,
                "metrics": metrics,
                "elapsed_seconds": elapsed,
                "flat_config": worker_config,
            }
        )
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})
    finally:
        if X_train is not None:
            del X_train
        if y_train is not None:
            del y_train
        if targets_df is not None:
            del targets_df
        if era_train is not None:
            del era_train
        if metrics:
            del metrics
        release_cuda_memory()
        if wandb_initialized:
            import wandb

            from alphapulse.logging_.wandb_logging import detach_wandb_loguru

            detach_wandb_loguru()
            wandb.finish()
            gc.collect()


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


def _load_diagnostics_train_data(
    best_config: dict,
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
) -> tuple[
    pd.DataFrame, pd.Series, pd.DataFrame | None, list[str], dict[str, list[str]]
]:
    feature_catalog = load_feature_catalog(data_dir)
    routing = resolve_feature_routing(best_config, feature_catalog)
    feature_columns = routing.feature_columns or None
    strategy = strategy_from_flat(best_config)
    primary = strategy.primary_target or target_col

    if strategy.target_mode == "multi_blend" and strategy.auxiliary_targets:
        X_train, y_train, targets_df, feature_cols = load_train_targets_frame(
            data_dir,
            train_subsample=train_subsample,
            primary_target=primary,
            auxiliary_targets=strategy.auxiliary_targets,
            seed=seed,
            feature_columns=feature_columns,
            need_era=True,
        )
    else:
        X_train, y_train, feature_cols = load_train_only_frame(
            data_dir,
            train_subsample=train_subsample,
            target_col=primary,
            seed=seed,
            feature_columns=feature_columns,
            need_era=True,
        )
        targets_df = None

    return X_train, y_train, targets_df, feature_cols, routing.feature_groups


def _run_best_trial_diagnostics(
    *,
    best_config: dict,
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    wandb_project: str,
    wandb_group: str,
    feature_cols: list[str],
) -> None:
    """Retrain the best config and log a comprehensive XAI diagnostic WandB run.

    Splits the training data into an 80/20 era train/holdout split, retrains
    the best pipeline on the train portion, and logs universal feature importance,
    the per-era stability report, and era-stratified importance from the actual
    trained models. Results are logged as a dedicated 'best-trial-diagnostics' run
    within the same WandB group as the HPO trials.

    Args:
        best_config: Flat config dict of the best HPO trial.
        data_dir: Path to the data directory.
        train_subsample: Fraction of training data used (same as HPO).
        target_col: Target column name.
        seed: Random seed.
        wandb_project: WandB project name.
        wandb_group: WandB group name (same as HPO run group).
        feature_cols: Feature column names.
    """
    import wandb

    from alphapulse.evaluation.backtester import Backtester
    from alphapulse.evaluation.shap_report import compute_universal_feature_importance
    from alphapulse.evaluation.wandb_diagnostics import log_experiment_diagnostics
    from alphapulse.hpo.objective import _fit_pipeline
    from alphapulse.hpo.search_space import (
        get_train_kwargs_from_flat,
        resolve_flat_config,
    )
    from alphapulse.logging_.wandb_utils import log_importance_artifact

    try:
        X_train, y_train, targets_df, feature_cols, feature_groups = (
            _load_diagnostics_train_data(
                best_config,
                data_dir,
                train_subsample,
                target_col,
                seed,
            )
        )
        era_train = X_train["era"]
        primary_target = strategy_from_flat(best_config).primary_target

        eras_sorted = sorted(era_train.unique(), key=str)
        n_holdout = max(5, len(eras_sorted) // 5)
        holdout_set = set(eras_sorted[-n_holdout:])
        train_mask = ~era_train.isin(holdout_set)

        pipeline_cfg = resolve_flat_config(best_config)
        if best_config.get("use_gpu"):
            from alphapulse.hpo.search_space import apply_gpu_pipeline_config

            pipeline_cfg = apply_gpu_pipeline_config(pipeline_cfg)
        train_kwargs = get_train_kwargs_from_flat(best_config)

        targets_train = targets_df.loc[train_mask] if targets_df is not None else None
        pipeline = _fit_pipeline(
            pipeline_cfg,
            feature_cols,
            X_train.loc[train_mask],
            y_train.loc[train_mask],
            train_kwargs,
            flat_config=best_config,
            seed=seed,
            feature_groups=feature_groups or None,
            targets_df=targets_train,
        )

        ho_mask = era_train.isin(holdout_set)
        X_ho = X_train.loc[ho_mask]
        y_ho = y_train.loc[ho_mask]
        era_ho = era_train.loc[ho_mask]

        X_feat = X_ho[feature_cols]
        metrics = Backtester(pipeline, feature_columns=feature_cols).evaluate(
            X_ho, y_ho, era_ho
        )
        from alphapulse.hpo.objective import _merge_validation_mmc_metrics

        metrics, _ = _merge_validation_mmc_metrics(
            metrics,
            pipeline=pipeline,
            data_dir=data_dir,
            feature_cols=feature_cols,
            target_col=primary_target,
            train_subsample=train_subsample,
            seed=seed,
        )

        wandb.init(
            project=wandb_project,
            group=wandb_group,
            name="best-trial-diagnostics",
            job_type="diagnostics",
            config=best_config,
            reinit="finish_previous",
            settings=wandb.Settings(console="wrap"),
        )
        from alphapulse.logging_.wandb_logging import attach_wandb_loguru

        attach_wandb_loguru()

        log_experiment_diagnostics(
            pipeline=pipeline,
            X_val=X_ho,
            y_val=y_ho,
            era_val=era_ho,
            feature_cols=feature_cols,
            metrics=metrics,
            log_shap=True,
            log_feature_report=True,
            log_era_importance=True,
        )

        importance, _ = compute_universal_feature_importance(
            pipeline, X_feat, feature_cols=feature_cols, top_n=50
        )
        if importance:
            log_importance_artifact(importance, name="best-trial-feature-importance")

        from alphapulse.logging_.wandb_logging import detach_wandb_loguru

        detach_wandb_loguru()
        wandb.finish()
    except Exception as exc:
        logger.warning("Best-trial diagnostics failed: {}", exc)
        try:
            from alphapulse.logging_.wandb_logging import detach_wandb_loguru

            detach_wandb_loguru()
            wandb.finish()
        except Exception as finish_exc:
            logger.debug("wandb.finish cleanup error: {}", finish_exc)


def _run_local(
    *,
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    num_trials: int,
    output_dir: Path,
    objective: str = "payout_score",
    wandb_project: str | None = None,
    resume: bool = False,
    trial_timeout: int = 3600,
    gpu: bool = False,
    fast: bool = True,
    wandb_diagnostics: bool = True,
    max_hours: float | None = None,
    sampler: SamplerName = "tpe",
    n_startup_trials: int = DEFAULT_N_STARTUP_TRIALS,
) -> None:
    """Local HPO with subprocess isolation, Optuna TPE, and SQLite trial DB."""

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
    if gpu:
        preflight = cleanup_stale_gpu_processes(parent_pid=os.getpid())
        if preflight["killed_pids"]:
            logger.warning(
                "GPU preflight killed stale PIDs: {}",
                preflight["killed_pids"],
            )
        if preflight["remaining_gpu_pids"]:
            logger.warning(
                "GPU preflight: non-parent processes still on GPU: {}",
                preflight["remaining_gpu_pids"],
            )
    db_path = output_dir / "trials.db"
    wandb_group = _load_or_create_wandb_group(output_dir, wandb_project)
    if wandb_project:
        logger.info(
            "WandB logging enabled: project={} group={} diagnostics={}",
            wandb_project,
            wandb_group,
            wandb_diagnostics,
        )

    trial_kwargs: dict[str, str | float] = {
        "data_dir": str(data_dir),
        "train_subsample": train_subsample,
        "target_col": target_col,
    }

    results: list[TrialResult] = []
    best_score = float("-inf")
    best_config: dict = {}
    sweep_t0 = time.perf_counter()
    max_seconds = max_hours * 3600.0 if max_hours is not None else None
    if max_seconds is not None:
        logger.info(
            "Time budget: {:.1f}h ({} trials max cap)",
            max_hours,
            num_trials,
        )

    study = create_hpo_study(
        output_dir,
        seed=seed,
        sampler=sampler,
        resume=resume,
        n_startup_trials=n_startup_trials,
    )
    logger.info(
        "Optuna sampler: {} (storage={}, n_startup_trials={})",
        sampler,
        output_dir / "optuna.db",
        n_startup_trials if sampler == "tpe" else "n/a",
    )

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
            if max_seconds is not None:
                elapsed_sweep = time.perf_counter() - sweep_t0
                if elapsed_sweep >= max_seconds:
                    logger.info(
                        "Time budget reached ({:.1f}h elapsed), stopping after {} "
                        "trial(s)",
                        elapsed_sweep / 3600.0,
                        len(results),
                    )
                    break
            if i in already_done:
                logger.info(
                    "Trial {}/{}: skipped (already completed)", i + 1, num_trials
                )
                continue

            optuna_trial = study.ask()
            flat_config = suggest_flat_config(
                optuna_trial, fast=fast, data_dir=data_dir
            )
            if gpu:
                flat_config["use_gpu"] = True
            if fast:
                flat_config["hpo_fast"] = True
            flat_config["hpo_objective"] = objective
            db.insert_trial(i, flat_config)

            time_left_suffix = ""
            if max_seconds is not None:
                remaining_min = (
                    (max_seconds or 0) - (time.perf_counter() - sweep_t0)
                ) / 60
                time_left_suffix = f", {remaining_min:.0f}m left"

            logger.info(
                "Trial {}/{} starting (fast={}, models={}, groups={}, features={}{})",
                i + 1,
                num_trials,
                fast,
                flat_config.get("model_1_type", "?"),
                "+".join(flat_config.get("active_groups", [])) or "default",
                flat_config.get("routed_feature_count", "n/a"),
                time_left_suffix,
            )

            t0 = time.perf_counter()
            result_queue: multiprocessing.Queue = _MP_CTX.Queue()
            p = _MP_CTX.Process(
                target=_trial_worker,
                kwargs={
                    "flat_config": flat_config,
                    "data_dir": trial_kwargs["data_dir"],
                    "train_subsample": trial_kwargs["train_subsample"],
                    "target_col": trial_kwargs["target_col"],
                    "seed": seed + i,
                    "result_queue": result_queue,
                    "wandb_project": wandb_project,
                    "wandb_group": wandb_group,
                    "trial_number": i,
                    "wandb_diagnostics": wandb_diagnostics,
                },
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
                    worker_elapsed = payload.get("elapsed_seconds")
                    worker_flat = payload.get("flat_config")
                    if isinstance(worker_flat, dict):
                        if "ensemble_weights" in worker_flat:
                            flat_config["ensemble_weights"] = worker_flat[
                                "ensemble_weights"
                            ]
                    result = TrialResult(
                        trial_number=i,
                        sharpe=corr_sharpe,
                        metrics=metrics,
                        model_type=flat_config.get("model_1_type", "XGBoost"),
                        elapsed_seconds=float(worker_elapsed)
                        if worker_elapsed is not None
                        else elapsed,
                        params=flat_config,
                        corr_sharpe=corr_sharpe,
                    )
                    db.update_trial(
                        i,
                        status="completed",
                        metrics=metrics,
                        elapsed_seconds=elapsed,
                        flat_config=flat_config,
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

            if gpu:
                trial_failed_hard = result.error is not None
                cleanup = cleanup_after_trial_subprocess(
                    p.pid,
                    parent_pid=os.getpid(),
                    kill_worker_tree=trial_failed_hard,
                )
                if cleanup["killed_pids"]:
                    logger.warning(
                        "GPU cleanup killed PIDs: {}",
                        cleanup["killed_pids"],
                    )
                remaining = cleanup["remaining_gpu_pids"]
                if remaining:
                    logger.warning(
                        "GPU cleanup: other processes still on GPU: {}",
                        remaining,
                    )

            results.append(result)
            logger.info(
                "Trial {}/{}: corr_sharpe={:.4f}, payout_score={:.4f} ({:.1f}s){}",
                i + 1,
                num_trials,
                result.sharpe,
                float(result.metrics.get("payout_score", float("nan")))
                if result.metrics
                else float("nan"),
                result.elapsed_seconds,
                f" [ERROR: {result.error}]" if result.error else "",
            )
            print_leaderboard(
                logger,
                [entry_from_hpo_result(r) for r in _all_results_from_db(db)],
                current_trial=i,
            )

            if (
                wandb_project
                and wandb_group
                and not result.error
                and not wandb_diagnostics
            ):
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

            tell_trial_result(
                study,
                optuna_trial,
                trial_score,
                failed=bool(result.error),
            )

        best_score, best_config = _best_from_db(db, objective)
        results = _all_results_from_db(db)

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
        from alphapulse.logging_.wandb_utils import (
            log_hpo_convergence,
            log_hpo_summary_table,
        )

        log_hpo_summary_table(results, project=wandb_project, group=wandb_group)
        log_hpo_convergence(
            results,
            project=wandb_project,
            group=wandb_group,
            objective=objective,
        )
        logger.info("WandB summary table logged to project={}", wandb_project)

    if wandb_project and wandb_group and best_config:
        logger.info("Running best-trial XAI diagnostics in WandB...")
        _run_best_trial_diagnostics(
            best_config=best_config,
            data_dir=data_dir,
            train_subsample=train_subsample,
            target_col=target_col,
            seed=seed,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            feature_cols=feature_cols,
        )


def _run_ray(
    *,
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    num_trials: int,
    output_dir: Path,
    objective: str = "payout_score",
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
        "corr_sharpe", "mean_per_era_correlation", "max_drawdown", "payout_score"
    ] = "payout_score",
    wandb_project: str | None = None,
    trial_timeout: int = 3600,
    gpu: bool = False,
    fast: bool = True,
    wandb_diagnostics: bool = True,
    max_hours: float | None = None,
    sampler: SamplerName = "tpe",
    n_startup_trials: int = DEFAULT_N_STARTUP_TRIALS,
) -> None:
    """Run HPO search over preprocessing, models, and ensemble strategies.

    Use --local for Optuna-guided search without Ray, or omit for Ray Tune.
    Use --objective to choose the optimization target (default: payout_score).
    Use --sampler tpe for Bayesian optimization (default) or --sampler random.
    Use --n-startup-trials N to set TPE random exploration trials before
    Bayesian optimization (default: 25).
    Pass --wandb-project <name> to log every trial to Weights & Biases.
    The project name is suffixed with a launch timestamp
    (e.g. alphapulse-hpo-20260614-232943) and saved in the output dir for --resume.
    With WandB enabled, diagnostics (per-era charts, feature exposure, SHAP
    for XGBoost) are logged under the ``diagnostics/`` prefix in each trial run.
    Pass --no-wandb-diagnostics to log metrics only.
    Pass --resume to continue an interrupted sweep (requires --local).
    Pass --trial-timeout N to cap each subprocess trial at N seconds (default: 1800).
    Pass --max-hours N to stop after N wall-clock hours (local mode only).
    Pass --gpu to enable CUDA for XGBoost, LightGBM, CatBoost, and PackBoost.
    Fast mode (default) uses era holdout and a tighter search space so trials
    finish within ~30 minutes on full data.
    Pass --no-fast for full walk-forward evaluation (slower).
    """
    load_dotenv()
    set_global_seed(seed)
    if wandb_project:
        from alphapulse.logging_.wandb_utils import resolve_wandb_project

        wandb_project = resolve_wandb_project(wandb_project, output_dir=output_dir)
        logger.info("WandB project: {}", wandb_project)
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
            fast=fast,
            wandb_diagnostics=wandb_diagnostics,
            max_hours=max_hours,
            sampler=sampler,
            n_startup_trials=n_startup_trials,
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
