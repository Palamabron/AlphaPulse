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

from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing
import os
import queue
import random
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import tyro
from dotenv import load_dotenv
from loguru import logger

from alphapulse.experiments.data import (
    load_mmc_validation_frame,
    load_train_only_frame,
    load_train_targets_frame,
)
from alphapulse.features.catalog import load_feature_catalog, load_target_catalog
from alphapulse.hpo.feature_routing import resolve_feature_routing
from alphapulse.hpo.objective import TrialResult, run_trial
from alphapulse.hpo.optimization import (
    is_better_optimization_score,
    optimization_mode,
    worst_optimization_score,
)
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
    BestCriteria,
    entry_from_hpo_result,
    print_leaderboard,
    save_leaderboard,
    selection_score_from_metrics,
)
from alphapulse.utils import set_global_seed
from alphapulse.utils.gpu_cleanup import (
    cleanup_after_trial_subprocess,
    cleanup_stale_gpu_processes,
    release_cuda_memory,
    snapshot_process_tree,
)

_MP_CTX = multiprocessing.get_context("spawn")
_WANDB_GROUP_FILE = "wandb_group.txt"
_PROCESS_SNAPSHOT_INTERVAL_SECONDS = 10.0
CliBestCriteria = Literal["auto", "objective", "robust_payout"]
_PROTOCOL_VERSION = "corrected-hpo-2026-08-v5"
_TORCH_MODEL_TYPES = frozenset({"Packboost", "TabICL", "TabPFN", "TabPFN3"})
_SEM_FAILCRITICALERRORS = 0x0001
_SEM_NOGPFAULTERRORBOX = 0x0002
_WINDOWS_NATIVE_WORKER_ATTEMPTS = 5


def _configure_windows_cpu_affinity(
    logical_cpu: int | None,
    logical_cpu_count: int | None,
) -> list[int] | None:
    if logical_cpu is None and logical_cpu_count is None:
        return None
    if sys.platform != "win32":
        raise ValueError("CPU affinity workarounds are supported only on Windows")

    import psutil  # type: ignore[import-untyped]

    process = psutil.Process()
    available = [int(cpu) for cpu in process.cpu_affinity()]
    allowed = available
    if logical_cpu_count is not None:
        if logical_cpu_count < 1:
            raise ValueError("--logical-cpu-count must be >= 1")
        if logical_cpu_count > len(available):
            raise ValueError(
                "--logical-cpu-count cannot exceed the available logical CPU count"
            )
        allowed = available[:logical_cpu_count]
    if logical_cpu is not None and logical_cpu not in allowed:
        logger.warning(
            "Logical CPU {} is already outside the process affinity mask: {}",
            logical_cpu,
            allowed,
        )
    elif logical_cpu is not None:
        allowed = [cpu for cpu in allowed if cpu != logical_cpu]
    if not allowed:
        raise ValueError("Cannot exclude the only logical CPU available to the process")

    process.cpu_affinity(allowed)
    logger.warning(
        "Hardware workaround active: logical CPU {} excluded, CPU count limit={}; "
        "allowed CPUs={}",
        logical_cpu,
        logical_cpu_count,
        allowed,
    )
    return allowed


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_sha256(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _source_tree_sha256(repo_root: Path | None = None) -> str:
    root = repo_root or Path(__file__).resolve().parents[1]
    candidates = [
        root / "scripts" / "hpo_pipeline.py",
        root / "scripts" / "hpo_trial_worker.py",
        root / "pyproject.toml",
        root / "uv.lock",
        *(root / "src" / "alphapulse").rglob("*.py"),
    ]
    digest = hashlib.sha256()
    for path in sorted({p.resolve() for p in candidates if p.is_file()}):
        digest.update(path.relative_to(root.resolve()).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _resolved_model_pool(*, fast: bool, gpu: bool) -> list[str]:
    from alphapulse.hpo.search_space import (
        available_boosting_models,
        available_foundation_models,
    )

    boosting = available_boosting_models()
    if not gpu:
        boosting = [model for model in boosting if model != "Packboost"]
    return list(dict.fromkeys([*boosting, *available_foundation_models(hpo_fast=fast)]))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _hpo_protocol(
    *,
    data_dir: Path,
    X_train: pd.DataFrame,
    feature_cols: list[str],
    train_subsample: float,
    target_col: str,
    seed: int,
    objective: str,
    purge_eras: int,
    fast: bool,
    sampler: str,
    n_startup_trials: int,
    max_models: int,
    gpu: bool,
    multi_target: bool,
    excluded_logical_cpu: int | None,
    logical_cpu_count: int | None,
) -> dict[str, Any]:
    return {
        "protocol_version": _PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "source_tree_sha256": _source_tree_sha256(),
        "python_version": sys.version,
        "data_dir": str(data_dir.resolve()),
        "train_parquet_sha256": _sha256_file(data_dir / "train.parquet"),
        "features_json_sha256": _sha256_file(data_dir / "features.json"),
        "validation_parquet_sha256": _sha256_file(data_dir / "validation.parquet"),
        "meta_model_parquet_sha256": _sha256_file(data_dir / "meta_model.parquet"),
        "target_col": target_col,
        "train_subsample": train_subsample,
        "data_seed": seed,
        "sampled_rows": len(X_train),
        "sample_index_sha256": _sequence_sha256(X_train.index),
        "feature_columns_sha256": _sequence_sha256(feature_cols),
        "objective": objective,
        "purge_eras": purge_eras,
        "evaluation_mode": "purged_holdout" if fast else "purged_walk_forward",
        "sampler": sampler,
        "n_startup_trials": n_startup_trials,
        "max_models": max_models,
        "gpu": gpu,
        "multi_target": multi_target,
        "excluded_logical_cpu": excluded_logical_cpu,
        "logical_cpu_count": logical_cpu_count,
        "resolved_model_pool": _resolved_model_pool(fast=fast, gpu=gpu),
        "official_metric_reference": "numerai-tools==0.6.0",
    }


def _write_or_validate_protocol(
    output_dir: Path,
    protocol: dict[str, Any],
    *,
    resume: bool,
) -> None:
    path = output_dir / "protocol.json"
    if path.exists():
        if not resume:
            raise ValueError(
                "The output directory already contains an HPO protocol. "
                "Use --resume only for the same protocol or choose a new directory."
            )
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError(
                "HPO protocol differs from the existing run. Use a new output "
                "directory instead of mixing incomparable trials."
            )
        return
    if resume:
        raise ValueError(
            "Cannot safely resume because protocol.json is missing. "
            "Use a new output directory."
        )
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")


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


def _worker_wandb_enabled(
    *,
    project: str | None,
    group: str | None,
    trial_number: int | None,
    diagnostics: bool,
) -> bool:
    return (
        diagnostics
        and project is not None
        and group is not None
        and trial_number is not None
    )


def _config_requires_torch(flat_config: dict[str, Any]) -> bool:
    num_models = int(flat_config.get("num_models", 1))
    model_types = {
        str(flat_config.get(f"model_{index}_type", "XGBoost"))
        for index in range(1, num_models + 1)
    }
    return bool(
        model_types & _TORCH_MODEL_TYPES
        or flat_config.get("use_packboost")
        or flat_config.get("compression_type") == "autoencoder"
    )


def _suppress_windows_error_dialogs() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    ctypes.windll.kernel32.SetErrorMode(
        _SEM_FAILCRITICALERRORS | _SEM_NOGPFAULTERRORBOX
    )


def _start_trial_process(
    *,
    worker_kwargs: dict[str, Any],
    output_dir: Path,
    trial_number: int,
) -> tuple[Any, multiprocessing.Queue | None, tuple[Path, Path] | None]:
    if sys.platform != "win32":
        result_queue: multiprocessing.Queue = _MP_CTX.Queue()
        process = _MP_CTX.Process(
            target=_trial_worker,
            kwargs={**worker_kwargs, "result_queue": result_queue},
        )
        process.start()
        return process, result_queue, None

    input_path = output_dir / f".trial_{trial_number:04d}_worker_input.json"
    output_path = output_dir / f".trial_{trial_number:04d}_worker_output.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(worker_kwargs), encoding="utf-8")
    output_path.unlink(missing_ok=True)
    process = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "scripts.hpo_trial_worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
    )
    return process, None, (input_path, output_path)


def _process_is_alive(process: Any) -> bool:
    if isinstance(process, subprocess.Popen):
        return process.poll() is None
    return bool(process.is_alive())


def _process_join(process: Any, timeout: float | None = None) -> None:
    if isinstance(process, subprocess.Popen):
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        return
    process.join(timeout=timeout)


def _process_exitcode(process: Any) -> int | None:
    if isinstance(process, subprocess.Popen):
        return process.poll()
    return cast(int | None, process.exitcode)


def _load_worker_payload(
    result_queue: multiprocessing.Queue | None,
    worker_files: tuple[Path, Path] | None,
) -> dict[str, Any]:
    if worker_files is None:
        if result_queue is None:
            raise RuntimeError("worker result queue is missing")
        return cast(dict[str, Any], result_queue.get(timeout=5))

    _, output_path = worker_files
    if not output_path.exists():
        raise queue.Empty
    return cast(dict[str, Any], json.loads(output_path.read_text(encoding="utf-8")))


def _cleanup_worker_files(worker_files: tuple[Path, Path] | None) -> None:
    if worker_files is None:
        return
    input_path, output_path = worker_files
    input_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)


def _execute_trial_process(
    *,
    worker_kwargs: dict[str, Any],
    output_dir: Path,
    trial_number: int,
    trial_timeout: int,
    gpu: bool,
) -> tuple[dict[str, Any] | None, str | None, float, dict[str, Any]]:
    started_at = time.perf_counter()
    process, result_queue, worker_files = _start_trial_process(
        worker_kwargs=worker_kwargs,
        output_dir=output_dir,
        trial_number=trial_number,
    )
    worker_pid = process.pid
    if worker_pid is None:
        raise RuntimeError("trial subprocess did not receive a process ID")

    worker_snapshot: dict[int, str] = {}
    deadline = time.monotonic() + trial_timeout
    next_snapshot_at = time.monotonic()
    while _process_is_alive(process):
        now = time.monotonic()
        if gpu and now >= next_snapshot_at:
            worker_snapshot.update(snapshot_process_tree(worker_pid))
            next_snapshot_at = now + _PROCESS_SNAPSHOT_INTERVAL_SECONDS
        remaining_seconds = deadline - now
        if remaining_seconds <= 0:
            break
        _process_join(process, timeout=min(0.25, remaining_seconds))

    error: str | None = None
    payload: dict[str, Any] | None = None
    if _process_is_alive(process):
        process.terminate()
        _process_join(process, timeout=5)
        if _process_is_alive(process):
            process.kill()
            _process_join(process)
        error = f"timeout after {trial_timeout}s"
    else:
        exitcode = _process_exitcode(process)
        if exitcode != 0:
            error = f"subprocess exit code {exitcode}"
        else:
            try:
                payload = _load_worker_payload(result_queue, worker_files)
            except queue.Empty:
                error = "no result"

    _cleanup_worker_files(worker_files)
    cleanup: dict[str, Any] = {
        "killed_pids": [],
        "remaining_gpu_pids": [],
    }
    if gpu:
        cleanup = cleanup_after_trial_subprocess(
            worker_pid,
            parent_pid=os.getpid(),
            kill_worker_tree=error is not None,
            worker_snapshot=worker_snapshot,
        )
    return payload, error, time.perf_counter() - started_at, cleanup


def _trial_worker(
    flat_config: dict,
    data_dir: str,
    train_subsample: float,
    target_col: str,
    data_seed: int,
    model_seed: int,
    result_queue: multiprocessing.Queue[dict],
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
    mmc_frame = None
    metrics: dict = {}
    try:
        _suppress_windows_error_dialogs()
        load_dotenv()
        worker_config = dict(flat_config)
        worker_config["_data_dir"] = data_dir
        worker_config["_train_subsample"] = train_subsample
        worker_config["data_seed"] = data_seed
        worker_config["model_seed"] = model_seed
        worker_config.setdefault("primary_target", target_col)
        if "hpo_objective" not in worker_config:
            worker_config.setdefault("hpo_objective", "corr_sharpe")

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
                seed=data_seed,
                feature_columns=feature_columns,
                need_era=True,
            )
        else:
            X_train, y_train, feature_cols = load_train_only_frame(
                Path(data_dir),
                train_subsample=train_subsample,
                target_col=strategy.primary_target,
                seed=data_seed,
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
            rng=random.Random(data_seed),
        )
        if not validation.ok:
            raise ValueError(validation.reason or "target strategy validation failed")
        worker_config = apply_target_strategy_to_flat(
            worker_config, validation.strategy
        )
        if validation.strategy.target_mode == "single":
            targets_df = None

        logger.info(
            "Worker data ready: train_rows={} features={} model={}",
            len(X_train),
            len(feature_cols),
            worker_config.get("model_1_type", "unknown"),
        )
        mmc_frame = load_mmc_validation_frame(
            Path(data_dir),
            feature_cols=feature_cols,
            target_col=validation.strategy.primary_target,
            train_subsample=train_subsample,
            seed=data_seed,
        )
        logger.info("Worker validation data ready")
        set_global_seed(
            model_seed,
            seed_torch=_config_requires_torch(worker_config),
        )
        logger.info("Worker RNG initialization ready")

        wandb_active = _worker_wandb_enabled(
            project=wandb_project,
            group=wandb_group,
            trial_number=trial_number,
            diagnostics=wandb_diagnostics,
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
        logger.info("Worker trial evaluation starting")
        metrics = run_trial(
            worker_config,
            X_train=X_train,
            y_train=y_train,
            era_train=era_train,
            feature_cols=feature_cols,
            seed=model_seed,
            data_seed=data_seed,
            targets_df=targets_df,
            catalog=feature_catalog,
            mmc_frame=mmc_frame,
            mmc_frame_preloaded=True,
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
        error_traceback = traceback.format_exc()
        logger.exception("HPO trial worker failed: {}", exc)
        result_queue.put(
            {
                "ok": False,
                "error": str(exc),
                "traceback": error_traceback,
            }
        )
    finally:
        if X_train is not None:
            del X_train
        if y_train is not None:
            del y_train
        if targets_df is not None:
            del targets_df
        if era_train is not None:
            del era_train
        if mmc_frame is not None:
            del mmc_frame
        if metrics:
            del metrics
        release_cuda_memory()
        if wandb_initialized:
            import wandb

            from alphapulse.logging_.wandb_logging import detach_wandb_loguru

            detach_wandb_loguru()
            wandb.finish()
            gc.collect()


def _best_from_db(
    db: TrialDB,
    objective: str,
    *,
    criteria: BestCriteria = "objective",
) -> tuple[float, dict]:
    best_score = worst_optimization_score(objective)
    best_config: dict = {}
    for row in db.load_all_trials():
        if row["status"] != "completed" or not row["metrics"]:
            continue
        metrics = row["metrics"]
        score = selection_score_from_metrics(
            metrics, objective=objective, criteria=criteria
        )
        if not np.isfinite(score):
            continue
        if is_better_optimization_score(score, best_score, objective):
            best_score = score
            best_config = row["flat_config"]
    return best_score, best_config


_WORKER_RUNTIME_KEYS = frozenset(
    {
        "_data_dir",
        "_train_subsample",
        "log_wandb_diagnostics",
        "wandb_log_shap",
    }
)


def _persistable_flat_config(flat: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in flat.items() if k not in _WORKER_RUNTIME_KEYS}


def _resolve_best_criteria(
    objective: str,
    best_criteria: CliBestCriteria,
) -> BestCriteria:
    if best_criteria == "auto" and objective == "payout_score":
        return "robust_payout"
    if best_criteria == "auto":
        return "objective"
    return best_criteria


def _warn_resume_eval_mode_mismatch(
    *,
    resume: bool,
    fast: bool,
    completed_rows: list[dict],
) -> None:
    if not resume or not completed_rows:
        return
    prior_fast_flags = {
        bool(row["flat_config"].get("hpo_fast"))
        for row in completed_rows
        if row["status"] == "completed" and row.get("flat_config")
    }
    if len(prior_fast_flags) != 1:
        return
    prior_fast = prior_fast_flags.pop()
    if prior_fast == fast:
        return
    logger.warning(
        "Resume eval-mode mismatch: completed trials used fast={} but this run uses "
        "fast={}. New trials will not be comparable with earlier leaderboard scores.",
        prior_fast,
        fast,
    )


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
        data_seed = int(best_config.get("data_seed", seed))
        model_seed = int(best_config.get("model_seed", seed))
        set_global_seed(model_seed)
        X_train, y_train, targets_df, feature_cols, feature_groups = (
            _load_diagnostics_train_data(
                best_config,
                data_dir,
                train_subsample,
                target_col,
                data_seed,
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
            seed=model_seed,
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
            seed=data_seed,
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
    objective: str = "corr_sharpe",
    wandb_project: str | None = None,
    resume: bool = False,
    trial_timeout: int = 3600,
    gpu: bool = False,
    fast: bool = True,
    max_models: int = 2,
    wandb_diagnostics: bool = True,
    max_hours: float | None = None,
    sampler: SamplerName = "tpe",
    n_startup_trials: int = DEFAULT_N_STARTUP_TRIALS,
    best_criteria: CliBestCriteria = "auto",
    purge_eras: int = 8,
    multi_target: bool = False,
    excluded_logical_cpu: int | None = None,
    logical_cpu_count: int | None = None,
) -> None:
    """Local HPO with subprocess isolation, Optuna TPE, and SQLite trial DB."""

    if purge_eras < 0:
        raise ValueError("purge_eras must be >= 0")

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
    _write_or_validate_protocol(
        output_dir,
        _hpo_protocol(
            data_dir=data_dir,
            X_train=X_train,
            feature_cols=feature_cols,
            train_subsample=train_subsample,
            target_col=target_col,
            seed=seed,
            objective=objective,
            purge_eras=purge_eras,
            fast=fast,
            sampler=sampler,
            n_startup_trials=n_startup_trials,
            max_models=max_models,
            gpu=gpu,
            multi_target=multi_target,
            excluded_logical_cpu=excluded_logical_cpu,
            logical_cpu_count=logical_cpu_count,
        ),
        resume=resume,
    )
    del X_train, y_train
    gc.collect()
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
    best_score = worst_optimization_score(objective)
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
        objective=objective,
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
        resolved_best_criteria = _resolve_best_criteria(objective, best_criteria)
        if resolved_best_criteria == "robust_payout":
            logger.info(
                "Best config selection uses the robust legacy proxy "
                "(validation proxy penalized by weak holdout CORR)"
            )
        _warn_resume_eval_mode_mismatch(
            resume=resume,
            fast=fast,
            completed_rows=db.load_all_trials() if resume else [],
        )
        if resume:
            best_score, best_config = _best_from_db(
                db, objective, criteria=resolved_best_criteria
            )
            if best_config:
                logger.info(
                    "Resuming: global best {} ({})={:.4f} from {} completed trial(s)",
                    objective,
                    resolved_best_criteria,
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
                optuna_trial,
                fast=fast,
                max_models=max_models,
                data_dir=data_dir,
                use_gpu=gpu,
                primary_target=target_col,
            )
            if not multi_target:
                flat_config.update(
                    {
                        "target_mode": "single",
                        "primary_target": target_col,
                        "auxiliary_targets": [],
                        "target_blend_method": "equal",
                    }
                )
            if fast:
                flat_config["hpo_fast"] = True
            flat_config["hpo_objective"] = objective
            flat_config["purge_eras"] = purge_eras
            flat_config["data_seed"] = seed
            flat_config["model_seed"] = seed + i
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

            worker_kwargs = {
                "flat_config": flat_config,
                "data_dir": trial_kwargs["data_dir"],
                "train_subsample": trial_kwargs["train_subsample"],
                "target_col": trial_kwargs["target_col"],
                "data_seed": seed,
                "model_seed": seed + i,
                "wandb_project": wandb_project,
                "wandb_group": wandb_group,
                "trial_number": i,
                "wandb_diagnostics": wandb_diagnostics,
            }
            max_attempts = (
                _WINDOWS_NATIVE_WORKER_ATTEMPTS if sys.platform == "win32" else 1
            )
            total_elapsed = 0.0
            payload: dict[str, Any] | None = None
            process_error: str | None = None
            for attempt in range(1, max_attempts + 1):
                payload, process_error, attempt_elapsed, cleanup = (
                    _execute_trial_process(
                        worker_kwargs=worker_kwargs,
                        output_dir=output_dir,
                        trial_number=i,
                        trial_timeout=trial_timeout,
                        gpu=gpu,
                    )
                )
                total_elapsed += attempt_elapsed
                if cleanup["killed_pids"]:
                    logger.warning(
                        "GPU cleanup killed PIDs: {}", cleanup["killed_pids"]
                    )
                if cleanup["remaining_gpu_pids"]:
                    logger.warning(
                        "GPU cleanup: other processes still on GPU: {}",
                        cleanup["remaining_gpu_pids"],
                    )
                if process_error is None:
                    break
                if attempt < max_attempts:
                    logger.warning(
                        "Trial {}/{} worker attempt {}/{} failed: {}; retrying",
                        i + 1,
                        num_trials,
                        attempt,
                        max_attempts,
                        process_error,
                    )

            elapsed = total_elapsed
            if process_error is not None:
                error_msg = process_error
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
                trial_score = worst_optimization_score(objective)
                optuna_score = trial_score
            else:
                assert payload is not None
                if payload.get("ok"):
                    metrics = payload["metrics"]
                    corr_sharpe = metrics.get("corr_sharpe", float("-inf"))
                    trial_score = selection_score_from_metrics(
                        metrics,
                        objective=objective,
                        criteria=resolved_best_criteria,
                    )
                    optuna_score = selection_score_from_metrics(
                        metrics,
                        objective=objective,
                        criteria="objective",
                    )
                    worker_elapsed = payload.get("elapsed_seconds")
                    worker_flat = payload.get("flat_config")
                    completed_flat = (
                        _persistable_flat_config(worker_flat)
                        if isinstance(worker_flat, dict)
                        else flat_config
                    )
                    result = TrialResult(
                        trial_number=i,
                        sharpe=corr_sharpe,
                        metrics=metrics,
                        model_type=completed_flat.get("model_1_type", "XGBoost"),
                        elapsed_seconds=float(worker_elapsed)
                        if worker_elapsed is not None
                        else elapsed,
                        params=completed_flat,
                        corr_sharpe=corr_sharpe,
                    )
                    db.update_trial(
                        i,
                        status="completed",
                        metrics=metrics,
                        elapsed_seconds=elapsed,
                        flat_config=completed_flat,
                    )
                    flat_config = completed_flat
                else:
                    error_msg = payload.get("error", "unknown")
                    trial_score = worst_optimization_score(objective)
                    optuna_score = trial_score
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
            metrics = result.metrics or {}
            logger.info(
                "Trial {}/{}: val_sharpe={:.4f}, holdout_sharpe={:.4f}, "
                "legacy_proxy={:.4f} ({:.1f}s){}",
                i + 1,
                num_trials,
                float(metrics.get("val_corr_sharpe", float("nan"))),
                float(
                    metrics.get(
                        "holdout_corr_sharpe",
                        metrics.get("corr_sharpe", float("nan")),
                    )
                ),
                float(metrics.get("payout_score", float("nan"))),
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
                    objective=optuna_score,
                )

            if is_better_optimization_score(trial_score, best_score, objective):
                best_score = trial_score
                best_config = flat_config

            tell_trial_result(
                study,
                optuna_trial,
                optuna_score,
                failed=bool(result.error),
            )

        best_score, best_config = _best_from_db(
            db, objective, criteria=resolved_best_criteria
        )
        results = _all_results_from_db(db)

    best_path = output_dir / "best_config.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    logger.info(
        "Best {} score ({}) : {:.4f}",
        objective,
        resolved_best_criteria,
        best_score,
    )
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

    if wandb_project and wandb_group and best_config and wandb_diagnostics:
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


_RAY_OBJECTIVES = frozenset(
    {
        "corr_sharpe",
        "numerai_corr_sharpe",
        "mean_per_era_correlation",
        "max_drawdown",
    }
)


def _validate_ray_objective(objective: str) -> None:
    if objective in _RAY_OBJECTIVES:
        return
    allowed = ", ".join(sorted(_RAY_OBJECTIVES))
    raise ValueError(
        f"Ray HPO cannot optimize {objective!r}: distributed trials do not receive "
        f"the validation meta-model dataset. Use --local or choose one of: {allowed}."
    )


def _resolve_max_models(max_models: int | None, *, fast: bool) -> int:
    resolved = max_models if max_models is not None else (2 if fast else 3)
    if resolved < 1:
        raise ValueError("max_models must be >= 1")
    return resolved


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
    purge_eras: int = 8,
    fast: bool = True,
    max_models: int = 2,
    gpu: bool = False,
) -> None:
    """Ray Tune distributed HPO."""
    if purge_eras < 0:
        raise ValueError("purge_eras must be >= 0")
    _validate_ray_objective(objective)
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

    from alphapulse.hpo.objective import ray_model_seed, ray_trainable
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

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_protocol(
        output_dir,
        _hpo_protocol(
            data_dir=data_dir,
            X_train=X_train,
            feature_cols=feature_cols,
            train_subsample=train_subsample,
            target_col=target_col,
            seed=seed,
            objective=objective,
            purge_eras=purge_eras,
            fast=fast,
            sampler="ray",
            n_startup_trials=0,
            max_models=max_models,
            gpu=gpu,
            multi_target=False,
            excluded_logical_cpu=None,
            logical_cpu_count=None,
        ),
        resume=False,
    )

    ray.init(ignore_reinit_error=True)

    trainable = tune.with_parameters(
        ray_trainable,
        X_train=X_train,
        y_train=y_train,
        era_train=era_train,
        feature_cols=feature_cols,
        data_seed=seed,
    )

    param_space = get_full_param_space(
        fast=fast,
        max_models=max_models,
        use_gpu=gpu,
    )
    param_space["purge_eras"] = purge_eras
    param_space["hpo_fast"] = fast
    param_space["use_gpu"] = gpu
    param_space["target_mode"] = "single"
    param_space["primary_target"] = target_col
    param_space["auxiliary_targets"] = []
    param_space["target_blend_method"] = "equal"

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
        mode=optimization_mode(objective),
        progress_reporter=reporter,
        local_dir=str(output_dir / "ray_results"),
        verbose=1,
        callbacks=callbacks,
        resources_per_trial={"cpu": 1, "gpu": 1 if gpu else 0},
    )

    best_trial = analysis.best_trial
    best_config = best_trial.config if best_trial else {}
    if best_config:
        model_seed = ray_model_seed(best_config, seed)
        best_config = dict(best_config)
        best_config["data_seed"] = seed
        best_config["model_seed"] = model_seed

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
        "corr_sharpe",
        "numerai_corr_sharpe",
        "mean_per_era_correlation",
        "max_drawdown",
        "payout_score",
    ] = "corr_sharpe",
    wandb_project: str | None = None,
    trial_timeout: int = 3600,
    gpu: bool = False,
    fast: bool = True,
    max_models: int | None = None,
    wandb_diagnostics: bool = True,
    max_hours: float | None = None,
    sampler: SamplerName = "tpe",
    n_startup_trials: int = DEFAULT_N_STARTUP_TRIALS,
    best_criteria: CliBestCriteria = "auto",
    purge_eras: int = 8,
    multi_target: bool = False,
    exclude_logical_cpu: int | None = None,
    logical_cpu_count: int | None = None,
) -> None:
    """Run HPO search over preprocessing, models, and ensemble strategies.

    Use --local for Optuna-guided search without Ray, or omit for Ray Tune.
    Use --objective to choose the optimization target (default: corr_sharpe).
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
    Pass --trial-timeout N to cap each subprocess trial at N seconds (default: 3600).
    Pass --max-hours N to stop after N wall-clock hours (local mode only).
    Set --purge-eras 8 for explicit 20-day targets or 16 for 60-day targets.
    Pass --gpu to enable CUDA for XGBoost, LightGBM, CatBoost, and PackBoost.
    Pass --multi-target to include experimental multi-target configurations.
    Single-target search is the stable default on Windows.
    Pass --exclude-logical-cpu N on Windows to keep the HPO parent and every
    inherited trial worker off a known-faulty logical processor.
    Pass --logical-cpu-count N to restrict the process tree to the first N
    logical processors; use 16 to isolate a 13900K workload from CPUs 16-31.
    Fast mode (default) uses era holdout and a tighter search space so trials
    finish within ~30 minutes on full data.
    Pass --no-fast for full walk-forward evaluation (slower).
    Pass --max-models N to cap ensemble size per trial (default: 2 in fast mode, 3
    in walk-forward mode). Use --max-models 3 in fast mode for 3-model ensembles.
    The compatibility objective --objective payout_score is a legacy AlphaPulse
    proxy, not official Numerai payout. It defaults to robust-proxy selection;
    use --best-criteria objective to keep the raw validation proxy.
    """
    load_dotenv()
    _configure_windows_cpu_affinity(exclude_logical_cpu, logical_cpu_count)
    set_global_seed(seed, seed_torch=False)
    resolved_max_models = _resolve_max_models(max_models, fast=fast)
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
            max_models=resolved_max_models,
            wandb_diagnostics=wandb_diagnostics,
            max_hours=max_hours,
            sampler=sampler,
            n_startup_trials=n_startup_trials,
            best_criteria=best_criteria,
            purge_eras=purge_eras,
            multi_target=multi_target,
            excluded_logical_cpu=exclude_logical_cpu,
            logical_cpu_count=logical_cpu_count,
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
            purge_eras=purge_eras,
            fast=fast,
            max_models=resolved_max_models,
            gpu=gpu,
        )


if __name__ == "__main__":
    tyro.cli(main)
