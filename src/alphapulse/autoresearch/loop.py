"""Time-bounded, agent-driven research loop."""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..evaluation.era_split import (
    WF_MIN_TRAIN_ERAS,
    WF_N_PURGE,
    WF_N_SPLITS,
    EraSplitEvaluator,
)
from ..experiments.split import internal_val_split
from ..hpo.builder import build_pipeline_or_multi
from ..hpo.search_space import resolve_flat_config, sample_random_config
from ..logging_.leaderboard import (
    entry_from_trial_record,
    print_leaderboard,
    save_leaderboard,
)
from ..validation.purge import effective_purge_eras
from . import agent as research_agent
from .mutations import (
    add_model,
    add_preprocessor,
    change_ensemble,
    remove_model,
    remove_preprocessor,
    set_neutralization,
    tune_model_params,
)
from .state import ResearchState, TrialRecord


def _run_one_trial(
    config: dict[str, Any],
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    era_train: pd.Series,
    feature_cols: list[str],
    seed: int,
    meta_model: pd.Series | None = None,
    target_col: str = "target",
) -> tuple[dict[str, float], float]:
    rng = np.random.default_rng(seed)
    np.random.seed(int(rng.integers(0, 2**31)))
    random.seed(seed)

    t0 = time.perf_counter()
    internal_purge_eras = effective_purge_eras(WF_N_PURGE, [target_col])

    def train_fn(X_tr: pd.DataFrame, y_tr: pd.Series) -> Any:
        pipeline = build_pipeline_or_multi(config, feature_columns=feature_cols)
        era_col = X_tr["era"] if "era" in X_tr.columns else None
        stacking_needs_val = (
            config.get("ensemble_method") == "stacking"
            and len(config.get("models", [])) > 1
        )
        X_fit, y_fit, X_val_inner, y_val_inner = internal_val_split(
            X_tr,
            y_tr,
            era_train=era_col,
            force_internal=stacking_needs_val,
            purge_eras=internal_purge_eras,
        )
        pipeline.fit(X_fit, y_fit, X_val=X_val_inner, y_val=y_val_inner)
        return pipeline

    metrics = EraSplitEvaluator(
        feature_columns=feature_cols,
        n_splits=WF_N_SPLITS,
        n_purge=internal_purge_eras,
        min_train_eras=WF_MIN_TRAIN_ERAS,
    ).evaluate_walk_forward(
        X_train, y_train, era_train, train_fn, meta_model=meta_model
    )
    return metrics, time.perf_counter() - t0


def _apply_decision(
    config: dict[str, Any],
    decision: research_agent.MutationDecision,
    seed: int,
) -> dict[str, Any]:
    name = decision.action_name
    kw = decision.action_kwargs

    if name == "tune_model_params":
        return tune_model_params(config, kw["model_index"], kw["param_updates"])
    if name == "add_model":
        return add_model(config, kw["model_type"], kw.get("params", {}))
    if name == "remove_model":
        return remove_model(config, kw["model_index"])
    if name == "change_ensemble":
        return change_ensemble(config, kw["method"], kw.get("params", {}))
    if name == "add_preprocessor":
        return add_preprocessor(
            config,
            kw["preprocessor_type"],
            kw.get("params", {}),
            kw.get("position", 999),
        )
    if name == "remove_preprocessor":
        return remove_preprocessor(config, kw["position"])
    if name == "set_neutralization":
        return set_neutralization(config, kw["proportion"])
    if name == "try_random_config":
        return resolve_flat_config(sample_random_config(seed=seed))
    raise ValueError(f"Unknown action: {name!r}")


def run_autoresearch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    era_train: pd.Series,
    feature_cols: list[str],
    *,
    max_hours: float | None = None,
    max_trials: int | None = 50,
    output_dir: Path,
    seed_config: dict[str, Any] | None = None,
    seed: int = 42,
    agent_model: str = "claude-sonnet-4-6",
    resume: bool = False,
    wandb_project: str | None = None,
    data_dir: Path | None = None,
    target_col: str = "target",
) -> ResearchState:
    """Run the agent-driven research loop.

    Stops when either max_hours wall-clock time or max_trials count is reached,
    whichever comes first. At least one must be provided.

    Each trial scores the pipeline via walk-forward backtesting (n_splits=3)
    rather than a fixed holdout, so the leaderboard reflects temporal
    out-of-sample performance.

    Args:
        max_hours: Wall-clock budget in hours. None = no limit.
        max_trials: Maximum number of trials. None = no limit.
        seed_config: Nested pipeline config to start from. Defaults to a random config.
        seed: Base random seed.

    Returns:
        ResearchState with full trial history and best config.
    """
    if max_hours is None and max_trials is None:
        raise ValueError("At least one of max_hours or max_trials must be set.")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trials_summary.csv"
    state_path = output_dir / "research_state.json"

    meta_model: pd.Series | None = None
    if data_dir is not None:
        from ..experiments.data import load_meta_model_series

        meta_model = load_meta_model_series(data_dir, X_train.index)
        if meta_model is not None:
            logger.info("Loaded meta_model.parquet for legacy MMC/proxy scoring")

    if wandb_project:
        from ..logging_.wandb_utils import finish_wandb_run, init_wandb_run

        init_wandb_run(
            project=wandb_project,
            name=f"autoresearch-{output_dir.name}",
            config={
                "max_hours": max_hours,
                "max_trials": max_trials,
                "seed": seed,
                "agent_model": agent_model,
            },
        )
        logger.info("WandB run initialized: project={}", wandb_project)

    if resume and state_path.exists():
        state = ResearchState.load(state_path)
        state.start_time = time.time()
        trial_num = len(state.trials)
        last_action = (
            state.trials[-1].action_taken if state.trials else "initial_config"
        )
        last_reasoning = (
            state.trials[-1].agent_reasoning if state.trials else "Resumed from state."
        )
        logger.info(
            "Resumed from {} ({} trials completed, best sharpe={:.4f})",
            state_path,
            trial_num,
            state.best_trial.sharpe if state.best_trial else float("-inf"),
        )
    else:
        state = ResearchState()
        state.start_time = time.time()
        state.current_config = (
            seed_config
            if seed_config is not None
            else resolve_flat_config(sample_random_config(seed=seed))
        )
        last_action = "initial_config"
        last_reasoning = (
            "Starting from provided seed config."
            if seed_config is not None
            else "Starting from random config."
        )
        trial_num = 0

    deadline = time.time() + max_hours * 3600 if max_hours is not None else float("inf")

    while True:
        if max_trials is not None and trial_num >= max_trials:
            logger.info("Trial budget reached ({} trials).", max_trials)
            break
        if time.time() >= deadline:
            logger.info("Time budget reached ({} hours).", max_hours)
            break

        n_models = len(state.current_config.get("models", []))
        logger.info(
            "Trial {}{} | action={} | models={}",
            trial_num,
            f"/{max_trials}" if max_trials else "",
            last_action,
            n_models,
        )

        try:
            metrics, elapsed = _run_one_trial(
                state.current_config,
                X_train=X_train,
                y_train=y_train,
                era_train=era_train,
                feature_cols=feature_cols,
                seed=seed + trial_num,
                meta_model=meta_model,
                target_col=target_col,
            )
            sharpe = metrics.get("corr_sharpe", float("-inf"))
            error = None
        except Exception as exc:
            logger.warning("Trial {} failed: {}", trial_num, exc)
            metrics = {}
            sharpe = float("-inf")
            elapsed = 0.0
            error = str(exc)

        model_types = [
            m.get("type", "?") for m in state.current_config.get("models", [])
        ]
        record = TrialRecord(
            trial_number=trial_num,
            sharpe=sharpe,
            metrics=metrics,
            config=state.current_config,
            model_types=model_types,
            elapsed_seconds=elapsed,
            action_taken=last_action,
            agent_reasoning=last_reasoning,
            error=error,
            mmc_sharpe=metrics.get("mmc_sharpe"),
            payout_score=metrics.get("payout_score"),
        )
        state.add_trial(record)

        best_sharpe = state.best_trial.sharpe if state.best_trial else float("-inf")
        logger.info(
            "  sharpe={:.4f} corr={:.4f} ({:.1f}s) | best={:.4f}{}",
            sharpe,
            metrics.get("mean_per_era_correlation", 0.0),
            elapsed,
            best_sharpe,
            " [ERROR]" if error else "",
        )
        print_leaderboard(
            logger,
            [entry_from_trial_record(t) for t in state.trials],
            current_trial=trial_num,
        )

        state.save(output_dir / "research_state.json")
        _append_csv_row(csv_path, record, write_header=(trial_num == 0))

        if wandb_project and not record.error:
            from ..logging_.wandb_utils import log_research_step

            log_research_step(
                trial_number=trial_num,
                metrics=record.metrics,
                model_types=record.model_types,
                action_taken=record.action_taken,
                elapsed_seconds=record.elapsed_seconds,
                sharpe=record.sharpe,
                mmc_sharpe=record.mmc_sharpe,
                payout_score=record.payout_score,
            )

        trial_num += 1

        if max_trials is not None and trial_num >= max_trials:
            break
        if time.time() >= deadline:
            break

        try:
            decision = research_agent.decide_next_action(state, model=agent_model)
            logger.info(
                "  Agent → {} | {}",
                decision.action_name,
                decision.reasoning[:120],
            )
            state.current_config = _apply_decision(
                state.current_config, decision, seed + trial_num
            )
            last_action = decision.action_name
            last_reasoning = decision.reasoning
        except Exception as exc:
            logger.warning("Agent/mutation error — keeping current config: {}", exc)
            last_action = "no_change_error"
            last_reasoning = str(exc)

    if state.best_trial is not None:
        import json

        best_path = output_dir / "best_config.json"
        best_path.write_text(json.dumps(state.best_trial.config, indent=2))
        logger.info(
            "Best trial #{}: sharpe={:.4f} | config saved to {}",
            state.best_trial.trial_number,
            state.best_trial.sharpe,
            best_path,
        )
    else:
        logger.warning("No successful trials — no best_config.json written.")

    save_leaderboard(
        output_dir / "leaderboard.json",
        [entry_from_trial_record(t) for t in state.trials],
    )
    logger.info("Leaderboard saved to: {}", output_dir / "leaderboard.json")

    state.save(output_dir / "research_state.json")

    if wandb_project:
        from ..logging_.wandb_utils import finish_wandb_run

        finish_wandb_run()

    return state


def _append_csv_row(path: Path, record: TrialRecord, write_header: bool) -> None:
    row = {
        "trial_number": record.trial_number,
        "sharpe": record.sharpe,
        "mmc_sharpe": record.mmc_sharpe if record.mmc_sharpe is not None else "",
        "payout_score": record.payout_score if record.payout_score is not None else "",
        "mean_per_era_correlation": record.metrics.get("mean_per_era_correlation", ""),
        "std_per_era_correlation": record.metrics.get("std_per_era_correlation", ""),
        "model_types": "+".join(record.model_types),
        "elapsed_seconds": f"{record.elapsed_seconds:.1f}",
        "action_taken": record.action_taken,
        "agent_reasoning": (record.agent_reasoning or "")[:200],
        "error": record.error or "",
    }
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
