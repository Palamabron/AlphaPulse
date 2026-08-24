"""AutoResearch: Claude-agent-driven ML research loop for Numerai pipelines.

Runs trials within a time/count budget, with a Claude agent deciding what
to try next (tune hyperparams, add models, change ensemble, etc.).

Each trial is scored via walk-forward backtesting (3 folds) rather than a
fixed holdout, so the leaderboard reflects temporal out-of-sample performance.

Outputs in --output-dir:
  best_config.json      Nested pipeline config with the best corr_sharpe found.
  research_state.json   Full trial history + agent reasoning.
  trials_summary.csv    One row per trial with metrics and action taken.
"""

import json
from pathlib import Path

import tyro
from loguru import logger

from alphapulse.autoresearch.loop import run_autoresearch
from alphapulse.experiments.data import load_train_only_frame
from alphapulse.utils import set_global_seed


def main(
    data_dir: Path,
    output_dir: Path,
    train_subsample: float = 0.125,
    hours: float | None = None,
    trials: int | None = 50,
    seed_config: Path | None = None,
    target_col: str = "target",
    seed: int = 42,
    agent_model: str = "claude-sonnet-4-6",
    resume: bool = False,
    wandb_project: str | None = None,
) -> None:
    """Run the AutoResearch loop.

    Args:
        data_dir: Numerai data directory (e.g. data/v5.2/).
        output_dir: Directory for outputs.
        train_subsample: Fraction of training rows to use (reduces trial time).
        hours: Wall-clock budget in hours. Stops when this OR --trials is hit.
        trials: Maximum trial count. Stops when this OR --hours is hit.
        seed_config: Path to a nested JSON config to start from (optional).
        target_col: Target column name.
        seed: Base random seed.
        agent_model: Claude model for the research agent decisions.
        resume: Resume from existing research_state.json in output_dir.
        wandb_project: WandB project name. When set, logs all trials to W&B.
    """
    set_global_seed(seed)
    if hours is None and trials is None:
        logger.error("Provide at least one of --hours or --trials.")
        raise SystemExit(1)

    logger.info("Loading data from {} (subsample={:.0%})", data_dir, train_subsample)
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

    start_config: dict | None = None
    if seed_config is not None:
        start_config = json.loads(seed_config.read_text())
        logger.info("Loaded seed config from {}", seed_config)

    state = run_autoresearch(
        X_train,
        y_train,
        era_train,
        feature_cols,
        max_hours=hours,
        max_trials=trials,
        output_dir=output_dir,
        seed_config=start_config,
        seed=seed,
        agent_model=agent_model,
        resume=resume,
        wandb_project=wandb_project,
        data_dir=data_dir,
        target_col=target_col,
    )

    n_ok = sum(1 for t in state.trials if t.error is None)
    logger.info(
        "\n=== AutoResearch complete: {} trials ({} successful) ===",
        len(state.trials),
        n_ok,
    )

    pareto_path = output_dir / "pareto_front.json"
    pareto_members = state.pareto_front.to_list()
    pareto_path.write_text(json.dumps(pareto_members, indent=2))
    logger.info(
        "Pareto front ({} configs) saved to {}", len(pareto_members), pareto_path
    )

    if state.best_trial:
        b = state.best_trial
        payout_str = (
            f", legacy_proxy={b.payout_score:.4f}" if b.payout_score is not None else ""
        )
        mmc_str = f", mmc_sharpe={b.mmc_sharpe:.4f}" if b.mmc_sharpe is not None else ""
        logger.info(
            "Best → trial #{}: corr_sharpe={:.4f}{}{}, models={}",
            b.trial_number,
            b.sharpe,
            mmc_str,
            payout_str,
            "+".join(b.model_types),
        )
        logger.info("Outputs saved to: {}", output_dir)


if __name__ == "__main__":
    tyro.cli(main)
