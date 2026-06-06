"""Run a versioned experiment YAML/JSON (Experiment v1)."""

import json
from pathlib import Path
from typing import Any

import tyro

from alphapulse.utils import set_global_seed


def main(
    config: Path = Path("experiments/example_v1.yaml"),
    artifact_dir: Path | None = Path("artifacts/experiments"),
    seed: int = 42,
    wandb_project: str | None = None,
) -> None:
    """Load experiment file, train, evaluate; print metrics and config hash.

    Args:
        config: Path to experiment YAML or JSON config file.
        artifact_dir: Directory to save artifacts. None to disable.
        seed: Random seed.
        wandb_project: WandB project name. When set, logs config and metrics.
    """
    set_global_seed(seed)
    from alphapulse.experiments import run_experiment_from_path

    if wandb_project:
        from alphapulse.logging_.wandb_utils import init_wandb_run

        init_wandb_run(
            project=wandb_project,
            name=config.stem,
            config={"config_path": str(config), "seed": seed},
        )

    result = run_experiment_from_path(config, artifact_dir=artifact_dir)

    if wandb_project:
        from alphapulse.logging_.wandb_utils import (
            finish_wandb_run,
            log_backtest_results,
        )

        if not result.error and result.metrics:
            log_backtest_results(result.metrics)
            import wandb

            wandb.run.summary["config_hash"] = result.config_hash  # type: ignore[union-attr]
            wandb.run.summary["duration_sec"] = result.duration_sec  # type: ignore[union-attr]
            for name, path in result.paths.items():
                wandb.run.summary[f"artifact_{name}"] = path  # type: ignore[union-attr]
        finish_wandb_run()

    if result.error:
        print(json.dumps({"error": result.error}, indent=2))
        raise SystemExit(1)
    out: dict[str, Any] = {
        "config_hash": result.config_hash,
        "duration_sec": result.duration_sec,
        "metrics": result.metrics,
        "paths": result.paths,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
