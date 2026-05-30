"""Run a versioned experiment YAML/JSON (Experiment v1)."""

import json
from pathlib import Path
from typing import Any

import tyro


def main(
    config: Path = Path("experiments/example_v1.yaml"),
    artifact_dir: Path | None = Path("artifacts/experiments"),
) -> None:
    """Load experiment file, train, evaluate; print metrics and config hash."""
    from alphapulse.experiments import run_experiment_from_path

    result = run_experiment_from_path(config, artifact_dir=artifact_dir)
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
