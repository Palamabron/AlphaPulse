"""Run a versioned experiment YAML/JSON (Experiment v1)."""

import json
from pathlib import Path
from typing import Any

import tyro

from alphapulse.utils import set_global_seed


def _build_wandb_config(
    exp: Any, *, config_path: str, seed: int, gpu: bool
) -> dict[str, Any]:
    models = exp.models
    preprocessors = exp.preprocessing

    model_types = "+".join(m.type for m in models)
    preprocessor_types = (
        "+".join(p.type for p in preprocessors) if preprocessors else "none"
    )

    cfg: dict[str, Any] = {
        "config_path": config_path,
        "seed": seed,
        "gpu": gpu,
        "target_col": exp.data.target_col,
        "train_subsample": exp.data.train_subsample,
        "feature_columns": "all"
        if exp.features.columns is None
        else len(exp.features.columns),
        "n_feature_groups": len(exp.features.groups),
        "n_models": len(models),
        "model_types": model_types,
        "n_preprocessors": len(preprocessors),
        "preprocessor_types": preprocessor_types,
        "ensemble_method": exp.ensemble_method,
        "n_rounds": exp.train.n_rounds,
        "early_stopping_rounds": exp.train.early_stopping_rounds,
        "neutralization_proportion": exp.neutralization.proportion,
        "primary_metric": exp.evaluation.primary_metric,
    }

    cfg["is_multihead"] = any(
        m.input_group is not None or m.input_columns is not None for m in models
    )

    for i, m in enumerate(models, start=1):
        cfg[f"model_{i}_type"] = m.type
        cfg[f"model_{i}_input_group"] = (
            m.input_group if m.input_group is not None else "all"
        )
        if m.input_columns is not None:
            cfg[f"model_{i}_input_columns_count"] = len(m.input_columns)
        for k, v in m.params.items():
            if not isinstance(v, dict | list):
                cfg[f"model_{i}_{k}"] = v
        inner = m.params.get("params", {})
        for k, v in inner.items():
            if not isinstance(v, dict | list):
                cfg[f"model_{i}_{k}"] = v
        for j, lp in enumerate(m.preprocessors, start=1):
            cfg[f"model_{i}_local_preprocessor_{j}_type"] = lp.type
            for k, v in lp.params.items():
                if not isinstance(v, dict | list):
                    cfg[f"model_{i}_local_preprocessor_{j}_{k}"] = v

    for i, p in enumerate(preprocessors, start=1):
        cfg[f"preprocessor_{i}_type"] = p.type
        for k, v in p.params.items():
            if not isinstance(v, dict | list):
                cfg[f"preprocessor_{i}_{k}"] = v

    for group_name, cols in exp.features.groups.items():
        cfg[f"feature_group_{group_name}_n_features"] = len(cols)

    return cfg


def main(
    config: Path = Path("experiments/example_v1.yaml"),
    artifact_dir: Path | None = Path("artifacts/experiments"),
    seed: int = 42,
    wandb_project: str | None = None,
    gpu: bool = False,
) -> None:
    """Load experiment file, train, evaluate; print metrics and config hash.

    Args:
        config: Path to experiment YAML or JSON config file.
        artifact_dir: Directory to save artifacts. None to disable.
        seed: Random seed.
        wandb_project: WandB project name. When set, logs config and metrics.
        gpu: Enable CUDA/GPU params for XGBoost and CatBoost models.
    """
    set_global_seed(seed)
    from alphapulse.experiments import run_experiment_from_path
    from alphapulse.experiments.runner import load_experiment_dict
    from alphapulse.experiments.schema import ExperimentV1

    if wandb_project:
        from alphapulse.logging_.wandb_utils import (
            init_wandb_run,
            resolve_wandb_project,
        )

        exp_dict = load_experiment_dict(config)
        exp_parsed = ExperimentV1.model_validate(exp_dict)
        wandb_cfg = _build_wandb_config(
            exp_parsed, config_path=str(config), seed=seed, gpu=gpu
        )
        resolved_project = resolve_wandb_project(wandb_project, output_dir=artifact_dir)
        init_wandb_run(project=resolved_project, name=config.stem, config=wandb_cfg)

    result = run_experiment_from_path(
        config,
        artifact_dir=artifact_dir,
        use_gpu=gpu,
        log_wandb_diagnostics=bool(wandb_project),
    )

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
