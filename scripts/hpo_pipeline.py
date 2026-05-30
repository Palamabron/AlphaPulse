"""HPO pipeline: search over preprocessing, models, and ensembles.

Supports two modes:
  --local   Random search (no extra dependencies).
  (default) Ray Tune distributed search (requires ``pip install 'alphapulse[hpo]'``).
"""

import json
import time
from pathlib import Path

import tyro
from loguru import logger

from alphapulse.experiments.data import load_train_val_frames
from alphapulse.hpo.objective import TrialResult, run_trial
from alphapulse.hpo.search_space import sample_random_config
from alphapulse.logging_.leaderboard import (
    entry_from_hpo_result,
    print_leaderboard,
    save_leaderboard,
)


def _run_local(
    *,
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    num_trials: int,
    output_dir: Path,
) -> None:
    """Local random-search HPO (no Ray dependency)."""
    need_era = True
    X_train, y_train, X_val, y_val, era_val, feature_cols = load_train_val_frames(
        data_dir,
        train_subsample=train_subsample,
        target_col=target_col,
        seed=seed,
        feature_columns=None,
        need_era=need_era,
    )
    logger.info(
        "Data loaded: train={}, val={}, features={}",
        X_train.shape,
        X_val.shape,
        len(feature_cols),
    )

    results: list[TrialResult] = []
    best_sharpe = float("-inf")
    best_config: dict = {}

    for i in range(num_trials):
        flat_config = sample_random_config(seed=seed + i)
        t0 = time.perf_counter()
        try:
            metrics = run_trial(
                flat_config,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                era_val=era_val,
                feature_cols=feature_cols,
                seed=seed + i,
            )
            elapsed = time.perf_counter() - t0
            sharpe = metrics.get("sharpe", float("-inf"))
            result = TrialResult(
                trial_number=i,
                sharpe=sharpe,
                metrics=metrics,
                model_type=flat_config.get("model_1_type", "XGBoost"),
                elapsed_seconds=elapsed,
                params=flat_config,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.warning("Trial {} failed: {}", i, e)
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
            "Trial {}/{}: sharpe={:.4f} ({:.1f}s){}",
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

        if result.sharpe > best_sharpe:
            best_sharpe = result.sharpe
            best_config = flat_config

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_config.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    logger.info("Best sharpe: {:.4f}", best_sharpe)
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


def _run_ray(
    *,
    data_dir: Path,
    train_subsample: float,
    target_col: str,
    seed: int,
    num_trials: int,
    output_dir: Path,
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

    need_era = True
    X_train, y_train, X_val, y_val, era_val, feature_cols = load_train_val_frames(
        data_dir,
        train_subsample=train_subsample,
        target_col=target_col,
        seed=seed,
        feature_columns=None,
        need_era=need_era,
    )

    ray.init(ignore_reinit_error=True)

    trainable = tune.with_parameters(
        ray_trainable,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        era_val=era_val,
        feature_cols=feature_cols,
    )

    param_space = get_full_param_space()

    reporter = CLIReporter(
        metric_columns=["sharpe", "mean_per_era_correlation"],
        max_report_frequency=30,
    )

    analysis = tune.run(
        trainable,
        config=param_space,
        num_samples=num_trials,
        metric="sharpe",
        mode="max",
        progress_reporter=reporter,
        local_dir=str(output_dir / "ray_results"),
        verbose=1,
    )

    best_trial = analysis.best_trial
    best_config = best_trial.config if best_trial else {}

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_config.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, default=str)
    logger.info("Best config saved to: {}", best_path)
    best_sharpe = analysis.best_result.get("sharpe") if best_trial else "N/A"
    logger.info("Best sharpe: {}", best_sharpe)

    ray.shutdown()


def main(
    data_dir: Path = Path("data/v5.2"),
    train_subsample: float = 0.125,
    target_col: str = "target",
    seed: int = 42,
    num_trials: int = 30,
    output_dir: Path = Path("artifacts/hpo"),
    local: bool = False,
) -> None:
    """Run HPO search over preprocessing, models, and ensemble strategies.

    Use --local for random search without Ray, or omit for Ray Tune.
    """
    if local:
        _run_local(
            data_dir=data_dir,
            train_subsample=train_subsample,
            target_col=target_col,
            seed=seed,
            num_trials=num_trials,
            output_dir=output_dir,
        )
    else:
        _run_ray(
            data_dir=data_dir,
            train_subsample=train_subsample,
            target_col=target_col,
            seed=seed,
            num_trials=num_trials,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    tyro.cli(main)
