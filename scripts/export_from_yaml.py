"""Export a trained pipeline as Numerai-ready ``predict.pkl`` from a YAML config.

Unlike ``export_numerai_pickle.py`` (which exports from an HPO ``best_config.json``),
this script trains and exports directly from a YAML experiment definition::

    uv run python scripts/export_from_yaml.py \\
        --config experiments/my_experiment.yaml \\
        --output-dir artifacts/export
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import cloudpickle
import tyro
from loguru import logger

from alphapulse.experiments.runner import load_experiment_dict
from alphapulse.experiments.schema import ExperimentV1
from alphapulse.experiments.split import internal_val_split
from alphapulse.hpo.builder import build_pipeline_or_multi


def _needs_era(exp: ExperimentV1) -> bool:
    for m in exp.models:
        if m.type == "Packboost":
            return True
        for p in exp.preprocessing + m.preprocessors:
            if p.type == "Packboost":
                return True
    return False


def main(
    config: Path = Path("experiments/example_v1.yaml"),
    output_dir: Path = Path("artifacts/export"),
    benchmark_col: str | None = None,
) -> None:
    """Train a YAML-defined experiment and export a Numerai-compatible predict.pkl.

    Args:
        config: Path to the experiment YAML (or JSON) config file.
        output_dir: Directory to write predict.pkl and pipeline.pkl.
        benchmark_col: Optional benchmark column name for Numerai submission wrapper.
            When omitted, the pipeline predicts without benchmark blending.
    """
    if not config.exists():
        raise FileNotFoundError(f"config not found: {config}")

    d = load_experiment_dict(config)
    exp = ExperimentV1.model_validate(d)

    data_dir = Path(exp.data.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    logger.info("Config: {}", config)
    logger.info(
        "Data: {} (subsample={}, target={})",
        data_dir,
        exp.data.train_subsample,
        exp.data.target_col,
    )

    need_era = _needs_era(exp)
    from alphapulse.experiments.data import load_train_only_frame

    X_train, y_train, feature_cols = load_train_only_frame(
        data_dir=data_dir,
        train_subsample=exp.data.train_subsample,
        target_col=exp.data.target_col,
        seed=exp.data.seed,
        feature_columns=exp.features.columns,
        need_era=need_era,
    )
    gc.collect()

    logger.info(
        "Loaded train: {} rows, {} features ({:.1f} MB)",
        len(X_train),
        len(feature_cols),
        X_train.memory_usage(deep=True).sum() / 1e6,
    )

    pipeline_cfg = exp.to_pipeline_config()
    pipeline = build_pipeline_or_multi(
        pipeline_cfg,
        feature_columns=feature_cols,
        feature_groups=exp.features.groups,
    )

    stacking_needs_val = exp.ensemble_method == "stacking" and len(exp.models) > 1
    era_train = X_train["era"] if "era" in X_train.columns else None
    X_fit, y_fit, X_val_internal, y_val_internal = internal_val_split(
        X_train,
        y_train,
        era_train=era_train,
        force_internal=stacking_needs_val,
    )
    del X_train, y_train
    gc.collect()

    train_kw: dict[str, Any] = {
        "n_rounds": exp.train.n_rounds,
        "early_stopping_rounds": exp.train.early_stopping_rounds,
    }

    logger.info("Fitting pipeline...")
    pipeline.fit(X_fit, y_fit, X_val=X_val_internal, y_val=y_val_internal, **train_kw)
    del X_fit, y_fit, X_val_internal, y_val_internal
    gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "resolved_pipeline_config.json").write_text(
        json.dumps(pipeline_cfg, indent=2), encoding="utf-8"
    )

    predict_fn = pipeline.to_numerai_predict(benchmark_col)
    with open(output_dir / "predict.pkl", "wb") as f:
        cloudpickle.dump(predict_fn, f)

    pipeline.save_pipeline(output_dir / "pipeline.pkl")

    logger.info("Exported Numerai predict to: {}", output_dir / "predict.pkl")
    logger.info("Saved trained pipeline to:   {}", output_dir / "pipeline.pkl")


if __name__ == "__main__":
    tyro.cli(main)
