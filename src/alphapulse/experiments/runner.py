import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from ..evaluation import Backtester
from ..evaluation.era_split import EraSplitEvaluator, evaluate_holdout_last_n_eras
from ..hpo.builder import TREE_MODEL_NAMES, build_pipeline_or_multi
from ..pipeline.multihead import MultiHeadPipeline
from ..pipeline.pipeline import Pipeline
from .data import load_train_frame_with_era, load_train_val_frames
from .hashing import config_hash
from .schema import ExperimentV1
from .split import internal_val_split


@dataclass
class RunResult:
    metrics: dict[str, float] = field(default_factory=dict)
    config_hash: str = ""
    duration_sec: float = 0.0
    paths: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    pipeline_config: dict[str, Any] = field(default_factory=dict)


def load_experiment_dict(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        with open(path, encoding="utf-8") as f:
            out = yaml.safe_load(f)
            if not isinstance(out, dict):
                raise ValueError(f"Expected mapping in {path}")
            return out
    with open(path, encoding="utf-8") as f:
        out = json.load(f)
        if not isinstance(out, dict):
            raise ValueError(f"Expected mapping in {path}")
        return out


def _need_era_column(exp: ExperimentV1) -> bool:
    for m in exp.models:
        if m.type == "Packboost" or m.type in TREE_MODEL_NAMES:
            return True
        for p in exp.preprocessing + m.preprocessors:
            if p.type == "Packboost":
                return True
    return False


def run_experiment(exp: ExperimentV1, *, artifact_dir: Path | None = None) -> RunResult:
    """Execute a full experiment: load data, build pipeline, train, and backtest.

    Args:
        exp: Validated experiment configuration (Experiment v1 schema).
        artifact_dir: If provided, resolved pipeline config and config hash
            are written to this directory.

    Returns:
        A ``RunResult`` containing backtest metrics, config hash, duration,
        artifact paths, and any error string.
    """
    t0 = time.perf_counter()
    pipeline_cfg = exp.to_pipeline_config()
    ch = config_hash(
        {
            "version": exp.version,
            "pipeline": pipeline_cfg,
            "data": exp.data.model_dump(mode="json"),
            "train": exp.train.model_dump(),
            "evaluation": exp.evaluation.model_dump(),
        }
    )
    need_era = _need_era_column(exp)
    data_dir = Path(exp.data.data_dir)

    try:
        X_train, y_train, X_val, y_val, era_val, feature_cols = load_train_val_frames(
            data_dir,
            train_subsample=exp.data.train_subsample,
            target_col=exp.data.target_col,
            seed=exp.data.seed,
            feature_columns=exp.features.columns,
            need_era=need_era,
        )
    except Exception as e:
        logger.exception("Experiment data load failed")
        return RunResult(
            error=str(e),
            config_hash=ch,
            duration_sec=time.perf_counter() - t0,
            pipeline_config=pipeline_cfg,
        )

    stacking_needs_val = exp.ensemble_method == "stacking" and len(exp.models) > 1
    era_train = X_train["era"] if "era" in X_train.columns else None
    X_train_fit, y_train_fit, X_val_internal, y_val_internal = internal_val_split(
        X_train,
        y_train,
        era_train=era_train,
        force_internal=stacking_needs_val,
    )

    pipeline: Pipeline | MultiHeadPipeline = build_pipeline_or_multi(
        pipeline_cfg, feature_columns=feature_cols, feature_groups=exp.features.groups
    )
    train_kw: dict[str, Any] = {
        "n_rounds": exp.train.n_rounds,
        "early_stopping_rounds": exp.train.early_stopping_rounds,
    }
    try:
        pipeline.fit(
            X_train_fit,
            y_train_fit,
            X_val=X_val_internal,
            y_val=y_val_internal,
            **train_kw,
        )
    except Exception as e:
        logger.exception("Pipeline fit failed")
        return RunResult(
            error=str(e),
            config_hash=ch,
            duration_sec=time.perf_counter() - t0,
            pipeline_config=pipeline_cfg,
        )

    backtester = Backtester(pipeline, feature_columns=feature_cols)
    metrics = backtester.evaluate(X_val, y_val, era_val)

    ev = exp.evaluation
    if ev.era_holdout_last_n is not None:
        ho = evaluate_holdout_last_n_eras(
            pipeline, X_val, y_val, era_val, feature_cols, ev.era_holdout_last_n
        )
        for k, v in ho.items():
            metrics[f"holdout_{k}"] = v

    if ev.walk_forward:
        X_wf, y_wf, era_wf, _ = load_train_frame_with_era(
            data_dir,
            train_subsample=exp.data.train_subsample,
            target_col=exp.data.target_col,
            seed=exp.data.seed,
            feature_columns=exp.features.columns,
            need_era=need_era,
        )

        def train_fn(X_tr: Any, y_tr: Any) -> Pipeline | MultiHeadPipeline:
            p = build_pipeline_or_multi(
                pipeline_cfg,
                feature_columns=feature_cols,
                feature_groups=exp.features.groups,
            )
            p.fit(X_tr, y_tr, **train_kw)
            return p

        wf_metrics = EraSplitEvaluator(
            feature_columns=feature_cols,
            min_train_eras=ev.walk_forward_min_train_eras,
            n_purge=ev.walk_forward_n_purge,
            n_embargo=ev.walk_forward_n_embargo,
            n_splits=ev.walk_forward_n_splits,
        ).evaluate_walk_forward(X_wf, y_wf, era_wf, train_fn)
        for k, v in wf_metrics.items():
            metrics[f"walk_forward_{k}"] = v

    paths: dict[str, str] = {}
    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = artifact_dir / "resolved_pipeline_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_cfg, f, indent=2)
        paths["resolved_pipeline_config"] = str(cfg_path)
        hash_path = artifact_dir / "config_hash.txt"
        with open(hash_path, "w", encoding="utf-8") as f:
            f.write(ch)
        paths["config_hash"] = str(hash_path)

    return RunResult(
        metrics=metrics,
        config_hash=ch,
        duration_sec=time.perf_counter() - t0,
        paths=paths,
        pipeline_config=pipeline_cfg,
    )


def run_experiment_from_path(path: Path, **kwargs: Any) -> RunResult:
    """Load an experiment YAML/JSON file and run it.

    Args:
        path: Path to a YAML or JSON experiment config file.
        **kwargs: Forwarded to ``run_experiment`` (e.g. ``artifact_dir``).

    Returns:
        A ``RunResult`` from the executed experiment.
    """
    d = load_experiment_dict(path)
    exp = ExperimentV1.model_validate(d)
    return run_experiment(exp, **kwargs)
