import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from ..evaluation import Backtester
from ..evaluation.era_split import EraSplitEvaluator, evaluate_holdout_last_n_eras
from ..hpo.builder import (
    TREE_MODEL_NAMES,
    build_multi_target_from_config,
    build_pipeline_or_multi,
)
from ..pipeline.multi_target import MultiTargetPipeline
from ..pipeline.multihead import MultiHeadPipeline
from ..pipeline.pipeline import Pipeline
from .data import (
    load_meta_model_series,
    load_train_frame_with_era,
    load_train_only_frame,
    load_train_targets_frame,
    load_validation_frames,
)
from .hashing import config_hash
from .pipeline_build import (
    experiment_target_flat,
    is_multi_target_experiment,
    needs_internal_val_for_experiment,
)
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


def _describe_models(exp: ExperimentV1) -> str:
    parts: list[str] = []
    for m in exp.models:
        era_tag = (
            f", era_ensemble n_subs={m.n_subs}" if m.type in TREE_MODEL_NAMES else ""
        )
        parts.append(f"{m.type}{era_tag}")
    return " + ".join(parts)


def _describe_preprocessors(exp: ExperimentV1) -> str:
    names = [p.type for p in exp.preprocessing]
    return " -> ".join(names) if names else "none"


def _describe_pipeline_models(
    pipeline: Pipeline | MultiHeadPipeline | MultiTargetPipeline,
) -> str:
    if isinstance(pipeline, MultiHeadPipeline):
        return f"MultiHead({len(pipeline.heads)} heads)"
    if isinstance(pipeline, MultiTargetPipeline):
        return f"MultiTarget({len(pipeline.target_columns)} targets)"
    return " + ".join(m.name for m in pipeline.models)


def run_experiment(
    exp: ExperimentV1,
    *,
    artifact_dir: Path | None = None,
    use_gpu: bool = False,
    log_wandb_diagnostics: bool | None = None,
) -> RunResult:
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
    from ..logging_.cli import configure_cli_logging

    configure_cli_logging()
    model_summary = _describe_models(exp)
    logger.info(
        "Experiment start: target={} train_subsample={} models=[{}] preprocessors=[{}] "
        "ensemble={} n_rounds={}",
        exp.data.target_col,
        exp.data.train_subsample,
        model_summary,
        _describe_preprocessors(exp),
        exp.ensemble_method,
        exp.train.n_rounds,
    )
    pipeline_cfg = exp.to_pipeline_config()
    if use_gpu:
        from ..hpo.search_space import apply_gpu_pipeline_config

        pipeline_cfg = apply_gpu_pipeline_config(pipeline_cfg)
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

    logger.info("Loading train data from {} ...", data_dir)
    multi_target = is_multi_target_experiment(exp)
    targets_df = None
    try:
        if multi_target:
            X_train, y_train, targets_df, feature_cols = load_train_targets_frame(
                data_dir,
                train_subsample=exp.data.train_subsample,
                primary_target=exp.data.target_col,
                auxiliary_targets=exp.data.auxiliary_targets,
                seed=exp.data.seed,
                feature_columns=exp.features.columns,
                need_era=need_era,
                benchmark_columns=exp.data.benchmark_columns or None,
            )
        else:
            X_train, y_train, feature_cols = load_train_only_frame(
                data_dir,
                train_subsample=exp.data.train_subsample,
                target_col=exp.data.target_col,
                seed=exp.data.seed,
                feature_columns=exp.features.columns,
                need_era=need_era,
                benchmark_columns=exp.data.benchmark_columns or None,
            )
    except Exception as e:
        logger.exception("Experiment train data load failed")
        return RunResult(
            error=str(e),
            config_hash=ch,
            duration_sec=time.perf_counter() - t0,
            pipeline_config=pipeline_cfg,
        )

    n_eras = int(X_train["era"].nunique()) if "era" in X_train.columns else 0
    logger.info(
        "Train loaded: rows={} features={} eras={}",
        len(X_train),
        len(feature_cols),
        n_eras,
    )

    stacking_needs_val = needs_internal_val_for_experiment(exp)
    era_train = X_train["era"] if "era" in X_train.columns else None
    X_train_fit, y_train_fit, X_val_internal, y_val_internal = internal_val_split(
        X_train,
        y_train,
        era_train=era_train,
        force_internal=stacking_needs_val,
    )
    targets_train_fit = None
    targets_val_internal = None
    if multi_target and targets_df is not None:
        targets_train_fit = targets_df.loc[X_train_fit.index]
        if X_val_internal is not None:
            targets_val_internal = targets_df.loc[X_val_internal.index]
    if X_val_internal is not None:
        logger.info(
            "Internal val split: train_rows={} val_rows={}",
            len(X_train_fit),
            len(X_val_internal),
        )

    target_flat = experiment_target_flat(exp)
    if multi_target:
        pipeline: Pipeline | MultiHeadPipeline | MultiTargetPipeline = (
            build_multi_target_from_config(
                pipeline_cfg,
                target_flat,
                feature_columns=feature_cols,
                feature_groups=exp.features.groups,
            )
        )
    else:
        pipeline = build_pipeline_or_multi(
            pipeline_cfg,
            feature_columns=feature_cols,
            feature_groups=exp.features.groups,
        )
    logger.info("Pipeline built: {}", _describe_pipeline_models(pipeline))
    train_kw: dict[str, Any] = {
        "n_rounds": exp.train.n_rounds,
        "early_stopping_rounds": exp.train.early_stopping_rounds,
    }
    logger.info(
        "Training started (n_rounds={}, early_stopping={}) ...",
        exp.train.n_rounds,
        exp.train.early_stopping_rounds,
    )
    try:
        era_train_fit = (
            era_train.loc[X_train_fit.index] if era_train is not None else None
        )
        era_val_fit = (
            era_train.loc[X_val_internal.index]
            if era_train is not None and X_val_internal is not None
            else None
        )
        if multi_target:
            assert targets_train_fit is not None
            train_metrics = pipeline.fit(
                X_train_fit.drop(columns=["era"], errors="ignore"),
                targets_train_fit,
                X_val=X_val_internal.drop(columns=["era"], errors="ignore")
                if X_val_internal is not None
                else None,
                targets_val=targets_val_internal,
                era_train=era_train_fit,
                era_val=era_val_fit,
                **train_kw,
            )
        else:
            train_metrics = pipeline.fit(
                X_train_fit,
                y_train_fit,
                X_val=X_val_internal,
                y_val=y_val_internal,
                era_val=era_val_fit,
                **train_kw,
            )
            if isinstance(pipeline, Pipeline):
                from ..hpo.objective import (
                    _needs_validation_ensemble_opt,
                    _optimize_ensemble_on_validation,
                )

                if _needs_validation_ensemble_opt(pipeline_cfg):
                    _optimize_ensemble_on_validation(
                        pipeline,
                        data_dir=data_dir,
                        feature_cols=feature_cols,
                        target_col=exp.data.target_col,
                        train_subsample=exp.data.train_subsample,
                        seed=exp.data.seed,
                        pipeline_cfg=pipeline_cfg,
                    )
                if pipeline.ensemble_weights is not None:
                    pipeline_cfg.setdefault("ensemble_params", {})["weights"] = (
                        pipeline.ensemble_weights
                    )
    except Exception as e:
        logger.exception("Pipeline fit failed")
        return RunResult(
            error=str(e),
            config_hash=ch,
            duration_sec=time.perf_counter() - t0,
            pipeline_config=pipeline_cfg,
        )
    logger.info("Training finished: {}", train_metrics)

    del X_train, y_train, X_train_fit, y_train_fit, X_val_internal, y_val_internal
    if era_train is not None:
        del era_train
    gc.collect()

    logger.info("Loading validation data ...")
    try:
        X_val, y_val, era_val = load_validation_frames(
            data_dir,
            exp.data.target_col,
            feature_cols,
            need_era=need_era,
        )
    except Exception as e:
        logger.exception("Experiment validation data load failed")
        return RunResult(
            error=str(e),
            config_hash=ch,
            duration_sec=time.perf_counter() - t0,
            pipeline_config=pipeline_cfg,
        )
    logger.info(
        "Validation loaded: rows={} eras={}", len(X_val), int(era_val.nunique())
    )

    meta_path = exp.evaluation.meta_model_path
    meta_model_preds = None
    meta_series = load_meta_model_series(
        data_dir, X_val.index, meta_model_path=meta_path
    )
    if meta_series is not None:
        meta_model_preds = meta_series.reindex(X_val.index).to_numpy(dtype=np.float64)

    if (
        isinstance(pipeline, Pipeline)
        and pipeline._meta_neutralizer is not None
        and meta_model_preds is not None
        and np.isfinite(meta_model_preds).sum() >= 2
    ):
        X_feat = X_val[feature_cols]
        base_preds = pipeline.predict(X_feat, eras=era_val, meta_model=None)
        optimized = pipeline._meta_neutralizer.optimize_proportion(
            base_preds,
            meta_model_preds,
            y_val,
            era_val,
            objective="payout_score",
            bounds=(0.5, 0.75),
            corr_weight=exp.evaluation.corr_weight,
            mmc_weight=exp.evaluation.mmc_weight,
        )
        pipeline.meta_neutralize_proportion = optimized

    compute_fnc = len(feature_cols) <= 200
    backtester = Backtester(pipeline, feature_columns=feature_cols)
    logger.info("Backtesting on validation set ...")
    metrics = backtester.evaluate(
        X_val,
        y_val,
        era_val,
        meta_model_preds=meta_model_preds,
        compute_fnc=compute_fnc,
        corr_weight=exp.evaluation.corr_weight,
        mmc_weight=exp.evaluation.mmc_weight,
    )
    logger.info(
        "Backtest done: corr_sharpe={:.4f} mean_corr={:.4f} pct_positive_eras={:.1%}",
        metrics.get("corr_sharpe", float("nan")),
        metrics.get("mean_per_era_correlation", float("nan")),
        metrics.get("pct_positive_eras", float("nan")),
    )

    should_log_diag = log_wandb_diagnostics
    if should_log_diag is None:
        try:
            import wandb

            should_log_diag = wandb.run is not None
        except ImportError:
            should_log_diag = False
    if should_log_diag:
        logger.info("Logging W&B diagnostics ...")
        from ..evaluation.wandb_diagnostics import log_experiment_diagnostics

        log_experiment_diagnostics(
            pipeline=pipeline,
            X_val=X_val,
            y_val=y_val,
            era_val=era_val,
            feature_cols=feature_cols,
            metrics=metrics,
            meta_model_preds=meta_model_preds,
            compute_fnc=compute_fnc,
        )

    ev = exp.evaluation
    if ev.era_holdout_last_n is not None:
        ho = evaluate_holdout_last_n_eras(
            pipeline, X_val, y_val, era_val, feature_cols, ev.era_holdout_last_n
        )
        for k, v in ho.items():
            metrics[f"holdout_{k}"] = v

    if ev.walk_forward:
        if multi_target:
            X_wf, y_wf, targets_wf, _ = load_train_targets_frame(
                data_dir,
                train_subsample=exp.data.train_subsample,
                primary_target=exp.data.target_col,
                auxiliary_targets=exp.data.auxiliary_targets,
                seed=exp.data.seed,
                feature_columns=exp.features.columns,
                need_era=need_era,
                benchmark_columns=exp.data.benchmark_columns or None,
            )
            era_wf = X_wf["era"]
        else:
            X_wf, y_wf, era_wf, _ = load_train_frame_with_era(
                data_dir,
                train_subsample=exp.data.train_subsample,
                target_col=exp.data.target_col,
                seed=exp.data.seed,
                feature_columns=exp.features.columns,
                need_era=need_era,
                benchmark_columns=exp.data.benchmark_columns or None,
            )
            targets_wf = None

        def train_fn(
            X_tr: Any, y_tr: Any
        ) -> Pipeline | MultiHeadPipeline | MultiTargetPipeline:
            if multi_target:
                assert targets_wf is not None
                p_mt: Pipeline | MultiHeadPipeline | MultiTargetPipeline = (
                    build_multi_target_from_config(
                        pipeline_cfg,
                        target_flat,
                        feature_columns=feature_cols,
                        feature_groups=exp.features.groups,
                    )
                )
                era_col = X_tr["era"] if "era" in X_tr.columns else None
                X_fit, _, X_val_inner, _ = internal_val_split(
                    X_tr, y_tr, era_train=era_col, force_internal=stacking_needs_val
                )
                targets_split = targets_wf.loc[X_tr.index]
                targets_train = targets_split.loc[X_fit.index]
                targets_val = (
                    targets_split.loc[X_val_inner.index]
                    if X_val_inner is not None
                    else None
                )
                era_train_wf = era_col.loc[X_fit.index] if era_col is not None else None
                era_val_wf = (
                    era_col.loc[X_val_inner.index]
                    if era_col is not None and X_val_inner is not None
                    else None
                )
                p_mt.fit(
                    X_fit.drop(columns=["era"], errors="ignore"),
                    targets_train,
                    X_val=X_val_inner.drop(columns=["era"], errors="ignore")
                    if X_val_inner is not None
                    else None,
                    targets_val=targets_val,
                    era_train=era_train_wf,
                    era_val=era_val_wf,
                    **train_kw,
                )
                return p_mt

            p = build_pipeline_or_multi(
                pipeline_cfg,
                feature_columns=feature_cols,
                feature_groups=exp.features.groups,
            )
            era_col = X_tr["era"] if "era" in X_tr.columns else None
            X_fit, y_fit, X_val_inner, y_val_inner = internal_val_split(
                X_tr, y_tr, era_train=era_col, force_internal=stacking_needs_val
            )
            era_val_wf = (
                era_col.loc[X_val_inner.index]
                if era_col is not None and X_val_inner is not None
                else None
            )
            p.fit(
                X_fit,
                y_fit,
                X_val=X_val_inner,
                y_val=y_val_inner,
                era_val=era_val_wf,
                **train_kw,
            )
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

    duration = time.perf_counter() - t0
    logger.info("Experiment complete in {:.1f}s", duration)
    return RunResult(
        metrics=metrics,
        config_hash=ch,
        duration_sec=duration,
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
