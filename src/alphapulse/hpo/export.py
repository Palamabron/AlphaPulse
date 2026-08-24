import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ..experiments.data import load_train_only_frame, load_train_targets_frame
from ..features.catalog import FeatureCatalog, load_feature_catalog, load_target_catalog
from ..pipeline.model_access import PipelineLike
from ..utils import set_global_seed
from .builder import TREE_MODEL_NAMES
from .feature_routing import FeatureRoutingResult, resolve_feature_routing
from .objective import _fit_pipeline, _resolve_pipeline_cfg
from .search_space import apply_gpu_pipeline_config, get_train_kwargs_from_flat
from .target_strategy import (
    TargetStrategy,
    apply_target_strategy_to_flat,
    strategy_from_flat,
    validate_target_strategy_early,
)


@dataclass(frozen=True)
class HpoBuildContext:
    flat: dict[str, Any]
    strategy: TargetStrategy
    routing: FeatureRoutingResult
    pipeline_cfg: dict[str, Any]
    feature_groups: dict[str, list[str]] | None
    feature_columns: list[str]


@dataclass(frozen=True)
class HpoFitResult:
    pipeline: PipelineLike
    feature_columns: list[str]
    pipeline_cfg: dict[str, Any]
    flat: dict[str, Any]
    primary_target: str


def needs_era_from_flat_config(flat: dict[str, Any]) -> bool:
    if bool(flat.get("use_packboost", False)):
        return True
    num_models = int(flat.get("num_models", 1))
    for i in range(1, min(num_models, 3) + 1):
        model_type = flat.get(f"model_{i}_type", "")
        if model_type == "Packboost" or model_type in TREE_MODEL_NAMES:
            return True
    return False


def prepare_hpo_flat(
    flat: dict[str, Any],
    data_dir: Path,
    *,
    target_col_fallback: str = "target",
) -> dict[str, Any]:
    out = dict(flat)
    out["_data_dir"] = str(data_dir)
    out.setdefault("primary_target", target_col_fallback)
    return out


def resolve_hpo_build_context(
    flat: dict[str, Any],
    *,
    catalog: FeatureCatalog | None = None,
) -> HpoBuildContext:
    data_dir = flat.get("_data_dir")
    if catalog is None:
        if not data_dir:
            raise ValueError("_data_dir required in flat config for HPO build")
        catalog = load_feature_catalog(data_dir)

    strategy = strategy_from_flat(flat)
    routing = resolve_feature_routing(flat, catalog)
    pipeline_cfg, feature_groups, routed_columns = _resolve_pipeline_cfg(flat, catalog)
    if flat.get("use_gpu"):
        pipeline_cfg = apply_gpu_pipeline_config(pipeline_cfg)

    if routed_columns:
        feature_columns = list(routed_columns)
    elif routing.feature_columns:
        feature_columns = list(routing.feature_columns)
    else:
        feature_columns = (
            catalog.columns("medium") if "medium" in catalog.feature_sets else []
        )

    return HpoBuildContext(
        flat=flat,
        strategy=strategy,
        routing=routing,
        pipeline_cfg=pipeline_cfg,
        feature_groups=feature_groups,
        feature_columns=feature_columns,
    )


def load_hpo_training_frames(
    context: HpoBuildContext,
    data_dir: Path,
    *,
    train_subsample: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, list[str]]:
    need_era = needs_era_from_flat_config(context.flat)
    feature_columns = context.feature_columns or None
    strategy = context.strategy

    if strategy.target_mode == "multi_blend" and strategy.auxiliary_targets:
        X_train, y_train, targets_df, feature_cols = load_train_targets_frame(
            data_dir,
            train_subsample=train_subsample,
            primary_target=strategy.primary_target,
            auxiliary_targets=strategy.auxiliary_targets,
            seed=seed,
            feature_columns=feature_columns,
            need_era=need_era,
        )
        return X_train, y_train, targets_df, feature_cols

    X_train, y_train, feature_cols = load_train_only_frame(
        data_dir,
        train_subsample=train_subsample,
        target_col=strategy.primary_target,
        seed=seed,
        feature_columns=feature_columns,
        need_era=need_era,
    )
    return X_train, y_train, None, feature_cols


def validate_and_apply_target_strategy(
    flat: dict[str, Any],
    targets_df: pd.DataFrame,
    data_dir: Path,
    *,
    allow_resample: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    strategy = strategy_from_flat(flat)
    catalog = load_target_catalog(data_dir) if allow_resample else None
    rng = random.Random(seed) if allow_resample and seed is not None else None
    validation = validate_target_strategy_early(
        targets_df,
        strategy,
        catalog=catalog,
        rng=rng,
    )
    if not validation.ok:
        raise ValueError(validation.reason or "target strategy validation failed")
    return apply_target_strategy_to_flat(flat, validation.strategy)


def fit_hpo_pipeline(
    context: HpoBuildContext,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    targets_df: pd.DataFrame | None = None,
    seed: int | None = None,
) -> PipelineLike:
    if seed is not None:
        set_global_seed(seed)
    train_kwargs = get_train_kwargs_from_flat(context.flat)
    return cast(
        PipelineLike,
        _fit_pipeline(
            context.pipeline_cfg,
            context.feature_columns,
            X_train,
            y_train,
            train_kwargs,
            flat_config=context.flat,
            seed=seed,
            feature_groups=context.feature_groups,
            targets_df=targets_df,
        ),
    )


def build_hpo_pipeline_from_flat(
    flat: dict[str, Any],
    data_dir: Path,
    *,
    train_subsample: float,
    seed: int,
    target_col_fallback: str = "target",
    allow_target_resample: bool = False,
) -> HpoFitResult:
    prepared = prepare_hpo_flat(flat, data_dir, target_col_fallback=target_col_fallback)
    data_seed = int(prepared.get("data_seed", seed))
    model_seed = int(prepared.get("model_seed", seed))
    context = resolve_hpo_build_context(prepared)
    X_train, y_train, targets_df, feature_cols = load_hpo_training_frames(
        context,
        data_dir,
        train_subsample=train_subsample,
        seed=data_seed,
    )
    context = HpoBuildContext(
        flat=prepared,
        strategy=context.strategy,
        routing=context.routing,
        pipeline_cfg=context.pipeline_cfg,
        feature_groups=context.feature_groups,
        feature_columns=feature_cols,
    )

    targets_for_validation = (
        targets_df
        if targets_df is not None
        else pd.DataFrame({context.strategy.primary_target: y_train})
    )
    validated_flat = validate_and_apply_target_strategy(
        prepared,
        targets_for_validation,
        data_dir,
        allow_resample=allow_target_resample,
        seed=data_seed,
    )
    context = resolve_hpo_build_context(validated_flat)
    context = HpoBuildContext(
        flat=validated_flat,
        strategy=context.strategy,
        routing=context.routing,
        pipeline_cfg=context.pipeline_cfg,
        feature_groups=context.feature_groups,
        feature_columns=feature_cols,
    )
    if context.strategy.target_mode == "single":
        targets_df = None

    pipeline = fit_hpo_pipeline(
        context,
        X_train,
        y_train,
        targets_df=targets_df,
        seed=model_seed,
    )
    primary = str(validated_flat.get("primary_target", target_col_fallback))
    return HpoFitResult(
        pipeline=pipeline,
        feature_columns=feature_cols,
        pipeline_cfg=context.pipeline_cfg,
        flat=validated_flat,
        primary_target=primary,
    )
