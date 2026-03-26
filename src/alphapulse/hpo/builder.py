from typing import Any, Literal

from ..models.base import BaseModel
from ..pipeline.multihead import HeadSpec, MultiHeadPipeline
from ..pipeline.pipeline import Pipeline
from ..preprocessors.base import BasePreprocessor
from ..preprocessors.grouped import GroupedPreprocessor
from .registry import MODEL_REGISTRY, PREPROCESSOR_REGISTRY


def _merge_params(
    defaults: dict[str, Any], override: dict[str, Any] | None
) -> dict[str, Any]:
    out = dict(defaults)
    if override:
        for k, v in override.items():
            if isinstance(v, dict) and k in out and isinstance(out[k], dict):
                out[k] = _merge_params(out[k], v)
            else:
                out[k] = v
    return out


def _build_grouped_preprocessor(item: dict[str, Any]) -> GroupedPreprocessor:
    params = item.get("params") or {}
    groups = params.get("groups")
    pipelines_cfg = params.get("pipelines")
    if not isinstance(groups, dict) or not isinstance(pipelines_cfg, dict):
        raise ValueError(
            "Grouped preprocessor requires params.groups and params.pipelines"
        )
    if set(groups.keys()) != set(pipelines_cfg.keys()):
        raise ValueError("Grouped preprocessor: groups and pipelines keys must match")
    group_preprocessors: dict[str, list[BasePreprocessor]] = {}
    for g_name, steps in pipelines_cfg.items():
        if not isinstance(steps, list):
            raise ValueError(f"Grouped pipelines[{g_name}] must be a list")
        group_preprocessors[g_name] = build_preprocessors(steps)
    return GroupedPreprocessor(
        groups={k: list(v) for k, v in groups.items()},
        group_preprocessors=group_preprocessors,
    )


def build_preprocessors(config: list[dict[str, Any]]) -> list[BasePreprocessor]:
    out: list[BasePreprocessor] = []
    for item in config:
        name = item.get("type")
        if name == "Grouped":
            out.append(_build_grouped_preprocessor(item))
            continue
        if not name or name not in PREPROCESSOR_REGISTRY:
            raise ValueError(f"Unknown preprocessor type: {name}")
        cls, defaults = PREPROCESSOR_REGISTRY[name]
        params = _merge_params(defaults, item.get("params"))
        out.append(cls(**params))
    return out


def build_models(config: list[dict[str, Any]]) -> list[BaseModel]:
    out: list[BaseModel] = []
    for i, item in enumerate(config):
        name = item.get("type")
        params = item.get("params") or {}
        if not name or name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {name}")
        cls, defaults = MODEL_REGISTRY[name]
        merged = _merge_params(defaults, params)
        if "name" not in merged:
            merged["name"] = f"{name}_{i}"
        out.append(cls(**merged))
    return out


def model_spec_needs_head_split(item: dict[str, Any]) -> bool:
    if item.get("input_columns") or item.get("input_group"):
        return True
    return len(item.get("preprocessors") or []) > 0


def needs_multi_head_pipeline(config: dict[str, Any]) -> bool:
    for m in config.get("models", []):
        if model_spec_needs_head_split(m):
            return True
    return False


def build_multi_head_pipeline(
    config: dict[str, Any],
    feature_columns: list[str] | None = None,
    feature_groups: dict[str, list[str]] | None = None,
) -> MultiHeadPipeline:
    global_preprocessors = build_preprocessors(config.get("preprocessors", []))
    models_config = config.get("models", [])
    if not models_config:
        raise ValueError("Config must have at least one model in 'models'.")
    fg = dict(feature_groups or config.get("feature_groups") or {})
    models = build_models(models_config)
    heads: list[HeadSpec] = []
    for i, m in enumerate(models_config):
        local_pres = build_preprocessors(m.get("preprocessors") or [])
        heads.append(
            HeadSpec(
                model=models[i],
                input_columns=m.get("input_columns"),
                input_group=m.get("input_group"),
                local_preprocessors=local_pres,
                feature_groups=fg,
            )
        )
    ensemble_method: Literal["single", "weighted", "stacking"] = config.get(
        "ensemble_method", "single"
    )
    ensemble_params = config.get("ensemble_params") or {}
    return MultiHeadPipeline(
        global_preprocessors=global_preprocessors,
        heads=heads,
        feature_columns=feature_columns,
        ensemble_method=ensemble_method,
        ensemble_params=ensemble_params,
    )


def build_pipeline(
    config: dict[str, Any],
    feature_columns: list[str] | None = None,
) -> Pipeline:
    """Construct a single-head ``Pipeline`` from a nested config dict.

    Args:
        config: Nested pipeline configuration with ``preprocessors``,
            ``models``, ``ensemble_method``, and ``ensemble_params`` keys.
        feature_columns: Explicit feature column names (optional).

    Returns:
        An unfitted ``Pipeline`` instance.
    """
    preprocessors = build_preprocessors(config.get("preprocessors", []))
    models_config = config.get("models", [])
    if not models_config:
        raise ValueError("Config must have at least one model in 'models'.")
    models = build_models(models_config)
    ensemble_method: Literal["single", "weighted", "stacking"] = config.get(
        "ensemble_method", "single"
    )
    ensemble_params = config.get("ensemble_params") or {}
    return Pipeline(
        preprocessors=preprocessors,
        models=models,
        feature_columns=feature_columns,
        ensemble_method=ensemble_method,
        ensemble_params=ensemble_params,
    )


def build_pipeline_or_multi(
    config: dict[str, Any],
    feature_columns: list[str] | None = None,
    feature_groups: dict[str, list[str]] | None = None,
) -> Pipeline | MultiHeadPipeline:
    """Build either a ``Pipeline`` or ``MultiHeadPipeline`` from config.

    Automatically selects ``MultiHeadPipeline`` when any model specifies
    ``input_columns``, ``input_group``, or local preprocessors.

    Args:
        config: Nested pipeline configuration dict.
        feature_columns: Explicit feature column names (optional).
        feature_groups: Mapping of group name to column lists for
            multi-head routing.

    Returns:
        An unfitted pipeline instance.
    """
    if needs_multi_head_pipeline(config):
        return build_multi_head_pipeline(
            config, feature_columns=feature_columns, feature_groups=feature_groups
        )
    return build_pipeline(config, feature_columns=feature_columns)
