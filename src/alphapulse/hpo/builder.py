from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..models.base import BaseModel
from ..models.era_ensemble_model import EraEnsembleModel
from ..pipeline.multi_target import MultiTargetPipeline
from ..pipeline.multihead import HeadSpec, MultiHeadPipeline
from ..pipeline.pipeline import Pipeline
from ..preprocessors.base import BasePreprocessor
from ..preprocessors.grouped import GroupedPreprocessor
from .registry import MODEL_REGISTRY, PREPROCESSOR_REGISTRY
from .search_space import strip_catboost_gpu_incompatible_params

TREE_MODEL_NAMES = frozenset(
    {"XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees"}
)


class _PipelineModelAdapter(BaseModel):
    def __init__(self, pipeline: Pipeline | MultiHeadPipeline) -> None:
        super().__init__(name=f"Adapter_{type(pipeline).__name__}")
        self._pipeline = pipeline

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        metrics = self._pipeline.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, **kwargs
        )
        self.is_trained = True
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict(X)

    def save(self, path: Path) -> None:
        self._pipeline.save_pipeline(path)

    def load(self, path: Path) -> "_PipelineModelAdapter":
        raise NotImplementedError("Adapter load is not supported")


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


def _make_base_factory(
    cls: type[BaseModel], params: dict[str, Any]
) -> Callable[[], BaseModel]:
    def factory() -> BaseModel:
        return cls(**params)

    return factory


def instantiate_model(
    type_name: str,
    override: dict[str, Any] | None,
    *,
    index: int = 0,
    n_subs: int = 10,
    use_era_ensemble: bool = True,
) -> BaseModel:
    if not type_name or type_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {type_name}")
    cls, defaults = MODEL_REGISTRY[type_name]
    merged = _merge_params(defaults, override or {})
    if type_name == "CatBoost":
        merged = strip_catboost_gpu_incompatible_params(merged)
    if "name" not in merged:
        merged["name"] = f"{type_name}_{index}"

    if use_era_ensemble and type_name in TREE_MODEL_NAMES:
        factory = _make_base_factory(cls, dict(merged))
        return EraEnsembleModel(
            base_model_factory=factory,
            n_subs=n_subs,
            era_column="era",
            name=f"EraEnsemble_{merged['name']}",
        )
    return cls(**merged)


def build_models(config: list[dict[str, Any]]) -> list[BaseModel]:
    out: list[BaseModel] = []
    for i, item in enumerate(config):
        name = item.get("type")
        if not name:
            raise ValueError(f"Model entry at index {i} is missing 'type'")
        n_subs = int(item.get("n_subs", 10))
        use_era_ensemble = bool(item.get("use_era_ensemble", True))
        out.append(
            instantiate_model(
                name,
                item.get("params"),
                index=i,
                n_subs=n_subs,
                use_era_ensemble=use_era_ensemble,
            )
        )
    return out


def model_spec_needs_head_split(item: dict[str, Any]) -> bool:
    if item.get("input_columns") or item.get("input_group") or item.get("input_groups"):
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
                input_groups=m.get("input_groups"),
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
        neutralize_proportion=config.get("neutralize_proportion", 0.0),
        neutralize_features=config.get("neutralize_features"),
    )


def build_pipeline_or_multi(
    config: dict[str, Any],
    feature_columns: list[str] | None = None,
    feature_groups: dict[str, list[str]] | None = None,
) -> Pipeline | MultiHeadPipeline:
    """Build either a ``Pipeline`` or ``MultiHeadPipeline`` from config.

    Automatically selects ``MultiHeadPipeline`` when any model specifies
    ``input_columns``, ``input_group``, ``input_groups``, or local preprocessors.

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


def build_multi_target_from_config(
    config: dict[str, Any],
    flat: dict[str, Any],
    feature_columns: list[str] | None = None,
    feature_groups: dict[str, list[str]] | None = None,
) -> MultiTargetPipeline:
    preprocessors = build_preprocessors(config.get("preprocessors", []))
    strategy_targets = [str(flat.get("primary_target", "target"))]
    aux = flat.get("auxiliary_targets") or []
    if isinstance(aux, list):
        strategy_targets.extend(str(a) for a in aux)
    target_columns = list(dict.fromkeys(strategy_targets))
    blend_method = str(flat.get("target_blend_method", "equal"))
    if blend_method not in ("equal", "sharpe"):
        blend_method = "equal"

    def model_factory() -> BaseModel:
        if needs_multi_head_pipeline(config) or len(config.get("models", [])) > 1:
            pipeline = build_multi_head_pipeline(
                config,
                feature_columns=feature_columns,
                feature_groups=feature_groups,
            )
            return _PipelineModelAdapter(pipeline)
        models_cfg = config.get("models", [])
        if not models_cfg:
            raise ValueError("Config must have at least one model for multi-target HPO")
        spec = models_cfg[0]
        return instantiate_model(
            str(spec.get("type", "XGBoost")),
            spec.get("params"),
            index=0,
            n_subs=int(spec.get("n_subs", flat.get("n_subs", 10))),
            use_era_ensemble=bool(spec.get("use_era_ensemble", True)),
        )

    return MultiTargetPipeline(
        preprocessors=preprocessors,
        model_factory=model_factory,
        target_columns=target_columns,
        primary_target=str(flat.get("primary_target", "target")),
        blend_method=blend_method,
    )
