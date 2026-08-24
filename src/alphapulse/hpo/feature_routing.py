import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..features.catalog import SIZE_GROUPS, STAT_GROUPS, FeatureCatalog

if TYPE_CHECKING:
    import optuna

BuildPath = Literal["default", "simple", "grouped", "multihead"]

FAST_MAX_ACTIVE_GROUPS = 4
SLOW_MAX_ACTIVE_GROUPS = 6
MAX_ROUTED_FEATURES = 1000

LANE_PREPROCESSORS_FAST = ("StandardScaler", "RobustScaler", "VarianceFeatureSelector")
LANE_PREPROCESSORS_SLOW = LANE_PREPROCESSORS_FAST + ("EraStableFeatureSelector",)


@dataclass
class FeatureRoutingResult:
    build_path: BuildPath
    feature_groups: dict[str, list[str]]
    feature_columns: list[str]
    pipeline_config_patch: dict[str, Any] = field(default_factory=dict)


def _lane_steps_to_preprocessors(steps: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in steps:
        if step in ("StandardScaler", "RobustScaler"):
            out.append({"type": step, "params": {}})
        elif step == "VarianceFeatureSelector":
            out.append({"type": "VarianceSelector", "params": {"keep_fraction": 0.75}})
        elif step == "EraStableFeatureSelector":
            out.append(
                {
                    "type": "EraStableSelector",
                    "params": {"keep_fraction": 0.5, "n_estimators": 50},
                }
            )
    return out


def _biased_group_choice(
    rng: random.Random, catalog: FeatureCatalog, n: int
) -> list[str]:
    size_pool = [g for g in catalog.searchable_names if g in SIZE_GROUPS]
    stat_pool = [g for g in catalog.searchable_names if g in STAT_GROUPS]
    chosen: list[str] = []
    if size_pool and rng.random() < 0.85:
        weights = [3 if g == "medium" else (1 if g == "all" else 2) for g in size_pool]
        chosen.append(rng.choices(size_pool, weights=weights, k=1)[0])
    if stat_pool and len(chosen) < n:
        n_stat = min(n - len(chosen), rng.randint(0, n - len(chosen) + 1))
        if n_stat > 0:
            chosen.extend(rng.sample(stat_pool, min(n_stat, len(stat_pool))))
    while len(chosen) < n:
        pool = [g for g in catalog.searchable_names if g not in chosen]
        if not pool:
            break
        chosen.append(rng.choice(pool))
    return chosen[:n]


def _union_size(catalog: FeatureCatalog, groups: list[str]) -> int:
    return len(catalog.union(groups))


def _fit_groups_under_limit(
    catalog: FeatureCatalog, candidate_groups: list[str]
) -> list[str]:
    groups: list[str] = []
    for group in candidate_groups:
        next_groups = groups + [group]
        if _union_size(catalog, next_groups) <= MAX_ROUTED_FEATURES:
            groups = next_groups
    if groups:
        return groups

    searchable = sorted(
        catalog.searchable_names,
        key=lambda name: len(catalog.columns(name)),
    )
    for group in searchable:
        if _union_size(catalog, [group]) <= MAX_ROUTED_FEATURES:
            return [group]

    raise ValueError(
        f"No feature group fits the routing limit of {MAX_ROUTED_FEATURES} features"
    )


def sample_feature_routing(
    rng: random.Random,
    catalog: FeatureCatalog,
    num_models: int,
    *,
    fast: bool = False,
) -> dict[str, Any]:
    max_groups = FAST_MAX_ACTIVE_GROUPS if fast else SLOW_MAX_ACTIVE_GROUPS
    n_groups = rng.randint(1, min(max_groups, len(catalog.searchable_names)))
    sampled_groups = _biased_group_choice(rng, catalog, n_groups)
    active_groups = _fit_groups_under_limit(catalog, sampled_groups)
    routed_feature_columns = catalog.union(active_groups)

    lane_pool = LANE_PREPROCESSORS_FAST if fast else LANE_PREPROCESSORS_SLOW
    n_lanes = rng.randint(1, min(2, num_models))
    lane_steps: dict[int, list[str]] = {}
    for lane_id in range(n_lanes):
        if rng.random() < 0.5:
            lane_steps[lane_id] = []
        else:
            step = rng.choice(lane_pool)
            lane_steps[lane_id] = (
                [step] if step not in ("StandardScaler", "RobustScaler") else []
            )

    model_groups: dict[int, list[str]] = {i: [] for i in range(1, num_models + 1)}
    for group in active_groups:
        model_idx = rng.randint(1, num_models)
        model_groups[model_idx].append(group)

    for model_idx in range(1, num_models + 1):
        if not model_groups[model_idx]:
            model_groups[model_idx] = [rng.choice(active_groups)]

    flat: dict[str, Any] = {
        "use_feature_routing": True,
        "active_groups": active_groups,
        "active_groups_count": len(active_groups),
        "routed_feature_count": len(routed_feature_columns),
    }
    for model_idx in range(1, num_models + 1):
        flat[f"model_{model_idx}_groups"] = model_groups[model_idx]
        flat[f"model_{model_idx}_lane"] = rng.randint(0, n_lanes - 1)
    for lane_id, steps in lane_steps.items():
        flat[f"lane_{lane_id}_steps"] = steps
    return flat


def suggest_feature_routing(
    trial: "optuna.Trial",
    catalog: FeatureCatalog,
    num_models: int,
    *,
    fast: bool = False,
) -> dict[str, Any]:
    size_pool = [g for g in catalog.searchable_names if g in SIZE_GROUPS and g != "all"]
    candidate: list[str] = []
    if size_pool:
        size_group = trial.suggest_categorical("routing_size_group", size_pool)
        candidate.append(size_group)

    stat_pool = [g for g in catalog.searchable_names if g in STAT_GROUPS]
    for group in stat_pool:
        if trial.suggest_categorical(f"routing_use_{group}", [False, True]):
            candidate.append(group)

    if not candidate:
        candidate = _fit_groups_under_limit(catalog, [])
    active_groups = _fit_groups_under_limit(catalog, candidate)
    routed_feature_columns = catalog.union(active_groups)

    lane_pool = LANE_PREPROCESSORS_FAST if fast else LANE_PREPROCESSORS_SLOW
    n_lanes = trial.suggest_int("routing_n_lanes", 1, min(2, num_models))
    max_groups = FAST_MAX_ACTIVE_GROUPS if fast else SLOW_MAX_ACTIVE_GROUPS
    lane_steps: dict[int, list[str]] = {}
    for lane_id in range(n_lanes):
        use_lane_pp = trial.suggest_categorical(
            f"routing_lane_{lane_id}_use_pp", [False, True]
        )
        if use_lane_pp:
            step = trial.suggest_categorical(
                f"routing_lane_{lane_id}_step", list(lane_pool)
            )
            lane_steps[lane_id] = (
                [step] if step not in ("StandardScaler", "RobustScaler") else []
            )
        else:
            lane_steps[lane_id] = []

    model_groups: dict[int, list[str]] = {i: [] for i in range(1, num_models + 1)}
    for group in active_groups:
        model_idx = trial.suggest_int(f"routing_assign_{group}_model", 1, num_models)
        model_groups[model_idx].append(group)

    flat: dict[str, Any] = {
        "use_feature_routing": True,
        "active_groups": active_groups,
        "active_groups_count": len(active_groups),
        "routed_feature_count": len(routed_feature_columns),
    }
    for model_idx in range(1, num_models + 1):
        lane = trial.suggest_int(f"routing_model_{model_idx}_lane", 0, n_lanes - 1)
        fallback_idx = trial.suggest_int(
            f"routing_model_{model_idx}_fallback_idx", 0, max_groups - 1
        )
        if not model_groups[model_idx]:
            model_groups[model_idx] = [active_groups[fallback_idx % len(active_groups)]]
        flat[f"model_{model_idx}_groups"] = model_groups[model_idx]
        flat[f"model_{model_idx}_lane"] = lane
    for lane_id, steps in lane_steps.items():
        flat[f"lane_{lane_id}_steps"] = steps
    return flat


def validate_routing(
    flat: dict[str, Any],
    catalog: FeatureCatalog,
    num_models: int,
) -> None:
    if not flat.get("use_feature_routing"):
        return
    active = flat.get("active_groups") or []
    if not active:
        raise ValueError("use_feature_routing requires non-empty active_groups")
    for group in active:
        if group not in catalog.feature_sets:
            raise ValueError(f"Unknown active group: {group}")
    for model_idx in range(1, num_models + 1):
        groups = flat.get(f"model_{model_idx}_groups") or []
        if not groups:
            raise ValueError(f"model_{model_idx} has no assigned groups")
        for group in groups:
            if group not in active:
                raise ValueError(f"model_{model_idx} references inactive group {group}")


def resolve_feature_routing(
    flat: dict[str, Any],
    catalog: FeatureCatalog,
) -> FeatureRoutingResult:
    if not flat.get("use_feature_routing"):
        cols = catalog.columns("medium") if "medium" in catalog.feature_sets else []
        return FeatureRoutingResult(
            build_path="default",
            feature_groups={},
            feature_columns=cols,
            pipeline_config_patch={},
        )

    num_models = int(flat.get("num_models", 1))
    validate_routing(flat, catalog, num_models)

    active_groups: list[str] = list(flat.get("active_groups") or [])
    feature_groups = {g: catalog.columns(g) for g in active_groups}
    feature_columns = catalog.union(active_groups)
    if len(feature_columns) > MAX_ROUTED_FEATURES:
        raise ValueError(
            "Feature routing exceeds max feature limit: "
            f"{len(feature_columns)} > {MAX_ROUTED_FEATURES}"
        )

    model_group_map: dict[int, list[str]] = {}
    model_lane_map: dict[int, int] = {}
    for model_idx in range(1, num_models + 1):
        model_group_map[model_idx] = list(flat.get(f"model_{model_idx}_groups") or [])
        model_lane_map[model_idx] = int(flat.get(f"model_{model_idx}_lane", 0))

    unique_lanes = {model_lane_map[i] for i in range(1, num_models + 1)}
    single_lane_steps: list[str] = []
    if num_models == 1:
        lane_id = model_lane_map[1]
        single_lane_steps = list(flat.get(f"lane_{lane_id}_steps") or [])
    single_has_lane_preprocessors = bool(
        _lane_steps_to_preprocessors(single_lane_steps)
    )

    if num_models > 1:
        build_path: BuildPath = "multihead"
    elif len(unique_lanes) > 1 or single_has_lane_preprocessors:
        build_path = "grouped"
    else:
        build_path = "simple"

    patch: dict[str, Any] = {"feature_groups": feature_groups}

    if build_path == "simple":
        union_cols = catalog.union(model_group_map.get(1, active_groups))
        patch["models"] = [
            {
                "input_columns": union_cols,
            }
        ]
    elif build_path == "grouped":
        groups_for_model = model_group_map[1]
        lane_groups: dict[str, list[str]] = {}
        lane_pipelines: dict[str, list[dict[str, Any]]] = {}
        lane_id = model_lane_map[1]
        steps = flat.get(f"lane_{lane_id}_steps") or []
        local_steps = _lane_steps_to_preprocessors(list(steps))
        for group in groups_for_model:
            lane_groups[group] = catalog.columns(group)
            lane_pipelines[group] = list(local_steps)
        patch["preprocessors"] = [
            {
                "type": "Grouped",
                "params": {
                    "groups": lane_groups,
                    "pipelines": lane_pipelines,
                },
            }
        ]
        patch["models"] = [{}]
    else:
        models_patch: list[dict[str, Any]] = []
        for model_idx in range(1, num_models + 1):
            lane_id = model_lane_map[model_idx]
            steps = flat.get(f"lane_{lane_id}_steps") or []
            local_pp = _lane_steps_to_preprocessors(list(steps))
            models_patch.append(
                {
                    "input_groups": model_group_map[model_idx],
                    "preprocessors": local_pp,
                }
            )
        patch["models"] = models_patch

    return FeatureRoutingResult(
        build_path=build_path,
        feature_groups=feature_groups,
        feature_columns=feature_columns,
        pipeline_config_patch=patch,
    )


def merge_routing_into_pipeline_config(
    pipeline_cfg: dict[str, Any],
    routing: FeatureRoutingResult,
) -> dict[str, Any]:
    if routing.build_path == "default":
        return pipeline_cfg

    cfg = dict(pipeline_cfg)
    patch = routing.pipeline_config_patch

    if "preprocessors" in patch:
        if routing.build_path == "grouped":
            cfg["preprocessors"] = list(cfg.get("preprocessors", [])) + list(
                patch["preprocessors"]
            )
        else:
            cfg["preprocessors"] = patch["preprocessors"]

    if "feature_groups" in patch:
        cfg["feature_groups"] = patch["feature_groups"]

    models = [dict(m) for m in cfg.get("models", [])]
    patch_models = patch.get("models") or []
    for i, pm in enumerate(patch_models):
        if i >= len(models):
            break
        models[i] = {**models[i], **pm}
    cfg["models"] = models
    return cfg
