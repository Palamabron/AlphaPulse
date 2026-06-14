import random as _random_mod
from typing import Any

try:
    from ray import tune
except ImportError:
    tune = None

from ..evaluation.era_split import HPO_FAST_N_SUBS_CAP

BOOSTING_MODELS = ["XGBoost", "LightGBM", "Packboost", "CatBoost"]
FOUNDATION_MODELS = ["TabPFN", "TabICL", "TabPFN3", "TabPFN3Reasoning"]
FOUNDATION_SAMPLE_PROB = 0.05
AUGMENTER_SAMPLE_PROB = 0.05
HPO_FAST_FOUNDATION_SAMPLE_PROB = 0.03
HPO_SLOW_FOUNDATION_TYPES = ("TabPFN3", "TabPFN3Reasoning")
MIN_NEUTRALIZATION_PROPORTION = 0.15
DEFAULT_NEUTRALIZATION_PROPORTION = 0.35
NEUTRALIZATION_PROPORTION_RANGE = (MIN_NEUTRALIZATION_PROPORTION, 0.8)


def uses_neutralization_for_models(model_types: list[str]) -> bool:
    return any(t not in FOUNDATION_MODELS for t in model_types)


def _sampled_model_types(cfg: dict[str, Any]) -> list[str]:
    num_models = int(cfg.get("num_models", 1))
    return [
        str(cfg.get("model_1_type", "XGBoost")),
        str(cfg.get("model_2_type", "XGBoost")),
        str(cfg.get("model_3_type", "XGBoost")),
    ][:num_models]


def _finalize_neutralization_sampling(cfg: dict[str, Any]) -> dict[str, Any]:
    types = _sampled_model_types(cfg)
    if uses_neutralization_for_models(types):
        cfg["use_neutralization"] = True
        proportion = float(
            cfg.get("neutralization_proportion", DEFAULT_NEUTRALIZATION_PROPORTION)
        )
        cfg["neutralization_proportion"] = max(
            MIN_NEUTRALIZATION_PROPORTION, proportion
        )
    else:
        cfg["use_neutralization"] = False
        cfg["neutralization_proportion"] = 0.0
    return cfg


def resolve_neutralize_proportion(
    flat: dict[str, Any], model_types: list[str]
) -> float:
    if not uses_neutralization_for_models(model_types):
        return 0.0
    if not flat.get("use_neutralization", True):
        return 0.0
    proportion = float(
        flat.get("neutralization_proportion", DEFAULT_NEUTRALIZATION_PROPORTION)
    )
    return max(MIN_NEUTRALIZATION_PROPORTION, min(1.0, proportion))


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_foundation_compression(
    compression: str | None, *, hpo_fast: bool = False
) -> str | None:
    if compression is None and hpo_fast:
        compression = "autoencoder"
    if compression == "autoencoder" and not _torch_available():
        return "pca"
    return compression


def available_foundation_models(*, hpo_fast: bool = False) -> list[str]:
    available: list[str] = []
    try:
        import tabpfn  # noqa: F401

        available.extend(["TabPFN", "TabPFN3"])
    except ImportError:
        pass
    try:
        import tabicl  # noqa: F401

        available.append("TabICL")
    except ImportError:
        pass
    import os

    if os.environ.get("TABPFN_API_KEY"):
        try:
            import tabpfn_client  # noqa: F401

            available.append("TabPFN3Reasoning")
        except ImportError:
            pass
    if hpo_fast:
        return [m for m in available if m not in HPO_SLOW_FOUNDATION_TYPES]
    return available


def apply_gpu_model_params(model_type: str, params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    if model_type == "XGBoost":
        inner = out.get("params")
        if isinstance(inner, dict):
            inner = dict(inner)
            inner["device"] = "cuda"
            out["params"] = inner
        else:
            out["device"] = "cuda"
    elif model_type == "LightGBM":
        inner = out.get("params")
        if isinstance(inner, dict):
            inner = dict(inner)
            inner["device"] = "gpu"
            inner["gpu_platform_id"] = 0
            inner["gpu_device_id"] = 0
            inner.pop("n_jobs", None)
            out["params"] = inner
    elif model_type == "CatBoost":
        inner = out.get("params")
        if isinstance(inner, dict):
            inner = dict(inner)
            inner["task_type"] = "GPU"
            inner.pop("colsample_bylevel", None)
            out["params"] = inner
        else:
            out["task_type"] = "GPU"
            out.pop("colsample_bylevel", None)
    return out


def strip_catboost_gpu_incompatible_params(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    inner = out.get("params")
    if isinstance(inner, dict) and inner.get("task_type") == "GPU":
        inner = dict(inner)
        inner.pop("colsample_bylevel", None)
        out["params"] = inner
    return out


def apply_gpu_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    models = []
    for item in cfg.get("models", []):
        model_type = item.get("type", "")
        params = dict(item.get("params") or {})
        models.append(
            {
                **item,
                "params": apply_gpu_model_params(model_type, params),
            }
        )
    cfg["models"] = models
    return cfg


def _sample_model_type(
    phase: str, rng: _random_mod.Random, *, fast: bool = False
) -> str:
    if phase != "phase_a":
        roll = rng.random()
        foundation_prob = (
            HPO_FAST_FOUNDATION_SAMPLE_PROB if fast else FOUNDATION_SAMPLE_PROB
        )
        if roll < foundation_prob:
            foundation_models = available_foundation_models(hpo_fast=fast)
            if foundation_models:
                return rng.choice(foundation_models)
    if phase == "phase_a":
        return rng.choice(["XGBoost", "LightGBM"])
    return rng.choice(BOOSTING_MODELS)


def _loguniform(low: float, high: float, rng: _random_mod.Random) -> float:
    return float(low * (high / low) ** rng.random())


def sample_random_config(
    seed: int | None = None, *, phase: str = "phase_b", fast: bool = False
) -> dict[str, Any]:
    """Sample a random flat HPO configuration for local search.

    Args:
        seed: Random seed for reproducibility. *None* for non-deterministic.
        phase: Search phase (``"phase_a"`` for conservative exploration,
            ``"phase_b"`` for full search space).
        fast: When True, sample tighter hyperparameters for sub-30-minute trials
            on full data (lower rounds, fewer era subs, smaller foundation caps).

    Returns:
        Flat dictionary of hyperparameter values consumable by
        ``resolve_flat_config`` and ``run_trial``.
    """
    rng = _random_mod.Random(seed)

    if phase == "phase_a":
        cfg = {
            "scaler_type": rng.choice(["StandardScaler", "RobustScaler"]),
            "use_packboost": False,
            "packboost_n_worst_eras": 3,
            "packboost_boost_weight": rng.uniform(0.1, 0.3),
            "packboost_n_rounds_base": 200,
            "packboost_n_rounds_boost": 100,
            "num_models": 1,
            "model_1_type": _sample_model_type("phase_a", rng, fast=fast),
            "model_2_type": _sample_model_type("phase_a", rng, fast=fast),
            "model_3_type": _sample_model_type("phase_a", rng, fast=fast),
            "n_subs": rng.choice([8, 10]),
            "xgb_max_depth": rng.choice([3, 5]),
            "xgb_learning_rate": _loguniform(3e-3, 0.05, rng),
            "xgb_n_rounds": rng.choice([200, 300, 400]),
            "xgb_early_stopping": rng.choice([20, 30, 50]),
            "lgbm_num_leaves": rng.choice([16, 31, 63]),
            "lgbm_learning_rate": _loguniform(5e-3, 0.05, rng),
            "lgbm_n_rounds": rng.choice([300, 500, 800]),
            "lgbm_min_child_samples": rng.choice([100, 200]),
            "lgbm_early_stopping": rng.choice([50, 100]),
            "packboost_model_n_worst_eras": 3,
            "packboost_model_boost_weight": rng.uniform(0.2, 0.3),
            "packboost_model_n_rounds_base": 300,
            "packboost_model_n_rounds_boost": 100,
            "ensemble_method": "single",
            "stacking_meta_learner": "ridge",
            "use_neutralization": True,
            "neutralization_proportion": rng.uniform(*NEUTRALIZATION_PROPORTION_RANGE),
        }
        if fast:
            cfg["hpo_fast"] = True
            cfg["n_subs"] = rng.choice([3, 5])
            cfg["xgb_n_rounds"] = rng.choice([150, 250, 350])
            cfg["lgbm_n_rounds"] = rng.choice([200, 400, 600])
        return _finalize_neutralization_sampling(cfg)

    cfg = {
        "scaler_type": rng.choice(["StandardScaler", "RobustScaler"]),
        "use_packboost": rng.choice([True, False]),
        "packboost_n_worst_eras": rng.choice([3, 5, 7]),
        "packboost_boost_weight": rng.uniform(0.1, 0.5),
        "packboost_n_rounds_base": rng.choice([200, 300, 500]),
        "packboost_n_rounds_boost": rng.choice([100, 150, 200]),
        "num_models": rng.choice([1, 2, 3]),
        "model_1_type": _sample_model_type("phase_b", rng, fast=fast),
        "model_2_type": _sample_model_type("phase_b", rng, fast=fast),
        "model_3_type": _sample_model_type("phase_b", rng, fast=fast),
        "n_subs": rng.choice([5, 8, 10, 15]),
        "xgb_max_depth": rng.choice([3, 5, 7]),
        "xgb_learning_rate": _loguniform(1e-3, 0.1, rng),
        "xgb_n_rounds": rng.choice([300, 500, 800]),
        "xgb_early_stopping": rng.choice([30, 50, 100]),
        "lgbm_num_leaves": rng.choice([16, 31, 63, 127]),
        "lgbm_learning_rate": _loguniform(5e-3, 0.05, rng),
        "lgbm_n_rounds": rng.choice([300, 500, 800, 1500]),
        "lgbm_min_child_samples": rng.choice([100, 200, 500]),
        "lgbm_early_stopping": rng.choice([50, 100]),
        "packboost_model_n_worst_eras": rng.choice([3, 5, 7]),
        "packboost_model_boost_weight": rng.uniform(0.2, 0.5),
        "packboost_model_n_rounds_base": rng.choice([300, 500]),
        "packboost_model_n_rounds_boost": rng.choice([100, 200]),
        "ensemble_method": rng.choice(["single", "weighted", "stacking"]),
        "stacking_meta_learner": rng.choice(["ridge", "xgboost"]),
        "use_neutralization": True,
        "neutralization_proportion": rng.uniform(*NEUTRALIZATION_PROPORTION_RANGE),
        "augmenter_top_fraction": rng.uniform(0.05, 0.20),
        "augmenter_n_synthetic": rng.choice([200, 500, 1000]),
        "augmenter_backend": "auto",
        "use_augmentation": rng.random() < 0.05,
        "use_gpu": False,
    }
    if fast:
        cfg["hpo_fast"] = True
        cfg["num_models"] = rng.choice([1, 2])
        cfg["n_subs"] = rng.choice([3, 5])
        cfg["xgb_n_rounds"] = rng.choice([150, 250, 400])
        cfg["xgb_early_stopping"] = rng.choice([20, 30, 50])
        cfg["lgbm_n_rounds"] = rng.choice([200, 400, 600])
        cfg["lgbm_early_stopping"] = rng.choice([30, 50])
        cfg["packboost_n_rounds_base"] = rng.choice([150, 250, 350])
        cfg["packboost_model_n_rounds_base"] = rng.choice([200, 300])
        cfg["foundation_max_train_rows"] = rng.choice([3_000, 5_000, 8_000])
        cfg["foundation_compression"] = rng.choice(["autoencoder", "pca", "svd"])
        cfg["foundation_n_components"] = rng.choice([64, 128, 256])
        cfg["foundation_n_estimators"] = rng.choice([2, 4])
        cfg["foundation_compression_epochs"] = rng.choice([5, 10])
        cfg["use_packboost"] = False
        cfg["use_augmentation"] = rng.random() < 0.03
    return _finalize_neutralization_sampling(cfg)


def get_full_param_space() -> dict[str, Any]:
    if tune is None:
        raise ImportError(
            "ray[tune] is required. Install with: pip install 'ray[tune]'"
        )

    return {
        "scaler_type": tune.choice(["StandardScaler", "RobustScaler"]),
        "use_packboost": tune.choice([True, False]),
        "packboost_n_worst_eras": tune.choice([3, 5, 7]),
        "packboost_boost_weight": tune.uniform(0.1, 0.5),
        "packboost_n_rounds_base": tune.choice([200, 300, 500]),
        "packboost_n_rounds_boost": tune.choice([100, 150, 200]),
        "num_models": tune.choice([1, 2, 3]),
        "model_1_type": tune.choice(BOOSTING_MODELS + FOUNDATION_MODELS),
        "model_2_type": tune.choice(BOOSTING_MODELS + FOUNDATION_MODELS),
        "model_3_type": tune.choice(BOOSTING_MODELS + FOUNDATION_MODELS),
        "n_subs": tune.choice([5, 8, 10, 15]),
        "xgb_max_depth": tune.choice([3, 5, 7]),
        "xgb_learning_rate": tune.loguniform(1e-3, 0.1),
        "xgb_n_rounds": tune.choice([300, 500, 800]),
        "xgb_early_stopping": tune.choice([30, 50, 100]),
        "lgbm_num_leaves": tune.choice([16, 31, 63, 127]),
        "lgbm_learning_rate": tune.loguniform(5e-3, 0.05),
        "lgbm_n_rounds": tune.choice([300, 500, 800, 1500]),
        "lgbm_min_child_samples": tune.choice([100, 200, 500]),
        "lgbm_early_stopping": tune.choice([50, 100]),
        "packboost_model_n_worst_eras": tune.choice([3, 5, 7]),
        "packboost_model_boost_weight": tune.uniform(0.2, 0.5),
        "packboost_model_n_rounds_base": tune.choice([300, 500]),
        "packboost_model_n_rounds_boost": tune.choice([100, 200]),
        "ensemble_method": tune.choice(["single", "weighted", "stacking"]),
        "stacking_meta_learner": tune.choice(["ridge", "xgboost"]),
        "use_neutralization": tune.choice([True, False]),
        "neutralization_proportion": tune.uniform(0.1, 0.8),
        "foundation_max_train_rows": tune.choice([5_000, 10_000, 20_000]),
        "foundation_compression": tune.choice(["pca", "svd"]),
        "foundation_n_components": tune.choice([128, 256, 512]),
    }


def resolve_flat_config(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat HPO parameter dict into a nested pipeline config.

    Args:
        flat: Flat dictionary of hyperparameters (e.g. from
            ``sample_random_config`` or Ray Tune).

    Returns:
        Nested dictionary with keys ``preprocessors``, ``models``,
        ``ensemble_method``, ``ensemble_params``, and
        ``neutralize_proportion`` ready for ``build_pipeline_or_multi``.
    """
    num_models = flat.get("num_models", 1)
    types = [
        flat.get("model_1_type", "XGBoost"),
        flat.get("model_2_type", "XGBoost"),
        flat.get("model_3_type", "XGBoost"),
    ][:num_models]
    types = [t for t in types if t != "SyntheticDataAugmenter"]
    if not types:
        types = ["XGBoost"]

    preprocessors: list[dict[str, Any]] = [
        {"type": flat.get("scaler_type", "StandardScaler"), "params": {}}
    ]
    if flat.get("use_packboost"):
        preprocessors.append(
            {
                "type": "Packboost",
                "params": {
                    "n_worst_eras": flat.get("packboost_n_worst_eras", 5),
                    "boost_weight": flat.get("packboost_boost_weight", 0.3),
                    "n_rounds_base": flat.get("packboost_n_rounds_base", 300),
                    "n_rounds_boost": flat.get("packboost_n_rounds_boost", 100),
                },
            }
        )

    def model_params(t: str, index: int) -> dict[str, Any]:
        if t == "XGBoost":
            xgb_params = {
                "max_depth": flat.get("xgb_max_depth", 5),
                "learning_rate": flat.get("xgb_learning_rate", 0.01),
                "tree_method": "hist",
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
            }
            if flat.get("use_gpu"):
                xgb_params["device"] = "cuda"
            return {"params": xgb_params}
        if t == "LightGBM":
            lgb_params: dict[str, Any] = {
                "num_leaves": flat.get("lgbm_num_leaves", 31),
                "learning_rate": flat.get("lgbm_learning_rate", 0.01),
                "min_child_samples": flat.get("lgbm_min_child_samples", 200),
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
            }
            if flat.get("use_gpu"):
                lgb_params["device"] = "gpu"
                lgb_params["gpu_platform_id"] = 0
                lgb_params["gpu_device_id"] = 0
            return {
                "params": lgb_params,
                "n_estimators": flat.get("lgbm_n_rounds", 2000),
                "early_stopping_rounds": flat.get("lgbm_early_stopping", 100),
            }
        if t == "CatBoost":
            cb_params = {
                "depth": flat.get("catboost_depth", 6),
                "learning_rate": flat.get("catboost_learning_rate", 0.03),
                "l2_leaf_reg": flat.get("catboost_l2_leaf_reg", 5.0),
                "min_data_in_leaf": flat.get("catboost_min_data_in_leaf", 200),
                "loss_function": "RMSE",
                "verbose": 0,
                "allow_writing_files": False,
            }
            if flat.get("use_gpu"):
                cb_params["task_type"] = "GPU"
            else:
                cb_params["colsample_bylevel"] = flat.get(
                    "catboost_colsample_bylevel", 0.3
                )
            return {
                "params": cb_params,
                "iterations": flat.get("catboost_iterations", 2000),
                "early_stopping_rounds": flat.get("catboost_early_stopping", 100),
            }
        if t == "RandomForest":
            return {
                "params": {
                    "n_estimators": flat.get("rf_n_estimators", 300),
                    "min_samples_leaf": flat.get("rf_min_samples_leaf", 200),
                    "max_features": flat.get("rf_max_features", 0.3),
                    "n_jobs": -1,
                    "random_state": 42,
                },
            }
        if t == "ExtraTrees":
            return {
                "params": {
                    "n_estimators": flat.get("et_n_estimators", 300),
                    "min_samples_leaf": flat.get("et_min_samples_leaf", 200),
                    "max_features": flat.get("et_max_features", 0.3),
                    "n_jobs": -1,
                    "random_state": 42,
                },
            }
        if t == "Ridge":
            return {"alpha": flat.get("ridge_alpha", 100.0)}
        if t == "Packboost":
            return {
                "n_worst_eras": flat.get("packboost_model_n_worst_eras", 5),
                "boost_weight": flat.get("packboost_model_boost_weight", 0.3),
                "n_rounds_base": flat.get("packboost_model_n_rounds_base", 500),
                "n_rounds_boost": flat.get("packboost_model_n_rounds_boost", 200),
            }
        if t in FOUNDATION_MODELS:
            key_map = {
                "foundation_max_train_rows": "max_train_rows",
                "foundation_compression": "compression",
                "foundation_n_components": "compression_components",
                "foundation_n_estimators": "n_estimators",
                "foundation_compression_epochs": "compression_epochs",
            }
            params = {param: flat[key] for key, param in key_map.items() if key in flat}
            compression = resolve_foundation_compression(
                params.get("compression"), hpo_fast=bool(flat.get("hpo_fast"))
            )
            if compression is not None:
                params["compression"] = compression
            if flat.get("use_gpu"):
                params["device"] = "cuda"
                params["compression_device"] = "cuda"
            return params
        return {}

    tree_models = {"XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees"}
    n_subs_cap = HPO_FAST_N_SUBS_CAP if flat.get("hpo_fast") else None
    models = []
    for i, t in enumerate(types):
        spec: dict[str, Any] = {"type": t, "params": model_params(t, i)}
        if t in tree_models:
            n_subs = int(flat.get("n_subs", 10))
            if n_subs_cap is not None:
                n_subs = min(n_subs, n_subs_cap)
            spec["n_subs"] = n_subs
        models.append(spec)

    ensemble_method = flat.get("ensemble_method", "single")
    if num_models == 1:
        ensemble_method = "single"
    ensemble_params: dict[str, Any] = {}
    if ensemble_method == "weighted" and num_models > 1:
        ensemble_params["weights"] = [1.0 / num_models] * num_models
    if ensemble_method == "stacking" and num_models > 1:
        ensemble_params["meta_learner"] = flat.get("stacking_meta_learner", "ridge")
        ensemble_params["meta_params"] = {}

    neutralize_proportion = resolve_neutralize_proportion(flat, types)

    return {
        "preprocessors": preprocessors,
        "models": models,
        "ensemble_method": ensemble_method,
        "ensemble_params": ensemble_params,
        "neutralize_proportion": neutralize_proportion,
    }


_MODEL_TRAIN_KWARGS: dict[str, tuple[str, str, int, int]] = {
    "XGBoost": ("xgb_n_rounds", "xgb_early_stopping", 500, 50),
    "LightGBM": ("lgbm_n_rounds", "lgbm_early_stopping", 2000, 100),
    "CatBoost": ("catboost_iterations", "catboost_early_stopping", 2000, 100),
    "Packboost": ("packboost_model_n_rounds_base", "xgb_early_stopping", 500, 50),
    "RandomForest": ("rf_n_estimators", "xgb_early_stopping", 300, 50),
    "ExtraTrees": ("et_n_estimators", "xgb_early_stopping", 300, 50),
}


def get_train_kwargs_from_flat(flat: dict[str, Any]) -> dict[str, Any]:
    num_models = flat.get("num_models", 1)
    model_type = flat.get("model_1_type", "XGBoost")
    if num_models > 1:
        for key in ("model_1_type", "model_2_type", "model_3_type"):
            candidate = flat.get(key)
            if candidate in _MODEL_TRAIN_KWARGS:
                model_type = candidate
                break

    rounds_key, es_key, default_rounds, default_es = _MODEL_TRAIN_KWARGS.get(
        model_type,
        ("xgb_n_rounds", "xgb_early_stopping", 500, 50),
    )
    return {
        "n_rounds": flat.get(rounds_key, default_rounds),
        "early_stopping_rounds": flat.get(es_key, default_es),
    }
