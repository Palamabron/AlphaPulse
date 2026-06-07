import random as _random_mod
from typing import Any

try:
    from ray import tune
except ImportError:
    tune = None

BOOSTING_MODELS = ["XGBoost", "LightGBM", "Packboost", "CatBoost"]
FOUNDATION_MODELS = ["TabPFN", "TabICL", "TabPFN3", "TabPFN3Reasoning"]
FOUNDATION_SAMPLE_PROB = 0.05


def _sample_model_type(phase: str, rng: _random_mod.Random) -> str:
    if phase != "phase_a":
        roll = rng.random()
        if roll < FOUNDATION_SAMPLE_PROB:
            return rng.choice(FOUNDATION_MODELS)
    if phase == "phase_a":
        return rng.choice(["XGBoost", "LightGBM"])
    return rng.choice(BOOSTING_MODELS)


def _loguniform(low: float, high: float, rng: _random_mod.Random) -> float:
    return float(low * (high / low) ** rng.random())


def sample_random_config(
    seed: int | None = None, *, phase: str = "phase_b"
) -> dict[str, Any]:
    """Sample a random flat HPO configuration for local search.

    Args:
        seed: Random seed for reproducibility. *None* for non-deterministic.
        phase: Search phase (``"phase_a"`` for conservative exploration,
            ``"phase_b"`` for full search space).

    Returns:
        Flat dictionary of hyperparameter values consumable by
        ``resolve_flat_config`` and ``run_trial``.
    """
    rng = _random_mod.Random(seed)

    if phase == "phase_a":
        return {
            "scaler_type": rng.choice(["StandardScaler", "RobustScaler"]),
            "use_packboost": False,
            "packboost_n_worst_eras": 3,
            "packboost_boost_weight": rng.uniform(0.1, 0.3),
            "packboost_n_rounds_base": 200,
            "packboost_n_rounds_boost": 100,
            "num_models": 1,
            "model_1_type": _sample_model_type("phase_a", rng),
            "model_2_type": _sample_model_type("phase_a", rng),
            "model_3_type": _sample_model_type("phase_a", rng),
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
            "use_neutralization": False,
            "neutralization_proportion": 0.5,
        }

    return {
        "scaler_type": rng.choice(["StandardScaler", "RobustScaler"]),
        "use_packboost": rng.choice([True, False]),
        "packboost_n_worst_eras": rng.choice([3, 5, 7]),
        "packboost_boost_weight": rng.uniform(0.1, 0.5),
        "packboost_n_rounds_base": rng.choice([200, 300, 500]),
        "packboost_n_rounds_boost": rng.choice([100, 150, 200]),
        "num_models": rng.choice([1, 2, 3]),
        "model_1_type": _sample_model_type("phase_b", rng),
        "model_2_type": _sample_model_type("phase_b", rng),
        "model_3_type": _sample_model_type("phase_b", rng),
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
        "use_neutralization": rng.choice([True, False]),
        "neutralization_proportion": rng.uniform(0.1, 0.8),
        "augmenter_top_fraction": rng.uniform(0.05, 0.20),
        "augmenter_n_synthetic": rng.choice([200, 500, 1000]),
        "augmenter_backend": "auto",
    }


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
            return {
                "params": {
                    "max_depth": flat.get("xgb_max_depth", 5),
                    "learning_rate": flat.get("xgb_learning_rate", 0.01),
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                },
            }
        if t == "LightGBM":
            return {
                "params": {
                    "num_leaves": flat.get("lgbm_num_leaves", 31),
                    "learning_rate": flat.get("lgbm_learning_rate", 0.01),
                    "min_child_samples": flat.get("lgbm_min_child_samples", 200),
                    "objective": "regression",
                    "metric": "rmse",
                    "verbosity": -1,
                },
                "n_estimators": flat.get("lgbm_n_rounds", 2000),
                "early_stopping_rounds": flat.get("lgbm_early_stopping", 100),
            }
        if t == "CatBoost":
            return {
                "params": {
                    "depth": flat.get("catboost_depth", 6),
                    "learning_rate": flat.get("catboost_learning_rate", 0.03),
                    "l2_leaf_reg": flat.get("catboost_l2_leaf_reg", 5.0),
                    "min_data_in_leaf": flat.get("catboost_min_data_in_leaf", 200),
                    "loss_function": "RMSE",
                    "verbose": 0,
                    "allow_writing_files": False,
                },
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
        if t in ("TabPFN", "TabPFN3", "TabICL", "TabPFN3Reasoning"):
            return {}
        if t == "SyntheticDataAugmenter":
            return {
                "top_fraction": flat.get("augmenter_top_fraction", 0.10),
                "n_synthetic": flat.get("augmenter_n_synthetic", 500),
                "backend": flat.get("augmenter_backend", "auto"),
            }
        return {}

    tree_models = {"XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees"}
    models = []
    for i, t in enumerate(types):
        spec: dict[str, Any] = {
            "type": t,
            "params": model_params(t, i),
            "use_era_ensemble": False,
        }
        if t in tree_models and spec.get("use_era_ensemble", True):
            spec["n_subs"] = flat.get("n_subs", 10)
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

    neutralize_proportion = (
        float(flat.get("neutralization_proportion", 0.5))
        if flat.get("use_neutralization")
        else 0.0
    )

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
