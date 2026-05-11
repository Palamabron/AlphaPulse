import random
from typing import Any

try:
    from ray import tune
except ImportError:
    tune = None


def _loguniform(low: float, high: float) -> float:
    return float(low * (high / low) ** random.random())


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
    if seed is not None:
        random.seed(seed)

    if phase == "phase_a":
        return {
            "scaler_type": random.choice(["StandardScaler", "RobustScaler"]),
            "use_packboost": False,
            "packboost_n_worst_eras": 3,
            "packboost_boost_weight": random.uniform(0.1, 0.3),
            "packboost_n_rounds_base": 200,
            "packboost_n_rounds_boost": 100,
            "num_models": 1,
            "model_1_type": random.choice(["XGBoost", "LightGBM"]),
            "model_2_type": random.choice(["XGBoost", "LightGBM"]),
            "model_3_type": random.choice(["XGBoost", "LightGBM"]),
            "n_subs": random.choice([8, 10]),
            "xgb_max_depth": random.choice([3, 5]),
            "xgb_learning_rate": _loguniform(3e-3, 0.05),
            "xgb_n_rounds": random.choice([200, 300, 400]),
            "xgb_early_stopping": random.choice([20, 30, 50]),
            "lgbm_num_leaves": random.choice([16, 31, 63]),
            "lgbm_learning_rate": _loguniform(5e-3, 0.05),
            "lgbm_n_rounds": random.choice([300, 500, 800]),
            "lgbm_min_child_samples": random.choice([100, 200]),
            "lgbm_early_stopping": random.choice([50, 100]),
            "packboost_model_n_worst_eras": 3,
            "packboost_model_boost_weight": random.uniform(0.2, 0.3),
            "packboost_model_n_rounds_base": 300,
            "packboost_model_n_rounds_boost": 100,
            "ensemble_method": "single",
            "stacking_meta_learner": "ridge",
            "use_neutralization": False,
            "neutralization_proportion": 0.5,
        }

    return {
        "scaler_type": random.choice(["StandardScaler", "RobustScaler"]),
        "use_packboost": random.choice([True, False]),
        "packboost_n_worst_eras": random.choice([3, 5, 7]),
        "packboost_boost_weight": random.uniform(0.1, 0.5),
        "packboost_n_rounds_base": random.choice([200, 300, 500]),
        "packboost_n_rounds_boost": random.choice([100, 150, 200]),
        "num_models": random.choice([1, 2, 3]),
        "model_1_type": random.choice(["XGBoost", "LightGBM", "Packboost"]),
        "model_2_type": random.choice(["XGBoost", "LightGBM", "Packboost"]),
        "model_3_type": random.choice(["XGBoost", "LightGBM", "Packboost"]),
        "n_subs": random.choice([5, 8, 10, 15]),
        "xgb_max_depth": random.choice([3, 5, 7]),
        "xgb_learning_rate": _loguniform(1e-3, 0.1),
        "xgb_n_rounds": random.choice([300, 500, 800]),
        "xgb_early_stopping": random.choice([30, 50, 100]),
        "lgbm_num_leaves": random.choice([16, 31, 63, 127]),
        "lgbm_learning_rate": _loguniform(5e-3, 0.05),
        "lgbm_n_rounds": random.choice([300, 500, 800, 1500]),
        "lgbm_min_child_samples": random.choice([100, 200, 500]),
        "lgbm_early_stopping": random.choice([50, 100]),
        "packboost_model_n_worst_eras": random.choice([3, 5, 7]),
        "packboost_model_boost_weight": random.uniform(0.2, 0.5),
        "packboost_model_n_rounds_base": random.choice([300, 500]),
        "packboost_model_n_rounds_boost": random.choice([100, 200]),
        "ensemble_method": random.choice(["single", "weighted", "stacking"]),
        "stacking_meta_learner": random.choice(["ridge", "xgboost"]),
        "use_neutralization": random.choice([True, False]),
        "neutralization_proportion": random.uniform(0.1, 0.8),
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
        "model_1_type": tune.choice(["XGBoost", "LightGBM", "Packboost"]),
        "model_2_type": tune.choice(["XGBoost", "LightGBM", "Packboost"]),
        "model_3_type": tune.choice(["XGBoost", "LightGBM", "Packboost"]),
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
            }
        if t == "Packboost":
            return {
                "n_worst_eras": flat.get("packboost_model_n_worst_eras", 5),
                "boost_weight": flat.get("packboost_model_boost_weight", 0.3),
                "n_rounds_base": flat.get("packboost_model_n_rounds_base", 500),
                "n_rounds_boost": flat.get("packboost_model_n_rounds_boost", 200),
            }
        return {}

    models = [
        {"type": t, "params": model_params(t, i), "n_subs": flat.get("n_subs", 10)}
        for i, t in enumerate(types)
    ]

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


def get_train_kwargs_from_flat(flat: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_rounds": flat.get("xgb_n_rounds", 500),
        "early_stopping_rounds": flat.get("xgb_early_stopping", 50),
    }
