"""Pure functions that apply structured mutations to a nested pipeline config dict."""

from __future__ import annotations

import copy
from typing import Any

VALID_MODELS = [
    "XGBoost",
    "LightGBM",
    "Packboost",
    "CatBoost",
    "TabPFN",
    "TabPFN3",
    "TabPFN3Reasoning",
    "TabICL",
]
VALID_PREPROCESSORS = [
    "StandardScaler",
    "RobustScaler",
    "Packboost",
    "PCA",
    "TruncatedSVD",
    "VarianceSelector",
    "LGBMImportanceSelector",
    "GaussianNoise",
]
VALID_ENSEMBLE_METHODS = ["single", "weighted", "stacking"]
MAX_MODELS = 4


def tune_model_params(
    config: dict[str, Any], model_index: int, param_updates: dict[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    models = config.get("models", [])
    if not (0 <= model_index < len(models)):
        raise ValueError(
            f"model_index {model_index} out of range (have {len(models)} models)"
        )
    existing = dict(models[model_index].get("params") or {})
    existing.update(param_updates)
    models[model_index]["params"] = existing
    config["models"] = models
    return config


def add_model(
    config: dict[str, Any], model_type: str, params: dict[str, Any]
) -> dict[str, Any]:
    if model_type not in VALID_MODELS:
        raise ValueError(
            f"Unknown model type: {model_type!r}. Must be one of {VALID_MODELS}"
        )
    config = copy.deepcopy(config)
    models = config.get("models", [])
    if len(models) >= MAX_MODELS:
        raise ValueError(f"Cannot exceed {MAX_MODELS} models in the ensemble")
    models.append({"type": model_type, "params": params})
    config["models"] = models
    if len(models) > 1 and config.get("ensemble_method") == "single":
        config["ensemble_method"] = "weighted"
        config["ensemble_params"] = {"weights": [1.0 / len(models)] * len(models)}
    return config


def remove_model(config: dict[str, Any], model_index: int) -> dict[str, Any]:
    config = copy.deepcopy(config)
    models = config.get("models", [])
    if len(models) <= 1:
        raise ValueError("Cannot remove the only model")
    if not (0 <= model_index < len(models)):
        raise ValueError(
            f"model_index {model_index} out of range (have {len(models)} models)"
        )
    models.pop(model_index)
    config["models"] = models
    if len(models) == 1:
        config["ensemble_method"] = "single"
        config["ensemble_params"] = {}
    elif config.get("ensemble_method") == "weighted":
        config["ensemble_params"] = {"weights": [1.0 / len(models)] * len(models)}
    return config


def change_ensemble(
    config: dict[str, Any], method: str, params: dict[str, Any]
) -> dict[str, Any]:
    if method not in VALID_ENSEMBLE_METHODS:
        raise ValueError(
            f"Unknown ensemble method: {method!r}. "
            f"Must be one of {VALID_ENSEMBLE_METHODS}"
        )
    config = copy.deepcopy(config)
    n = len(config.get("models", []))
    if method == "single" and n > 1:
        raise ValueError("Cannot use 'single' ensemble with multiple models")
    config["ensemble_method"] = method
    config["ensemble_params"] = params
    return config


def add_preprocessor(
    config: dict[str, Any],
    preprocessor_type: str,
    params: dict[str, Any],
    position: int,
) -> dict[str, Any]:
    if preprocessor_type not in VALID_PREPROCESSORS:
        raise ValueError(
            f"Unknown preprocessor: {preprocessor_type!r}. "
            f"Must be one of {VALID_PREPROCESSORS}"
        )
    config = copy.deepcopy(config)
    preprocessors = config.get("preprocessors", [])
    insert_at = max(0, min(position, len(preprocessors)))
    preprocessors.insert(insert_at, {"type": preprocessor_type, "params": params})
    config["preprocessors"] = preprocessors
    return config


def remove_preprocessor(config: dict[str, Any], position: int) -> dict[str, Any]:
    config = copy.deepcopy(config)
    preprocessors = config.get("preprocessors", [])
    if not (0 <= position < len(preprocessors)):
        raise ValueError(
            f"position {position} out of range "
            f"(have {len(preprocessors)} preprocessors)"
        )
    preprocessors.pop(position)
    config["preprocessors"] = preprocessors
    return config


def set_neutralization(config: dict[str, Any], proportion: float) -> dict[str, Any]:
    if not 0.0 <= proportion <= 1.0:
        raise ValueError(f"proportion must be in [0, 1], got {proportion}")
    config = copy.deepcopy(config)
    config["neutralize_proportion"] = proportion
    return config
