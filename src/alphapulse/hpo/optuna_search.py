from pathlib import Path
from typing import Any, Literal

import optuna
from optuna.samplers import RandomSampler, TPESampler
from optuna.trial import TrialState

from ..features.catalog import load_feature_catalog, load_target_catalog
from .feature_routing import suggest_feature_routing
from .search_space import (
    BOOSTING_MODELS,
    NEUTRALIZATION_PROPORTION_RANGE,
    _finalize_neutralization_sampling,
    available_foundation_models,
)
from .target_strategy import apply_target_strategy_to_flat, suggest_target_strategy

SamplerName = Literal["tpe", "random"]


def create_hpo_study(
    output_dir: Path,
    *,
    seed: int,
    sampler: SamplerName = "tpe",
    resume: bool = False,
) -> optuna.Study:
    storage_url = f"sqlite:///{(output_dir / 'optuna.db').resolve()}"
    optuna_sampler: TPESampler | RandomSampler
    if sampler == "tpe":
        optuna_sampler = TPESampler(
            seed=seed, multivariate=True, warn_independent_sampling=False
        )
    else:
        optuna_sampler = RandomSampler(seed=seed)
    return optuna.create_study(
        direction="maximize",
        sampler=optuna_sampler,
        storage=storage_url,
        study_name="alphapulse_hpo",
        load_if_exists=resume,
    )


def tell_trial_result(
    study: optuna.Study,
    optuna_trial: optuna.trial.Trial,
    score: float,
    *,
    failed: bool = False,
) -> None:
    if failed or score != score or score in (float("inf"), float("-inf")):
        study.tell(optuna_trial, state=TrialState.FAIL)
        return
    study.tell(optuna_trial, score)


def _model_pool(*, fast: bool) -> list[str]:
    pool = list(BOOSTING_MODELS)
    pool.extend(available_foundation_models(hpo_fast=fast))
    return list(dict.fromkeys(pool))


def _suggest_model_type(trial: optuna.Trial, param: str, *, fast: bool) -> str:
    return trial.suggest_categorical(param, _model_pool(fast=fast))


def _suggest_core_params(trial: optuna.Trial, *, fast: bool) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "scaler_type": trial.suggest_categorical(
            "scaler_type", ["StandardScaler", "RobustScaler"]
        ),
        "packboost_n_worst_eras": trial.suggest_categorical(
            "packboost_n_worst_eras", [3, 5, 7]
        ),
        "packboost_boost_weight": trial.suggest_float(
            "packboost_boost_weight", 0.1, 0.5
        ),
        "packboost_n_rounds_boost": trial.suggest_categorical(
            "packboost_n_rounds_boost", [100, 150, 200]
        ),
        "model_1_type": _suggest_model_type(trial, "model_1_type", fast=fast),
        "model_2_type": _suggest_model_type(trial, "model_2_type", fast=fast),
        "model_3_type": _suggest_model_type(trial, "model_3_type", fast=fast),
        "xgb_max_depth": trial.suggest_categorical("xgb_max_depth", [3, 5, 7]),
        "xgb_learning_rate": trial.suggest_float(
            "xgb_learning_rate", 1e-3, 0.1, log=True
        ),
        "lgbm_num_leaves": trial.suggest_categorical(
            "lgbm_num_leaves", [16, 31, 63] if fast else [16, 31, 63, 127]
        ),
        "lgbm_learning_rate": trial.suggest_float(
            "lgbm_learning_rate", 5e-3, 0.05, log=True
        ),
        "lgbm_min_child_samples": trial.suggest_categorical(
            "lgbm_min_child_samples", [100, 200, 500]
        ),
        "lgbm_reg_alpha": trial.suggest_float("lgbm_reg_alpha", 0.1, 2.0),
        "lgbm_reg_lambda": trial.suggest_float("lgbm_reg_lambda", 1.0, 10.0),
        "lgbm_colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.2, 0.5),
        "lgbm_subsample": trial.suggest_float("lgbm_subsample", 0.5, 0.8),
        "catboost_depth": trial.suggest_categorical("catboost_depth", [4, 5, 6]),
        "catboost_learning_rate": trial.suggest_float(
            "catboost_learning_rate", 0.01, 0.05, log=True
        ),
        "catboost_l2_leaf_reg": trial.suggest_float("catboost_l2_leaf_reg", 3.0, 15.0),
        "catboost_min_data_in_leaf": trial.suggest_categorical(
            "catboost_min_data_in_leaf", [100, 200, 500]
        ),
        "catboost_colsample_bylevel": trial.suggest_float(
            "catboost_colsample_bylevel", 0.2, 0.4
        ),
        "packboost_model_n_worst_eras": trial.suggest_categorical(
            "packboost_model_n_worst_eras", [3, 5, 7]
        ),
        "packboost_model_boost_weight": trial.suggest_float(
            "packboost_model_boost_weight", 0.2, 0.5
        ),
        "packboost_model_n_rounds_boost": trial.suggest_categorical(
            "packboost_model_n_rounds_boost", [100, 200]
        ),
        "stacking_meta_learner": trial.suggest_categorical(
            "stacking_meta_learner", ["ridge", "xgboost"]
        ),
        "use_neutralization": True,
        "neutralization_proportion": trial.suggest_float(
            "neutralization_proportion", *NEUTRALIZATION_PROPORTION_RANGE
        ),
        "use_meta_neutralization": trial.suggest_categorical(
            "use_meta_neutralization", [False, True]
        ),
        "meta_neutralization_proportion": trial.suggest_float(
            "meta_neutralization_proportion", 0.5, 0.75
        ),
        "augmenter_top_fraction": trial.suggest_float(
            "augmenter_top_fraction", 0.05, 0.20
        ),
        "augmenter_n_synthetic": trial.suggest_categorical(
            "augmenter_n_synthetic", [200, 500, 1000]
        ),
        "augmenter_backend": "auto",
        "use_gpu": False,
    }

    if fast:
        cfg.update(
            {
                "hpo_fast": True,
                "use_packboost": False,
                "num_models": trial.suggest_int("num_models", 1, 2),
                "n_subs": trial.suggest_categorical("n_subs", [3, 5]),
                "xgb_n_rounds": trial.suggest_categorical(
                    "xgb_n_rounds", [150, 250, 400]
                ),
                "xgb_early_stopping": trial.suggest_categorical(
                    "xgb_early_stopping", [20, 30, 50]
                ),
                "lgbm_n_rounds": trial.suggest_categorical(
                    "lgbm_n_rounds", [200, 400, 600]
                ),
                "lgbm_early_stopping": trial.suggest_categorical(
                    "lgbm_early_stopping", [30, 50]
                ),
                "packboost_n_rounds_base": trial.suggest_categorical(
                    "packboost_n_rounds_base", [150, 250, 350]
                ),
                "packboost_model_n_rounds_base": trial.suggest_categorical(
                    "packboost_model_n_rounds_base", [200, 300]
                ),
                "foundation_max_train_rows": trial.suggest_categorical(
                    "foundation_max_train_rows", [2_000, 3_000, 5_000]
                ),
                "foundation_compression": trial.suggest_categorical(
                    "foundation_compression", ["pca", "svd"]
                ),
                "foundation_n_components": trial.suggest_categorical(
                    "foundation_n_components", [64, 128, 256]
                ),
                "foundation_n_estimators": trial.suggest_categorical(
                    "foundation_n_estimators", [2, 4]
                ),
                "foundation_compression_epochs": trial.suggest_categorical(
                    "foundation_compression_epochs", [5, 10]
                ),
                "use_augmentation": trial.suggest_categorical(
                    "use_augmentation", [False, True]
                ),
                "ensemble_method": trial.suggest_categorical(
                    "ensemble_method", ["single", "weighted", "stacking"]
                ),
            }
        )
    else:
        cfg.update(
            {
                "use_packboost": trial.suggest_categorical(
                    "use_packboost", [False, True]
                ),
                "num_models": trial.suggest_int("num_models", 1, 3),
                "n_subs": trial.suggest_categorical("n_subs", [5, 8, 10, 15]),
                "packboost_n_rounds_base": trial.suggest_categorical(
                    "packboost_n_rounds_base", [200, 300, 500]
                ),
                "packboost_model_n_rounds_base": trial.suggest_categorical(
                    "packboost_model_n_rounds_base", [300, 500]
                ),
                "xgb_n_rounds": trial.suggest_categorical(
                    "xgb_n_rounds", [300, 500, 800]
                ),
                "xgb_early_stopping": trial.suggest_categorical(
                    "xgb_early_stopping", [30, 50, 100]
                ),
                "lgbm_n_rounds": trial.suggest_categorical(
                    "lgbm_n_rounds", [300, 500, 800, 1500]
                ),
                "lgbm_early_stopping": trial.suggest_categorical(
                    "lgbm_early_stopping", [50, 100]
                ),
                "use_augmentation": trial.suggest_categorical(
                    "use_augmentation", [False, True]
                ),
                "ensemble_method": trial.suggest_categorical(
                    "ensemble_method", ["single", "weighted", "stacking"]
                ),
            }
        )

    return cfg


def suggest_flat_config(
    trial: optuna.Trial,
    *,
    fast: bool = False,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = _finalize_neutralization_sampling(_suggest_core_params(trial, fast=fast))

    if data_dir is not None:
        target_catalog = load_target_catalog(data_dir)
        strategy = suggest_target_strategy(trial, target_catalog, fast=fast)
        cfg = apply_target_strategy_to_flat(cfg, strategy)

        feature_catalog = load_feature_catalog(data_dir)
        routing_fragment = suggest_feature_routing(
            trial,
            feature_catalog,
            int(cfg.get("num_models", 1)),
            fast=fast,
        )
        cfg.update(routing_fragment)
    else:
        cfg.setdefault("target_mode", "single")
        cfg.setdefault("primary_target", "target")
        cfg.setdefault("auxiliary_targets", [])
        cfg.setdefault("target_blend_method", "equal")
        cfg.setdefault("use_feature_routing", False)

    return cfg
