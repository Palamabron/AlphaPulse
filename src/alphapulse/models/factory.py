from typing import Any

import optuna

from .base import BaseModel
from .catboost_model import CatBoostModel
from .diffusion_augmenter import SyntheticDataAugmenter
from .era_ensemble_model import EraEnsembleModel
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel

TREE_MODEL_TYPES = ("xgboost", "lightgbm", "catboost", "random_forest", "extra_trees")
ALL_MODEL_TYPES = TREE_MODEL_TYPES + ("ft_transformer", "mlp", "ridge")


class ModelFactory:
    def __init__(
        self, *, include_dl: bool = False, prefix: str = "model", use_gpu: bool = False
    ) -> None:
        self.include_dl = include_dl
        self.prefix = prefix
        self.use_gpu = use_gpu

    def _p(self, name: str) -> str:
        return f"{self.prefix}_{name}"

    def suggest(self, trial: optuna.Trial) -> BaseModel:
        choices = list(ALL_MODEL_TYPES) if self.include_dl else list(TREE_MODEL_TYPES)
        model_type = trial.suggest_categorical(self._p("type"), choices)

        if model_type == "xgboost":
            return self._suggest_xgboost(trial)
        elif model_type == "lightgbm":
            return self._suggest_lightgbm(trial)
        elif model_type == "catboost":
            return self._suggest_catboost(trial)
        elif model_type == "random_forest":
            return self._suggest_random_forest(trial)
        elif model_type == "extra_trees":
            return self._suggest_extra_trees(trial)
        elif model_type == "ridge":
            return self._suggest_ridge(trial)
        elif model_type == "ft_transformer":
            return self._suggest_dl(trial, "ft_transformer")
        elif model_type == "mlp":
            return self._suggest_dl(trial, "mlp")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _suggest_xgboost(self, trial: optuna.Trial) -> EraEnsembleModel:
        params = {
            "max_depth": trial.suggest_int(self._p("xgb_max_depth"), 3, 8),
            "learning_rate": trial.suggest_float(
                self._p("xgb_lr"), 5e-3, 0.1, log=True
            ),
            "min_child_weight": trial.suggest_int(
                self._p("xgb_min_child_weight"), 50, 500
            ),
            "colsample_bytree": trial.suggest_float(self._p("xgb_colsample"), 0.1, 0.5),
            "subsample": trial.suggest_float(self._p("xgb_subsample"), 0.5, 0.9),
            "reg_alpha": trial.suggest_float(
                self._p("xgb_alpha"), 1e-3, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                self._p("xgb_lambda"), 1e-3, 10.0, log=True
            ),
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }
        if self.use_gpu:
            params["device"] = "cuda"
        n_rounds = trial.suggest_int(self._p("xgb_n_rounds"), 200, 2000, step=100)
        n_subs = trial.suggest_int(self._p("n_subs"), 5, 20)
        base_name = f"XGB_{n_rounds}r"

        def factory() -> XGBoostModel:
            return XGBoostModel(params=params, name=base_name)

        return EraEnsembleModel(factory, n_subs=n_subs, name=f"EraXGB_{n_rounds}r")

    def _suggest_lightgbm(self, trial: optuna.Trial) -> EraEnsembleModel:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "max_depth": trial.suggest_int(self._p("lgb_max_depth"), 3, 8),
            "learning_rate": trial.suggest_float(
                self._p("lgb_lr"), 5e-3, 0.1, log=True
            ),
            "num_leaves": trial.suggest_int(self._p("lgb_num_leaves"), 15, 63),
            "min_child_samples": trial.suggest_int(
                self._p("lgb_min_child_samples"), 50, 500
            ),
            "colsample_bytree": trial.suggest_float(self._p("lgb_colsample"), 0.1, 0.5),
            "subsample": trial.suggest_float(self._p("lgb_subsample"), 0.5, 0.9),
            "subsample_freq": 1,
            "reg_alpha": trial.suggest_float(
                self._p("lgb_alpha"), 1e-3, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                self._p("lgb_lambda"), 1e-3, 10.0, log=True
            ),
            "verbosity": -1,
            "n_jobs": -1,
        }
        if self.use_gpu:
            params["device"] = "gpu"
            params["gpu_platform_id"] = 0
            params["gpu_device_id"] = 0
            params.pop("n_jobs", None)
        n_est = trial.suggest_int(self._p("lgb_n_estimators"), 200, 2000, step=100)
        n_subs = trial.suggest_int(self._p("n_subs"), 5, 20)
        base_name = f"LGB_{n_est}r"

        def factory() -> LightGBMModel:
            return LightGBMModel(params=params, n_estimators=n_est, name=base_name)

        return EraEnsembleModel(factory, n_subs=n_subs, name=f"EraLGB_{n_est}r")

    def _suggest_catboost(self, trial: optuna.Trial) -> EraEnsembleModel:
        params = {
            "loss_function": "RMSE",
            "depth": trial.suggest_int(self._p("cb_depth"), 4, 8),
            "learning_rate": trial.suggest_float(self._p("cb_lr"), 5e-3, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float(self._p("cb_l2"), 1e-1, 50.0, log=True),
            "min_data_in_leaf": trial.suggest_int(self._p("cb_min_data"), 50, 500),
            "random_strength": trial.suggest_float(
                self._p("cb_random_strength"), 0.1, 5.0
            ),
            "bagging_temperature": trial.suggest_float(
                self._p("cb_bagging_temp"), 0.0, 1.0
            ),
            "verbose": 0,
            "thread_count": -1,
            "allow_writing_files": False,
        }
        if self.use_gpu:
            params["task_type"] = "GPU"
        else:
            params["colsample_bylevel"] = trial.suggest_float(
                self._p("cb_colsample"), 0.1, 0.5
            )
        iters = trial.suggest_int(self._p("cb_iterations"), 200, 2000, step=100)
        n_subs = trial.suggest_int(self._p("n_subs"), 5, 20)
        base_name = f"CB_{iters}i"

        def factory() -> CatBoostModel:
            return CatBoostModel(params=params, iterations=iters, name=base_name)

        return EraEnsembleModel(factory, n_subs=n_subs, name=f"EraCB_{iters}i")

    def _suggest_random_forest(self, trial: optuna.Trial) -> EraEnsembleModel:
        from .sklearn_models import RandomForestModel

        params = {
            "n_estimators": trial.suggest_int(
                self._p("rf_n_estimators"), 100, 500, step=50
            ),
            "min_samples_leaf": trial.suggest_int(
                self._p("rf_min_samples_leaf"), 50, 500
            ),
            "max_features": trial.suggest_float(self._p("rf_max_features"), 0.1, 0.5),
            "n_jobs": -1,
            "random_state": 42,
        }
        n_subs = trial.suggest_int(self._p("n_subs"), 5, 20)
        base_name = f"RF_{params['n_estimators']}t"

        def factory() -> RandomForestModel:
            return RandomForestModel(params=params, name=base_name)

        return EraEnsembleModel(
            factory, n_subs=n_subs, name=f"EraRF_{params['n_estimators']}t"
        )

    def _suggest_extra_trees(self, trial: optuna.Trial) -> EraEnsembleModel:
        from .sklearn_models import ExtraTreesModel

        params = {
            "n_estimators": trial.suggest_int(
                self._p("et_n_estimators"), 100, 500, step=50
            ),
            "min_samples_leaf": trial.suggest_int(
                self._p("et_min_samples_leaf"), 50, 500
            ),
            "max_features": trial.suggest_float(self._p("et_max_features"), 0.1, 0.5),
            "n_jobs": -1,
            "random_state": 42,
        }
        n_subs = trial.suggest_int(self._p("n_subs"), 5, 20)
        base_name = f"ET_{params['n_estimators']}t"

        def factory() -> ExtraTreesModel:
            return ExtraTreesModel(params=params, name=base_name)

        return EraEnsembleModel(
            factory, n_subs=n_subs, name=f"EraET_{params['n_estimators']}t"
        )

    def _suggest_ridge(self, trial: optuna.Trial) -> BaseModel:
        from .sklearn_models import RidgeModel

        alpha = trial.suggest_float(self._p("ridge_alpha"), 1.0, 1000.0, log=True)
        return RidgeModel(alpha=alpha, name=f"Ridge_a{alpha:.1f}")

    def _suggest_dl(self, trial: optuna.Trial, architecture: str) -> BaseModel:
        from .tabular_dl_model import TabularDLModel

        dl_params: dict[str, Any] = {
            "learning_rate": trial.suggest_float(
                self._p("dl_lr"), 1e-4, 1e-2, log=True
            ),
            "dropout": trial.suggest_float(self._p("dl_dropout"), 0.0, 0.5),
            "weight_decay": trial.suggest_float(self._p("dl_wd"), 1e-6, 1e-2, log=True),
        }
        trainer_params: dict[str, Any] = {
            "epochs": trial.suggest_int(self._p("dl_epochs"), 20, 100, step=10),
            "batch_size": trial.suggest_categorical(
                self._p("dl_batch_size"), [512, 1024, 2048]
            ),
            "patience": 5,
        }

        if architecture == "ft_transformer":
            dl_params["embed_dim"] = trial.suggest_categorical(
                self._p("ft_embed_dim"), [16, 32, 64]
            )
            dl_params["num_heads"] = trial.suggest_categorical(
                self._p("ft_num_heads"), [2, 4, 8]
            )
            dl_params["num_attn_blocks"] = trial.suggest_int(
                self._p("ft_num_blocks"), 2, 6
            )
        else:
            dl_params["layers"] = trial.suggest_categorical(
                self._p("mlp_layers"),
                ["128-64-32", "256-128-64", "512-256-128", "64-32"],
            )

        return TabularDLModel(
            architecture=architecture,  # type: ignore[arg-type]
            dl_params=dl_params,
            trainer_params=trainer_params,
            name=f"DL_{architecture}",
        )

    def suggest_fixed(
        self, model_type: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> BaseModel:
        from ..hpo.builder import instantiate_model

        p = dict(params or {})
        n_subs = int(p.pop("n_subs", kwargs.get("n_subs", 10)))

        if model_type in ("ft_transformer", "mlp"):
            from .tabular_dl_model import TabularDLModel

            return TabularDLModel(
                architecture=model_type,  # type: ignore[arg-type]
                dl_params=p.get("dl_params"),
                trainer_params=p.get("trainer_params"),
                name=p.get("name"),
            )

        registry_name = {
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
            "catboost": "CatBoost",
            "random_forest": "RandomForest",
            "extra_trees": "ExtraTrees",
            "ridge": "Ridge",
        }.get(model_type, model_type)
        return instantiate_model(registry_name, p, index=0, n_subs=n_subs)


def suggest_augmentation(
    trial: optuna.Trial, prefix: str = "aug"
) -> SyntheticDataAugmenter | None:
    use_aug = trial.suggest_categorical(f"{prefix}_use_augmentation", [True, False])
    if not use_aug:
        return None
    n_synthetic = trial.suggest_int(f"{prefix}_n_synthetic", 100, 2000, step=100)
    top_fraction = trial.suggest_float(f"{prefix}_top_fraction", 0.05, 0.20)
    return SyntheticDataAugmenter(
        top_fraction=top_fraction, n_synthetic=n_synthetic, backend="auto", seed=42
    )
