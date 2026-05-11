from typing import Any

from ..models.base import BaseModel
from ..models.catboost_model import CatBoostModel
from ..models.foundation_models import TabICLModel, TabPFNModel
from ..models.lightgbm_model import LightGBMModel
from ..models.packboost_model import PackboostModel
from ..models.sklearn_models import ExtraTreesModel, RandomForestModel, RidgeModel
from ..models.xgboost_model import XGBoostModel
from ..preprocessors.base import BasePreprocessor
from ..preprocessors.compression import PCAPreprocessor, TruncatedSVDPreprocessor
from ..preprocessors.feature_selection import (
    LGBMImportanceSelector,
    VarianceFeatureSelector,
)
from ..preprocessors.noise import GaussianNoiseInjector
from ..preprocessors.packboost import PackboostPreprocessor
from ..preprocessors.scaling import RobustScalerPreprocessor, StandardScalerPreprocessor

MODEL_REGISTRY: dict[str, tuple[type[BaseModel], dict[str, Any]]] = {
    "XGBoost": (
        XGBoostModel,
        {
            "params": {
                "max_depth": 5,
                "learning_rate": 0.01,
                "tree_method": "hist",
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
            },
        },
    ),
    "Packboost": (
        PackboostModel,
        {
            "era_column": "era",
            "n_worst_eras": 5,
            "boost_weight": 0.3,
            "n_rounds_base": 500,
            "early_stopping_rounds_base": 50,
            "n_rounds_boost": 200,
            "early_stopping_rounds_boost": 30,
        },
    ),
    "LightGBM": (
        LightGBMModel,
        {
            "params": {
                "objective": "regression",
                "metric": "rmse",
                "max_depth": 5,
                "learning_rate": 0.01,
                "num_leaves": 31,
                "min_child_samples": 200,
                "colsample_bytree": 0.3,
                "subsample": 0.7,
                "verbosity": -1,
            },
            "n_estimators": 2000,
        },
    ),
    "CatBoost": (
        CatBoostModel,
        {
            "params": {
                "loss_function": "RMSE",
                "depth": 6,
                "learning_rate": 0.03,
                "l2_leaf_reg": 5.0,
                "min_data_in_leaf": 200,
                "colsample_bylevel": 0.3,
                "verbose": 0,
                "allow_writing_files": False,
            },
            "iterations": 2000,
        },
    ),
    "RandomForest": (
        RandomForestModel,
        {
            "params": {
                "n_estimators": 300,
                "min_samples_leaf": 200,
                "max_features": 0.3,
                "n_jobs": -1,
                "random_state": 42,
            },
        },
    ),
    "ExtraTrees": (
        ExtraTreesModel,
        {
            "params": {
                "n_estimators": 300,
                "min_samples_leaf": 200,
                "max_features": 0.3,
                "n_jobs": -1,
                "random_state": 42,
            },
        },
    ),
    "Ridge": (
        RidgeModel,
        {
            "alpha": 100.0,
        },
    ),
    "TabPFN": (
        TabPFNModel,
        {
            "n_estimators": 8,
            "ignore_pretraining_limits": False,
        },
    ),
    "TabICL": (
        TabICLModel,
        {
            "n_estimators": 8,
            "kv_cache": False,
            "batch_size": 8,
            "random_state": 42,
        },
    ),
}

PREPROCESSOR_REGISTRY: dict[str, tuple[type[BasePreprocessor], dict[str, Any]]] = {
    "StandardScaler": (StandardScalerPreprocessor, {}),
    "RobustScaler": (RobustScalerPreprocessor, {}),
    "Packboost": (
        PackboostPreprocessor,
        {
            "era_column": "era",
            "output_column": "packboost_pred",
            "n_worst_eras": 5,
            "boost_weight": 0.3,
            "n_rounds_base": 300,
            "n_rounds_boost": 100,
        },
    ),
    "PCA": (
        PCAPreprocessor,
        {"n_components": None},
    ),
    "TruncatedSVD": (
        TruncatedSVDPreprocessor,
        {"n_components": 10},
    ),
    "VarianceSelector": (
        VarianceFeatureSelector,
        {"keep_fraction": 0.75, "mode": "quantile"},
    ),
    "LGBMImportanceSelector": (
        LGBMImportanceSelector,
        {"keep_fraction": 0.75, "n_estimators": 100},
    ),
    "GaussianNoise": (
        GaussianNoiseInjector,
        {"sigma": 0.01},
    ),
}
