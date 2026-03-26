# SPDX-FileCopyrightText: 2025-present jakub.szulc <szulcak05@gmail.com>
#
# SPDX-License-Identifier: MIT

from .data import NumeraiDataLoader
from .evaluation import (
    Backtester,
    calculate_metrics,
    era_correlation_metrics,
    era_sharpe,
    per_era_correlation,
    per_era_spearman,
    rank_normalize,
)
from .hpo import (
    TrialResult,
    build_pipeline,
    build_pipeline_or_multi,
    ray_trainable,
    resolve_flat_config,
    run_trial,
    sample_random_config,
)
from .models import (
    BaseModel,
    CatBoostModel,
    LightGBMModel,
    ModelFactory,
    PackboostModel,
    SyntheticDataAugmenter,
    XGBoostModel,
)
from .pipeline import EnsembleOptimizer, FeatureNeutralizer, Pipeline, Stacker
from .preprocessors import (
    BasePreprocessor,
    GaussianNoiseInjector,
    LGBMImportanceSelector,
    PackboostPreprocessor,
    PreprocessorFactory,
    RobustScalerPreprocessor,
    StandardScalerPreprocessor,
    VarianceFeatureSelector,
)
from .validation import PurgedEraCV

__all__ = [
    "Backtester",
    "BaseModel",
    "BasePreprocessor",
    "CatBoostModel",
    "EnsembleOptimizer",
    "FeatureNeutralizer",
    "GaussianNoiseInjector",
    "LGBMImportanceSelector",
    "LightGBMModel",
    "ModelFactory",
    "NumeraiDataLoader",
    "PackboostModel",
    "PackboostPreprocessor",
    "Pipeline",
    "PreprocessorFactory",
    "PurgedEraCV",
    "RobustScalerPreprocessor",
    "Stacker",
    "StandardScalerPreprocessor",
    "SyntheticDataAugmenter",
    "TrialResult",
    "VarianceFeatureSelector",
    "XGBoostModel",
    "build_pipeline",
    "build_pipeline_or_multi",
    "calculate_metrics",
    "era_correlation_metrics",
    "era_sharpe",
    "per_era_correlation",
    "per_era_spearman",
    "rank_normalize",
    "ray_trainable",
    "resolve_flat_config",
    "run_trial",
    "sample_random_config",
]
