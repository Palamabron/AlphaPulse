from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    data_dir: Path
    train_subsample: float = Field(1.0, ge=0.0, le=1.0)
    target_col: str = "target"
    seed: int = 42


class FeatureConfig(BaseModel):
    columns: list[str] | None = None
    groups: dict[str, list[str]] = Field(default_factory=dict)


class PreprocessorStep(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    input_columns: list[str] | None = None
    input_group: str | None = None
    preprocessors: list[PreprocessorStep] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    primary_metric: Literal["mean_per_era_correlation", "sharpe", "correlation"] = (
        "mean_per_era_correlation"
    )
    era_holdout_last_n: int | None = None
    walk_forward: bool = False
    walk_forward_min_train_eras: int = Field(default=1, ge=1)


class TrainConfig(BaseModel):
    n_rounds: int = 500
    early_stopping_rounds: int = 50


class ExperimentV1(BaseModel):
    version: Literal["1"] = "1"
    data: DataConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    preprocessing: list[PreprocessorStep] = Field(default_factory=list)
    models: list[ModelSpec]
    ensemble_method: Literal["single", "weighted", "stacking"] = "single"
    ensemble_params: dict[str, Any] = Field(default_factory=dict)
    train: TrainConfig = Field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def to_pipeline_config(self) -> dict[str, Any]:
        return {
            "preprocessors": [s.model_dump() for s in self.preprocessing],
            "models": [m.model_dump() for m in self.models],
            "ensemble_method": self.ensemble_method,
            "ensemble_params": dict(self.ensemble_params),
            "feature_groups": dict(self.features.groups),
        }
