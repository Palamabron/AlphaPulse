from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

_TOP_LEVEL_MODEL_KEYS = frozenset(
    {
        "params",
        "name",
        "n_estimators",
        "iterations",
        "n_subs",
        "alpha",
        "era_column",
        "n_worst_eras",
        "boost_weight",
        "n_rounds_base",
        "early_stopping_rounds_base",
        "n_rounds_boost",
        "early_stopping_rounds_boost",
        "base_params",
        "boost_params",
        "model_path",
        "ignore_pretraining_limits",
        "random_state",
        "thinking_mode",
        "thinking_effort",
        "thinking_timeout_s",
        "thinking_metric",
        "kv_cache",
        "batch_size",
        "dl_params",
        "trainer_params",
        "architecture",
    }
)


class DataConfig(BaseModel):
    data_dir: Path
    train_subsample: float = Field(1.0, gt=0.0, le=1.0)
    target_col: str = "target"
    seed: int = 42
    auxiliary_targets: list[str] | None = None
    benchmark_columns: list[str] = Field(default_factory=list)


class FeatureConfig(BaseModel):
    columns: list[str] | None = None
    groups: dict[str, list[str]] = Field(default_factory=dict)
    use_numerai_groups: bool = False


class PreprocessorStep(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    input_columns: list[str] | None = None
    input_group: str | None = None
    preprocessors: list[PreprocessorStep] = Field(default_factory=list)
    n_subs: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def normalize_model_params(self) -> Self:
        if not self.params:
            return self
        stray = {k: v for k, v in self.params.items() if k not in _TOP_LEVEL_MODEL_KEYS}
        if stray and "params" not in self.params:
            kept = {k: v for k, v in self.params.items() if k in _TOP_LEVEL_MODEL_KEYS}
            kept["params"] = stray
            object.__setattr__(self, "params", kept)
        return self


class NeutralizationConfig(BaseModel):
    """Feature neutralization applied to model predictions after ensembling.

    Removes exposure to specified features (or all features when *features* is
    None), reducing factor-driven variance. The neutralized predictions are
    rank-normalized back to [0, 1] before scoring or submission.
    """

    proportion: float = Field(0.0, ge=0.0, le=1.0)
    features: list[str] | None = None


class EvaluationConfig(BaseModel):
    primary_metric: Literal[
        "mean_per_era_correlation",
        "sharpe",
        "correlation",
        "corr_sharpe",
        "payout_score",
        "mmc_sharpe",
    ] = "corr_sharpe"
    era_holdout_last_n: int | None = None
    walk_forward: bool = False
    walk_forward_min_train_eras: int = Field(default=1, ge=1)
    walk_forward_n_purge: int = Field(default=4, ge=0)
    walk_forward_n_embargo: int = Field(default=0, ge=0)
    walk_forward_n_splits: Annotated[int, Field(ge=2)] | None = None
    meta_model_path: str | None = None
    corr_weight: float = Field(default=0.75, ge=0.0)
    mmc_weight: float = Field(default=2.25, ge=0.0)


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
    neutralization: NeutralizationConfig = Field(default_factory=NeutralizationConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def validate_feature_routing(self) -> Self:
        defined_groups = set(self.features.groups.keys())
        for i, model in enumerate(self.models):
            group = model.input_group
            if group is not None and group not in defined_groups:
                if defined_groups:
                    available = ", ".join(sorted(defined_groups))
                    raise ValueError(
                        f"models[{i}] (type={model.type!r}) references"
                        f" input_group={group!r}, which is not defined in"
                        f" features.groups. Available groups: {available}"
                    )
                else:
                    raise ValueError(
                        f"models[{i}] (type={model.type!r}) references"
                        f" input_group={group!r}, but features.groups is"
                        f" empty. Define the group in your YAML."
                    )
            if group is not None and model.input_columns is not None:
                raise ValueError(
                    f"models[{i}] (type={model.type!r}) sets both"
                    f" input_group and input_columns — use one, not both."
                )
        return self

    def to_pipeline_config(self) -> dict[str, Any]:
        return {
            "preprocessors": [s.model_dump() for s in self.preprocessing],
            "models": [m.model_dump() for m in self.models],
            "ensemble_method": self.ensemble_method,
            "ensemble_params": dict(self.ensemble_params),
            "feature_groups": dict(self.features.groups),
            "neutralize_proportion": self.neutralization.proportion,
            "neutralize_features": self.neutralization.features,
        }
