from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alphapulse.evaluation.backtester import Backtester
from alphapulse.evaluation.wandb_diagnostics import log_experiment_diagnostics
from alphapulse.experiments.data import META_MODEL_COLUMN, meta_model_from_benchmarks
from alphapulse.hpo.builder import (
    build_multi_head_pipeline,
    build_multi_target_from_config,
)
from alphapulse.models.base import BaseModel
from alphapulse.pipeline.multi_target import MultiTargetPipeline
from alphapulse.pipeline.multihead import HeadSpec, MultiHeadPipeline
from alphapulse.pipeline.neutralizer import (
    MetaModelNeutralizer,
    neutralize_against_meta,
)
from alphapulse.pipeline.pipeline import Pipeline
from alphapulse.preprocessors.base import BasePreprocessor


class _IdentityPreprocessor(BasePreprocessor):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> BasePreprocessor:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class _EraRequiredPreprocessor(BasePreprocessor):
    def __init__(self) -> None:
        super().__init__("era_required")
        self.saw_era = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> BasePreprocessor:
        self.saw_era = "era" in X.columns and X["era"].notna().all()
        if not self.saw_era:
            raise ValueError("era is required")
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class _ConstModel(BaseModel):
    def __init__(self, value: float, name: str = "m") -> None:
        self.value = value
        self.name = name
        self.is_trained = True

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=np.float64)

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> BaseModel:
        return self


def test_neutralize_against_meta_reduces_correlation() -> None:
    rng = np.random.default_rng(0)
    n = 200
    meta = rng.standard_normal(n)
    preds = 0.9 * meta + 0.1 * rng.standard_normal(n)
    eras = pd.Series(np.repeat([f"era_{i}" for i in range(10)], n // 10))
    neutral = neutralize_against_meta(preds, meta, eras=eras, proportion=1.0)
    before = np.corrcoef(preds, meta)[0, 1]
    after = np.corrcoef(neutral, meta)[0, 1]
    assert abs(after) < abs(before)


def test_pipeline_meta_neutralization_in_predict() -> None:
    rng = np.random.default_rng(1)
    n = 80
    X = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    meta = rng.standard_normal(n)
    pipeline = Pipeline(
        preprocessors=[_IdentityPreprocessor()],
        models=[_ConstModel(0.5)],
        feature_columns=list(X.columns),
        meta_neutralize_proportion=0.8,
    )
    preds = pipeline.predict(X, meta_model=meta)
    assert preds.shape == (n,)
    assert np.all(np.isfinite(preds))


def test_pipeline_meta_neutralization_without_eras_reduces_exposure() -> None:
    rng = np.random.default_rng(11)
    n = 200
    meta = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "f1": meta + 0.05 * rng.standard_normal(n),
            "f2": rng.standard_normal(n),
        }
    )
    pipeline = Pipeline(
        preprocessors=[],
        models=[_FeatureModel()],
        feature_columns=list(X.columns),
        meta_neutralize_proportion=1.0,
    )

    raw = _FeatureModel().predict(X)
    neutralized = pipeline.predict(X, meta_model=meta)

    assert abs(np.corrcoef(neutralized, meta)[0, 1]) < abs(np.corrcoef(raw, meta)[0, 1])


def test_meta_neutralizer_optimize_proportion() -> None:
    rng = np.random.default_rng(2)
    n = 120
    eras = pd.Series(np.repeat([f"era_{i}" for i in range(6)], n // 6))
    meta = rng.standard_normal(n)
    y = 0.2 * meta + rng.standard_normal(n) * 0.5
    preds = 0.85 * meta + rng.standard_normal(n) * 0.1
    neutralizer = MetaModelNeutralizer(proportion=0.5)
    optimized = neutralizer.optimize_proportion(
        preds, meta, pd.Series(y), eras, objective="mmc_sharpe"
    )
    assert 0.0 <= optimized <= 1.0


def test_meta_model_from_benchmarks_aligns_to_index() -> None:
    rng = np.random.default_rng(3)
    index = pd.Index(["a", "b", "c"])
    benchmarks = pd.DataFrame(
        {META_MODEL_COLUMN: rng.standard_normal(3)},
        index=index,
    )
    meta = meta_model_from_benchmarks(benchmarks, index)
    assert meta is not None
    assert meta.shape == (3,)


class _FeatureModel(BaseModel):
    def __init__(self) -> None:
        super().__init__("feature")
        self.is_trained = True

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            0.9 * X["f1"].to_numpy(dtype=np.float64)
            + 0.1 * X["f2"].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> BaseModel:
        return self


def test_to_numerai_predict_applies_meta_neutralization() -> None:
    from alphapulse.evaluation.metrics import rank_normalize

    rng = np.random.default_rng(4)
    n = 80
    X = pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
        }
    )
    meta = rng.standard_normal(n)
    pipeline = Pipeline(
        preprocessors=[_IdentityPreprocessor()],
        models=[_FeatureModel()],
        feature_columns=list(X.columns),
        meta_neutralize_proportion=0.8,
    )
    direct_with = pipeline.predict(X, meta_model=meta)
    with pytest.raises(ValueError, match="were not provided"):
        pipeline.predict(X, meta_model=None)

    predict_fn = pipeline.to_numerai_predict()
    benchmarks = pd.DataFrame({META_MODEL_COLUMN: meta}, index=X.index)
    exported_with = predict_fn(X, benchmarks)["prediction"].to_numpy()
    assert np.allclose(exported_with, rank_normalize(direct_with))
    with pytest.raises(ValueError, match="requires benchmark column"):
        predict_fn(X, pd.DataFrame(index=X.index))


def test_multihead_meta_neutralization_reduces_exposure() -> None:
    rng = np.random.default_rng(12)
    n = 120
    meta = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "f1": meta + 0.05 * rng.standard_normal(n),
            "f2": rng.standard_normal(n),
        }
    )
    head = HeadSpec(
        model=_FeatureModel(),
        input_columns=list(X.columns),
        input_group=None,
        local_preprocessors=[],
        feature_groups={},
    )
    pipeline = MultiHeadPipeline(
        global_preprocessors=[],
        heads=[head],
        feature_columns=list(X.columns),
        meta_neutralize_proportion=1.0,
    )

    raw = _FeatureModel().predict(X)
    neutralized = pipeline.predict(X, meta_model=meta)

    assert abs(np.corrcoef(neutralized, meta)[0, 1]) < abs(np.corrcoef(raw, meta)[0, 1])


def test_multitarget_meta_neutralization_reduces_exposure() -> None:
    rng = np.random.default_rng(13)
    n = 120
    meta = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "f1": meta + 0.05 * rng.standard_normal(n),
            "f2": rng.standard_normal(n),
        }
    )
    targets = pd.DataFrame(
        {
            "target": rng.standard_normal(n),
            "target_aux": rng.standard_normal(n),
        }
    )
    pipeline = MultiTargetPipeline(
        preprocessors=[],
        model_factory=_FeatureModel,
        target_columns=list(targets.columns),
        meta_neutralize_proportion=1.0,
    )
    pipeline.fit(X, targets)

    raw = _FeatureModel().predict(X)
    neutralized = pipeline.predict(X, meta_model=meta)

    assert abs(np.corrcoef(neutralized, meta)[0, 1]) < abs(np.corrcoef(raw, meta)[0, 1])


def test_wandb_diagnostics_aligns_meta_and_eras_for_all_pipeline_types() -> None:
    index = pd.Index([f"row_{i}" for i in range(12)])
    meta_values = np.linspace(-1.0, 1.0, len(index))
    X = pd.DataFrame(
        {
            "f1": meta_values + np.sin(np.arange(len(index))),
            "f2": np.cos(np.arange(len(index))),
        },
        index=index,
    )
    y = pd.Series(np.linspace(0.0, 1.0, len(index)), index=index)
    eras = pd.Series(np.repeat(["era_1", "era_2"], 6), index=index)
    meta = pd.Series(meta_values, index=index)

    pipeline = Pipeline(
        preprocessors=[],
        models=[_FeatureModel()],
        feature_columns=list(X.columns),
        meta_neutralize_proportion=1.0,
    )
    multihead = MultiHeadPipeline(
        global_preprocessors=[],
        heads=[
            HeadSpec(
                model=_FeatureModel(),
                input_columns=list(X.columns),
                input_group=None,
                local_preprocessors=[],
                feature_groups={},
            )
        ],
        feature_columns=list(X.columns),
        meta_neutralize_proportion=1.0,
    )
    multitarget = MultiTargetPipeline(
        preprocessors=[],
        model_factory=_FeatureModel,
        target_columns=["target"],
        meta_neutralize_proportion=1.0,
    )
    multitarget.fit(X, pd.DataFrame({"target": y}, index=index))

    shuffled_y = y.sample(frac=1.0, random_state=1)
    shuffled_eras = eras.sample(frac=1.0, random_state=2)
    shuffled_meta = meta.sample(frac=1.0, random_state=3)
    mock_wandb = MagicMock()

    for current in (pipeline, multihead, multitarget):
        expected = current.predict(X, eras=eras, meta_model=meta)
        with (
            patch(
                "alphapulse.evaluation.wandb_diagnostics._wandb_active",
                return_value=True,
            ),
            patch(
                "alphapulse.evaluation.wandb_diagnostics._log_per_era_correlation"
            ) as log_per_era,
            patch(
                "alphapulse.evaluation.wandb_diagnostics._log_prediction_diagnostics"
            ),
            patch("alphapulse.evaluation.wandb_diagnostics._log_feature_exposure"),
            patch.dict("sys.modules", {"wandb": mock_wandb}),
        ):
            log_experiment_diagnostics(
                pipeline=current,
                X_val=X,
                y_val=shuffled_y,
                era_val=shuffled_eras,
                feature_cols=list(X.columns),
                metrics={},
                meta_model_preds=shuffled_meta,
                log_shap=False,
                log_feature_report=False,
                compute_fnc=False,
            )

        logged_y, logged_predictions, logged_eras = log_per_era.call_args.args
        pd.testing.assert_series_equal(logged_y, y)
        pd.testing.assert_series_equal(logged_eras, eras)
        np.testing.assert_allclose(logged_predictions, expected)


def test_builders_preserve_outer_neutralization_config() -> None:
    config = {
        "models": [
            {
                "type": "Ridge",
                "params": {},
                "input_columns": ["f1"],
                "use_era_ensemble": False,
            }
        ],
        "neutralize_proportion": 0.4,
        "neutralize_features": ["f1"],
        "meta_neutralize_proportion": 0.6,
    }

    multihead = build_multi_head_pipeline(config, feature_columns=["f1"])
    multitarget = build_multi_target_from_config(
        config,
        {
            "primary_target": "target",
            "auxiliary_targets": ["target_aux"],
            "target_blend_method": "equal",
        },
        feature_columns=["f1"],
    )

    assert (
        multihead.neutralize_proportion,
        multihead.meta_neutralize_proportion,
        multitarget.neutralize_proportion,
        multitarget.meta_neutralize_proportion,
    ) == (0.4, 0.6, 0.4, 0.6)


def test_multitarget_attaches_era_for_outer_preprocessors() -> None:
    n = 12
    X = pd.DataFrame({"f1": np.linspace(0.0, 1.0, n), "f2": 0.0})
    eras = pd.Series(np.repeat(["era_1", "era_2"], n // 2), index=X.index)
    preprocessor = _EraRequiredPreprocessor()
    pipeline = MultiTargetPipeline(
        preprocessors=[preprocessor],
        model_factory=_FeatureModel,
        target_columns=["target"],
    )

    pipeline.fit(
        X,
        pd.DataFrame({"target": np.linspace(0.0, 1.0, n)}),
        era_train=eras,
    )

    assert preprocessor.saw_era


def test_multitarget_builder_does_not_duplicate_inner_preprocessors() -> None:
    config = {
        "preprocessors": [{"type": "StandardScaler", "params": {}}],
        "models": [
            {"type": "Ridge", "params": {}, "use_era_ensemble": False},
            {"type": "Ridge", "params": {}, "use_era_ensemble": False},
        ],
    }
    pipeline = build_multi_target_from_config(
        config,
        {"primary_target": "target", "auxiliary_targets": ["target_aux"]},
        feature_columns=["f1", "f2"],
    )
    adapter = pipeline.model_factory()
    inner_pipeline = cast(Any, adapter)._pipeline

    assert pipeline.preprocessors == []
    assert isinstance(inner_pipeline, MultiHeadPipeline)
    assert len(inner_pipeline.global_preprocessors) == 1


def test_multitarget_aligns_shuffled_validation_targets_by_row_id() -> None:
    recorded: list[pd.Series] = []

    class ValidationRecorder(_ConstModel):
        def __init__(self) -> None:
            super().__init__(0.5)

        def train(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None,
            **kwargs: Any,
        ) -> dict[str, float]:
            if y_val is not None:
                recorded.append(y_val.copy())
            return {}

    train_index = pd.Index([f"train_{i}" for i in range(12)])
    val_index = pd.Index(["row_a", "row_b", "row_c"])
    X = pd.DataFrame({"f1": np.arange(12), "f2": 0.0}, index=train_index)
    targets = pd.DataFrame({"target": np.arange(12)}, index=train_index)
    X_val = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": 0.0}, index=val_index)
    targets_val = pd.DataFrame(
        {"target": [30.0, 20.0, 10.0]},
        index=pd.Index(["row_c", "row_b", "row_a"]),
    )
    pipeline = MultiTargetPipeline(
        preprocessors=[],
        model_factory=ValidationRecorder,
        target_columns=["target"],
    )

    pipeline.fit(X, targets, X_val=X_val, targets_val=targets_val)

    assert recorded[0].index.equals(val_index)
    assert recorded[0].tolist() == [10.0, 20.0, 30.0]


def test_multitarget_rejects_mismatched_validation_row_ids() -> None:
    X = pd.DataFrame({"f1": np.arange(12), "f2": 0.0})
    targets = pd.DataFrame({"target": np.arange(12)})
    X_val = pd.DataFrame({"f1": [1.0, 2.0]}, index=["row_a", "row_b"])
    targets_val = pd.DataFrame({"target": [1.0, 2.0]}, index=["row_a", "row_c"])
    pipeline = MultiTargetPipeline(
        preprocessors=[],
        model_factory=lambda: _ConstModel(0.5),
        target_columns=["target"],
    )

    with pytest.raises(ValueError, match="row IDs must exactly match"):
        pipeline.fit(X, targets, X_val=X_val, targets_val=targets_val)


def test_multitarget_rejects_targets_without_enough_valid_rows() -> None:
    X = pd.DataFrame({"f1": np.arange(12), "f2": 0.0})
    targets = pd.DataFrame(
        {"target": [*np.arange(9, dtype=np.float64), np.nan, np.nan, np.nan]}
    )
    pipeline = MultiTargetPipeline(
        preprocessors=[],
        model_factory=lambda: _ConstModel(0.5),
        target_columns=["target"],
    )

    with pytest.raises(ValueError, match="enough valid training rows"):
        pipeline.fit(X, targets)


def test_multitarget_failed_refit_clears_previous_fitted_state() -> None:
    X = pd.DataFrame({"f1": np.arange(12), "f2": 0.0})
    pipeline = MultiTargetPipeline(
        preprocessors=[],
        model_factory=lambda: _ConstModel(0.5),
        target_columns=["target"],
    )
    pipeline.fit(X, pd.DataFrame({"target": np.arange(12)}))

    with pytest.raises(ValueError, match="enough valid training rows"):
        pipeline.fit(X, pd.DataFrame({"target": [np.nan] * 12}))
    with pytest.raises(ValueError, match="No fitted models"):
        pipeline.predict(X)


def test_feature_neutralization_rejects_missing_configured_columns() -> None:
    X = pd.DataFrame({"f1": [0.1, 0.2, 0.3], "f2": [0.3, 0.2, 0.1]})
    pipeline = Pipeline(
        preprocessors=[],
        models=[_FeatureModel()],
        feature_columns=list(X.columns),
        neutralize_proportion=0.5,
        neutralize_features=["missing_feature"],
    )

    with pytest.raises(ValueError, match="missing configured feature"):
        pipeline.predict(X)


def test_all_pipeline_exports_fail_when_benchmark_blend_input_is_missing() -> None:
    X = pd.DataFrame(
        {
            "f1": np.linspace(0.1, 0.9, 12),
            "f2": np.linspace(0.9, 0.1, 12),
        }
    )
    head = HeadSpec(
        model=_FeatureModel(),
        input_columns=list(X.columns),
        input_group=None,
        local_preprocessors=[],
        feature_groups={},
    )
    multi_target = MultiTargetPipeline(
        preprocessors=[],
        model_factory=_FeatureModel,
        target_columns=["target"],
        benchmark_blend_weight=0.2,
    )
    multi_target.fit(X, pd.DataFrame({"target": np.linspace(0.2, 0.8, 12)}))
    pipelines: list[Any] = [
        Pipeline(
            preprocessors=[],
            models=[_FeatureModel()],
            feature_columns=list(X.columns),
            benchmark_blend_weight=0.2,
        ),
        MultiHeadPipeline(
            global_preprocessors=[],
            heads=[head],
            feature_columns=list(X.columns),
            benchmark_blend_weight=0.2,
        ),
        multi_target,
    ]

    for pipeline in pipelines:
        with pytest.raises(ValueError, match="no benchmark column"):
            pipeline.to_numerai_predict()(X, pd.DataFrame(index=X.index))
        with pytest.raises(ValueError, match="requires column"):
            pipeline.to_numerai_predict("benchmark")(X, pd.DataFrame(index=X.index))


def test_all_pipelines_align_shuffled_meta_and_era_series_by_row_id() -> None:
    index = pd.Index([f"row_{i}" for i in range(12)])
    X = pd.DataFrame(
        {
            "f1": np.linspace(-1.0, 1.0, len(index)),
            "f2": np.cos(np.arange(len(index))),
        },
        index=index,
    )
    eras = pd.Series(np.repeat(["era_1", "era_2", "era_3"], 4), index=index)
    meta = pd.Series(np.sin(np.arange(len(index))), index=index)
    head = HeadSpec(
        model=_FeatureModel(),
        input_columns=list(X.columns),
        input_group=None,
        local_preprocessors=[],
        feature_groups={},
    )
    multi_target = MultiTargetPipeline(
        preprocessors=[],
        model_factory=_FeatureModel,
        target_columns=["target"],
        neutralize_proportion=0.5,
        meta_neutralize_proportion=0.5,
    )
    multi_target.fit(
        X,
        pd.DataFrame({"target": np.linspace(0.0, 1.0, len(index))}, index=index),
    )
    pipelines: list[Any] = [
        Pipeline(
            preprocessors=[],
            models=[_FeatureModel()],
            feature_columns=list(X.columns),
            neutralize_proportion=0.5,
            meta_neutralize_proportion=0.5,
        ),
        MultiHeadPipeline(
            global_preprocessors=[],
            heads=[head],
            feature_columns=list(X.columns),
            neutralize_proportion=0.5,
            meta_neutralize_proportion=0.5,
        ),
        multi_target,
    ]
    shuffled_eras = eras.sample(frac=1.0, random_state=1)
    shuffled_meta = meta.sample(frac=1.0, random_state=2)

    for pipeline in pipelines:
        expected = pipeline.predict(X, eras=eras, meta_model=meta)
        actual = pipeline.predict(
            X,
            eras=shuffled_eras,
            meta_model=shuffled_meta,
        )
        assert np.allclose(actual, expected)


def test_pipeline_and_multihead_align_targets_by_row_id() -> None:
    recorded: list[tuple[pd.Series, pd.Series | None]] = []

    class TargetRecorder(_ConstModel):
        def __init__(self) -> None:
            super().__init__(0.5)

        def train(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None,
            **kwargs: Any,
        ) -> dict[str, float]:
            recorded.append((y_train.copy(), None if y_val is None else y_val.copy()))
            return {}

    train_index = pd.Index([f"train_{i}" for i in range(12)])
    val_index = pd.Index(["val_a", "val_b", "val_c"])
    X = pd.DataFrame({"f1": np.arange(12), "f2": 0.0}, index=train_index)
    y = pd.Series(np.arange(12), index=train_index).iloc[::-1]
    X_val = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": 0.0}, index=val_index)
    y_val = pd.Series([10.0, 20.0, 30.0], index=val_index).iloc[::-1]

    models = [TargetRecorder(), TargetRecorder()]
    pipelines: list[Any] = [
        Pipeline(preprocessors=[], models=[models[0]]),
        MultiHeadPipeline(
            global_preprocessors=[],
            heads=[
                HeadSpec(
                    model=models[1],
                    input_columns=list(X.columns),
                    input_group=None,
                    local_preprocessors=[],
                    feature_groups={},
                )
            ],
        ),
    ]

    for pipeline in pipelines:
        pipeline.fit(X, y, X_val=X_val, y_val=y_val)

    for train_target, val_target in recorded:
        assert train_target.index.equals(train_index)
        assert val_target is not None and val_target.index.equals(val_index)
        assert train_target.tolist() == list(range(12))
        assert val_target.tolist() == [10.0, 20.0, 30.0]


def test_backtester_aligns_all_indexed_inputs_by_row_id() -> None:
    index = pd.Index([f"row_{i}" for i in range(12)])
    X = pd.DataFrame(
        {"f1": np.linspace(-1.0, 1.0, 12), "f2": np.linspace(1.0, -1.0, 12)},
        index=index,
    )
    y = pd.Series(np.linspace(0.0, 1.0, 12), index=index)
    era = pd.Series(np.repeat(["era_1", "era_2"], 6), index=index)
    meta = pd.Series(np.linspace(1.0, 0.0, 12), index=index)
    pipeline = Pipeline(
        preprocessors=[],
        models=[_FeatureModel()],
        feature_columns=list(X.columns),
    )
    backtester = Backtester(pipeline)

    expected = backtester.evaluate(X, y, era, meta_model_preds=meta)
    actual = backtester.evaluate(
        X,
        y.sample(frac=1.0, random_state=1),
        era.sample(frac=1.0, random_state=2),
        meta_model_preds=meta.sample(frac=1.0, random_state=3),
    )

    for key, expected_value in expected.items():
        if np.isnan(expected_value):
            assert np.isnan(actual[key])
        else:
            assert actual[key] == pytest.approx(expected_value)


def test_backtester_requires_meta_predictions_when_neutralization_is_configured() -> (
    None
):
    X = pd.DataFrame({"f1": np.linspace(-1.0, 1.0, 20), "f2": 0.0})
    y = pd.Series(np.linspace(0.0, 1.0, 20))
    eras = pd.Series(np.repeat(["era1", "era2"], 10))
    pipeline = Pipeline(
        preprocessors=[],
        models=[_FeatureModel()],
        feature_columns=list(X.columns),
        meta_neutralize_proportion=0.5,
    )

    with pytest.raises(ValueError, match="were not provided"):
        Backtester(pipeline).evaluate(X, y, eras)


def test_backtester_rejects_incomplete_meta_predictions() -> None:
    X = pd.DataFrame({"f1": np.linspace(-1.0, 1.0, 20), "f2": 0.0})
    y = pd.Series(np.linspace(0.0, 1.0, 20))
    eras = pd.Series(np.repeat(["era1", "era2"], 10))
    meta = np.linspace(0.0, 1.0, 20)
    meta[3] = np.nan
    pipeline = Pipeline(
        preprocessors=[],
        models=[_FeatureModel()],
        feature_columns=list(X.columns),
        meta_neutralize_proportion=0.5,
    )

    with pytest.raises(ValueError, match="missing or non-finite"):
        Backtester(pipeline).evaluate(X, y, eras, meta_model_preds=meta)
