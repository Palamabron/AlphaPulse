import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphapulse.evaluation import Backtester
from alphapulse.experiments.runner import run_experiment
from alphapulse.experiments.schema import ExperimentV1
from alphapulse.experiments.split import internal_val_split
from alphapulse.models import XGBoostModel
from alphapulse.models.era_ensemble_model import EraEnsembleModel
from alphapulse.pipeline import Pipeline
from alphapulse.pipeline.ensemble import EnsembleStrategy
from alphapulse.pipeline.multi_target import MultiTargetPipeline
from alphapulse.preprocessors import PCAPreprocessor, StandardScalerPreprocessor
from alphapulse.preprocessors.feature_selection import VarianceFeatureSelector


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(np.random.randn(n, 4).astype(np.float64), columns=list("ABCD"))
    y = pd.Series(X["A"] * 0.5 + X["B"] * 0.3 + np.random.randn(n) * 0.2)
    return X, y


@pytest.fixture
def numerai_dataset_dir(tmp_path: Path) -> Path:
    rng = np.random.RandomState(0)
    n = 240
    eras = np.repeat([f"era_{i:04d}" for i in range(6)], n // 6)
    df = pd.DataFrame(
        {
            "feature_a": rng.randn(n).astype(np.float32),
            "feature_b": rng.randn(n).astype(np.float32),
            "era": eras,
            "target": rng.randn(n).astype(np.float32),
            "id": [f"id_{i}" for i in range(n)],
        }
    )
    df.to_parquet(tmp_path / "train.parquet", index=False)
    (tmp_path / "validation.parquet").write_bytes(
        (tmp_path / "train.parquet").read_bytes()
    )
    (tmp_path / "features.json").write_text(
        json.dumps(
            {
                "feature_sets": {
                    "small": ["feature_a"],
                    "all": ["feature_a", "feature_b"],
                }
            }
        )
    )
    return tmp_path


def _xgb(name: str = "xgb") -> XGBoostModel:
    return XGBoostModel(
        params={
            "max_depth": 2,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "objective": "reg:squarederror",
        },
        name=name,
    )


@pytest.fixture
def fitted_pipeline(toy_data: tuple[pd.DataFrame, pd.Series]) -> Pipeline:
    X, y = toy_data
    pipeline = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        model=XGBoostModel(
            params={
                "max_depth": 3,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "objective": "reg:squarederror",
            }
        ),
    )
    pipeline.fit(X, y, n_rounds=10)
    return pipeline


def test_pipeline_fit_predict_preserves_shape(
    fitted_pipeline: Pipeline, toy_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, _ = toy_data
    preds = fitted_pipeline.predict(X)
    assert preds.shape == (len(X),)
    assert isinstance(preds, np.ndarray)


def test_to_numerai_predict_signature(fitted_pipeline: Pipeline) -> None:
    predict_fn = fitted_pipeline.to_numerai_predict()
    live = pd.DataFrame(
        np.random.randn(10, 4).astype(np.float64),
        columns=fitted_pipeline.feature_columns,
    )
    bench = pd.DataFrame()
    out = predict_fn(live, bench)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 1)
    assert "prediction" in out.columns
    assert list(out.index) == list(live.index)
    # rank-normalized output should be in [0, 1]
    assert out["prediction"].min() >= 0.0
    assert out["prediction"].max() <= 1.0


def test_backtester_returns_expected_keys(
    toy_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline: Pipeline
) -> None:
    X, y = toy_data
    era = pd.Series(np.repeat(["e1", "e2"], 100), index=X.index)
    backtester = Backtester(fitted_pipeline, feature_columns=list(X.columns))
    metrics = backtester.evaluate(X, y, era)
    expected = {
        "mean_per_era_correlation",
        "std_per_era_correlation",
        "corr_sharpe",
    }
    assert expected <= set(metrics.keys())


def test_pipeline_weighted_ensemble(toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = toy_data
    pipeline = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        models=[
            XGBoostModel(
                params={
                    "max_depth": 2,
                    "learning_rate": 0.1,
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                }
            ),
            XGBoostModel(
                params={
                    "max_depth": 3,
                    "learning_rate": 0.05,
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                }
            ),
        ],
        ensemble_method="weighted",
        ensemble_params={"weights": [0.5, 0.5]},
    )
    pipeline.fit(X, y, n_rounds=10)
    preds = pipeline.predict(X)
    assert preds.shape == (len(X),)
    assert isinstance(preds, np.ndarray)


def test_pipeline_model_and_models_mutually_exclusive(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    model = XGBoostModel(
        params={
            "max_depth": 2,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "objective": "reg:squarederror",
        }
    )
    with pytest.raises(ValueError, match="model or models"):
        Pipeline(
            preprocessors=[StandardScalerPreprocessor()],
            model=model,
            models=[model],
        )


def test_pipeline_single_model_normalizes_to_list(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    model = XGBoostModel(
        params={
            "max_depth": 2,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "objective": "reg:squarederror",
        }
    )
    pipeline = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        model=model,
    )
    assert pipeline.models == [model]
    pipeline.fit(X, y, n_rounds=5)
    assert pipeline.predict(X).shape == (len(X),)


def test_neutralization_rejects_missing_configured_features(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    pipeline = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        model=XGBoostModel(
            params={
                "max_depth": 2,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "objective": "reg:squarederror",
            }
        ),
        neutralize_proportion=0.5,
        neutralize_features=["missing_col"],
    )
    pipeline.fit(X, y, n_rounds=5)
    with pytest.raises(ValueError, match="missing_col"):
        pipeline.predict(X)


def test_unfitted_stacking_raises() -> None:
    es = EnsembleStrategy(method="stacking")
    with pytest.raises(RuntimeError, match="meta-learner is not fitted"):
        es.combine(np.random.randn(10, 2))


class TestInternalValSplit:
    def test_era_aware_split_uses_last_eras(self) -> None:
        n_eras, rows_per_era = 20, 5
        X = pd.DataFrame(np.random.randn(n_eras * rows_per_era, 2), columns=["a", "b"])
        y = pd.Series(np.random.randn(len(X)))
        era = pd.Series(np.repeat([f"e{i:03d}" for i in range(n_eras)], rows_per_era))
        X_tr, y_tr, X_va, y_va = internal_val_split(
            X, y, era_train=era, force_internal=True
        )
        assert len(X_tr) + len(X_va) == len(X)
        assert set(era.loc[X_va.index]).isdisjoint(set(era.loc[X_tr.index]))

    def test_internal_val_split_uses_temporal_last_eras(self) -> None:
        n_eras, rows_per_era = 10, 4
        eras = [f"e{i:03d}" for i in range(n_eras)]
        rng = np.random.RandomState(0)
        order = rng.permutation(n_eras * rows_per_era)
        era = pd.Series(np.repeat(eras, rows_per_era)[order])
        X = pd.DataFrame({"a": np.arange(n_eras * rows_per_era)[order]})
        y = pd.Series(np.arange(n_eras * rows_per_era)[order] * 0.1)
        _, _, X_va, _ = internal_val_split(X, y, era_train=era, force_internal=True)
        assert X_va is not None
        val_eras = set(era.loc[X_va.index])
        assert val_eras == {"e009"}

    def test_stacking_forces_internal_split(self) -> None:
        X = pd.DataFrame({"a": np.arange(100.0)})
        y = pd.Series(np.arange(100.0))
        era = pd.Series(np.repeat([f"e{i}" for i in range(10)], 10))
        _, _, X_va, _ = internal_val_split(X, y, era_train=era, force_internal=True)
        assert X_va is not None and len(X_va) > 0


def test_pipeline_save_load_roundtrip(
    toy_data: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    X, y = toy_data
    pipe = Pipeline(preprocessors=[StandardScalerPreprocessor()], model=_xgb())
    pipe.fit(X, y, n_rounds=5)
    path = tmp_path / "pipe.pkl"
    pipe.save_pipeline(path)
    loaded = Pipeline.load_pipeline(path)
    np.testing.assert_allclose(loaded.predict(X), pipe.predict(X))


def test_era_survives_pca_for_era_ensemble(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    era = pd.Series(np.repeat([f"e{i:03d}" for i in range(20)], 10), index=X.index)
    X = X.assign(era=era)

    def factory() -> XGBoostModel:
        return _xgb("sub")

    model = EraEnsembleModel(base_model_factory=factory, n_subs=4)
    pipe = Pipeline(
        preprocessors=[StandardScalerPreprocessor(), PCAPreprocessor(n_components=2)],
        model=model,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe.fit(X, y, n_rounds=5)
    assert not [w for w in caught if "falling back" in str(w.message)]
    assert len(model._sub_models) > 1


def test_all_nan_rows_returns_finite(toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = toy_data
    pipe = Pipeline(preprocessors=[StandardScalerPreprocessor()], model=_xgb())
    pipe.fit(X, y, n_rounds=5)
    X_bad = X.copy()
    X_bad[:] = np.nan
    preds = pipe.predict(X_bad)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()


def test_pipeline_predict_string_index_invalid_row_imputation(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    X = X.copy()
    X.index = [f"row_{i}" for i in range(len(X))]
    y.index = X.index
    pipe = Pipeline(preprocessors=[StandardScalerPreprocessor()], model=_xgb())
    pipe.fit(X, y, n_rounds=5)
    X_mixed = X.copy()
    X_mixed.iloc[0] = np.nan
    X_mixed.iloc[3] = np.inf
    preds = pipe.predict(X_mixed)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()
    assert preds[0] == pytest.approx(np.median(preds[1:]), rel=0.05)


def test_variance_selector_in_pipeline_with_era_column(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    era = pd.Series(np.repeat(["e1", "e2"], len(X) // 2), index=X.index)
    X = X.assign(era=era)
    pipe = Pipeline(
        preprocessors=[VarianceFeatureSelector(keep_fraction=0.5)],
        model=_xgb(),
    )
    pipe.fit(X, y, n_rounds=5)
    preds = pipe.predict(X)
    assert preds.shape == (len(X),)


def test_numerai_predict_passes_eras_when_neutralization_enabled(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    era = pd.Series(np.repeat(["e1", "e2"], len(X) // 2), index=X.index)
    X = X.assign(era=era)
    pipe = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        model=_xgb(),
        neutralize_proportion=0.5,
        feature_columns=list(X.columns),
    )
    pipe.fit(X, y, n_rounds=5)
    out = pipe.to_numerai_predict()(X, pd.DataFrame())
    assert out.shape == (len(X), 1)


def test_multi_target_pipeline_fit_predict() -> None:
    rng = np.random.default_rng(0)
    n_eras, rows = 40, 8
    n = n_eras * rows
    X = pd.DataFrame(
        rng.standard_normal((n, 4)).astype(np.float64), columns=list("ABCD")
    )
    X["era"] = np.repeat([f"era_{i:04d}" for i in range(n_eras)], rows)
    y = pd.Series(X["A"] * 0.5 + X["B"] * 0.3 + rng.standard_normal(n) * 0.2)
    targets = pd.DataFrame({"target": y, "target_aux": y * 0.5})

    def factory() -> XGBoostModel:
        return XGBoostModel(
            params={
                "max_depth": 3,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "objective": "reg:squarederror",
            }
        )

    pipeline = MultiTargetPipeline(
        preprocessors=[StandardScalerPreprocessor()],
        model_factory=factory,
        target_columns=["target", "target_aux"],
        primary_target="target",
    )
    pipeline.fit(X.drop(columns=["era"]), targets, era_train=X["era"], n_rounds=10)
    preds = pipeline.predict(X.drop(columns=["era"]).iloc[:20])
    assert preds.shape == (20,)


def test_run_experiment_e2e(numerai_dataset_dir: Path) -> None:
    exp = ExperimentV1.model_validate(
        {
            "version": "1",
            "data": {
                "data_dir": str(numerai_dataset_dir),
                "train_subsample": 1.0,
                "target_col": "target",
                "seed": 42,
            },
            "features": {"columns": ["feature_a", "feature_b"], "groups": {}},
            "preprocessing": [{"type": "StandardScaler", "params": {}}],
            "models": [
                {
                    "type": "XGBoost",
                    "params": {
                        "params": {
                            "max_depth": 3,
                            "learning_rate": 0.1,
                            "tree_method": "hist",
                            "objective": "reg:squarederror",
                        }
                    },
                }
            ],
            "ensemble_method": "single",
            "train": {"n_rounds": 10, "early_stopping_rounds": 5},
        }
    )
    result = run_experiment(exp, artifact_dir=numerai_dataset_dir / "artifacts")
    assert result.error is None
    assert "corr_sharpe" in result.metrics
    assert result.config_hash
    assert (
        numerai_dataset_dir / "artifacts" / "resolved_pipeline_config.json"
    ).exists()
