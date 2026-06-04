import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphapulse.autoresearch.loop import _run_one_trial
from alphapulse.experiments.runner import run_experiment
from alphapulse.experiments.schema import ExperimentV1, ModelSpec
from alphapulse.hpo.builder import instantiate_model
from alphapulse.models import CatBoostModel, LightGBMModel, ModelFactory
from alphapulse.models.diffusion_augmenter import SyntheticDataAugmenter
from alphapulse.models.era_ensemble_model import EraEnsembleModel
from alphapulse.models.xgboost_model import XGBoostModel
from alphapulse.pipeline.ensemble_optimizer import EnsembleOptimizer
from alphapulse.pipeline.multi_target import MultiTargetPipeline
from alphapulse.preprocessors import StandardScalerPreprocessor


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
    features_json = {
        "feature_sets": {"small": ["feature_a"], "all": ["feature_a", "feature_b"]}
    }
    (tmp_path / "features.json").write_text(json.dumps(features_json))
    return tmp_path


@pytest.fixture
def toy_data_with_era() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    n_eras = 40
    rows_per_era = 8
    n = n_eras * rows_per_era
    X = pd.DataFrame(
        rng.standard_normal((n, 4)).astype(np.float64), columns=list("ABCD")
    )
    X["era"] = np.repeat([f"era_{i:04d}" for i in range(n_eras)], rows_per_era)
    y = pd.Series(X["A"] * 0.5 + X["B"] * 0.3 + rng.standard_normal(n) * 0.2)
    feature_cols = list("ABCD")
    return {
        "X_train": X,
        "y_train": y,
        "era_train": X["era"],
        "feature_cols": feature_cols,
    }


def test_model_spec_normalizes_top_level_hyperparams() -> None:
    spec = ModelSpec(
        type="XGBoost",
        params={"max_depth": 4, "learning_rate": 0.05},
    )
    assert "params" in spec.params
    assert spec.params["params"]["max_depth"] == 4
    assert spec.params["params"]["learning_rate"] == 0.05


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
    assert "sharpe" in result.metrics
    assert result.config_hash
    artifact_path = numerai_dataset_dir / "artifacts" / "resolved_pipeline_config.json"
    assert artifact_path.exists()


def test_instantiate_model_matches_model_factory() -> None:
    params = {
        "params": {"max_depth": 3, "learning_rate": 0.1},
        "name": "TestXGB",
    }
    from_builder = instantiate_model("XGBoost", params, index=0, n_subs=3)
    from_factory = ModelFactory().suggest_fixed("xgboost", params, n_subs=3)
    assert isinstance(from_builder, EraEnsembleModel)
    assert isinstance(from_factory, EraEnsembleModel)
    assert from_builder.n_subs == from_factory.n_subs == 3


def test_lightgbm_train_predict_smoke(toy_data_with_era: dict[str, Any]) -> None:
    X = toy_data_with_era["X_train"].drop(columns=["era"])
    y = toy_data_with_era["y_train"]
    model = LightGBMModel(
        params={"objective": "regression", "verbosity": -1},
        n_estimators=20,
    )
    model.train(X, y, n_rounds=20)
    preds = model.predict(X.iloc[:10])
    assert preds.shape == (10,)


def test_catboost_train_predict_smoke(toy_data_with_era: dict[str, Any]) -> None:
    X = toy_data_with_era["X_train"].drop(columns=["era"])
    y = toy_data_with_era["y_train"]
    model = CatBoostModel(
        params={"verbose": 0, "allow_writing_files": False},
        iterations=20,
    )
    model.train(X, y, n_rounds=20)
    preds = model.predict(X.iloc[:10])
    assert preds.shape == (10,)


def test_multi_target_pipeline_fit_predict(toy_data_with_era: dict[str, Any]) -> None:
    X = toy_data_with_era["X_train"].drop(columns=["era"])
    targets = pd.DataFrame(
        {
            "target": toy_data_with_era["y_train"],
            "target_aux": toy_data_with_era["y_train"] * 0.5,
        }
    )
    era = toy_data_with_era["X_train"]["era"]

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
    pipeline.fit(X, targets, era_train=era, n_rounds=10)
    preds = pipeline.predict(X.iloc[:20])
    assert preds.shape == (20,)


def test_ensemble_optimizer_fit_predict() -> None:
    rng = np.random.RandomState(0)
    n = 200
    eras = pd.Series(np.repeat(["e1", "e2", "e3", "e4"], n // 4))
    y = rng.randn(n)
    oof = np.column_stack([y + rng.randn(n) * 0.5, y + rng.randn(n) * 0.8])

    optimizer = EnsembleOptimizer(seed=0)
    optimizer.fit(oof, y, eras)
    assert optimizer.weights_ is not None
    assert optimizer.weights_.sum() == pytest.approx(1.0)
    blend = optimizer.predict(oof[:10])
    assert blend.shape == (10,)


def test_synthetic_data_augmenter_kde_fit_once() -> None:
    rng = np.random.RandomState(0)
    n = 80
    X = pd.DataFrame(rng.randn(n, 3), columns=list("ABC"))
    y = pd.Series(rng.randn(n))

    aug = SyntheticDataAugmenter(
        top_fraction=0.25, n_synthetic=10, backend="kde", seed=0
    )
    aug.fit(X, y)
    assert aug._kde is not None or aug._kde_fallback_combined is not None

    X1, y1 = aug.generate()
    X2, y2 = aug.generate()
    assert len(X1) == len(X2) == 10
    assert list(X1.columns) == list("ABC")


def test_autoresearch_run_one_trial(toy_data_with_era: dict[str, Any]) -> None:
    config = {
        "preprocessors": [{"type": "StandardScaler", "params": {}}],
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
        "ensemble_params": {},
    }
    metrics, elapsed = _run_one_trial(
        config,
        seed=0,
        X_train=toy_data_with_era["X_train"],
        y_train=toy_data_with_era["y_train"],
        era_train=toy_data_with_era["era_train"],
        feature_cols=toy_data_with_era["feature_cols"],
    )
    assert elapsed > 0
    assert "corr_sharpe" in metrics


def test_autoresearch_run_one_trial_pipeline_is_fit(
    toy_data_with_era: dict[str, Any],
) -> None:
    config = {
        "preprocessors": [],
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
        "ensemble_params": {},
    }
    with patch("alphapulse.autoresearch.loop.build_pipeline_or_multi") as mock_build:
        mock_pipeline = mock_build.return_value
        mock_pipeline.predict.side_effect = lambda X: np.zeros(len(X))
        _run_one_trial(
            config,
            seed=0,
            X_train=toy_data_with_era["X_train"],
            y_train=toy_data_with_era["y_train"],
            era_train=toy_data_with_era["era_train"],
            feature_cols=toy_data_with_era["feature_cols"],
        )
        assert mock_pipeline.fit.called
