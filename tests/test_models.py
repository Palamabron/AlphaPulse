"""Smoke tests for model implementations, ensemble optimizer, and augmenter."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphapulse.models import CatBoostModel, LightGBMModel, ModelFactory
from alphapulse.models.diffusion_augmenter import SyntheticDataAugmenter
from alphapulse.pipeline.ensemble_optimizer import EnsembleOptimizer


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
    return {"X": X.drop(columns=["era"]), "y": y}


def test_lightgbm_train_predict_smoke(toy_data_with_era: dict[str, Any]) -> None:
    model = LightGBMModel(
        params={"objective": "regression", "verbosity": -1}, n_estimators=20
    )
    model.train(toy_data_with_era["X"], toy_data_with_era["y"], n_rounds=20)
    preds = model.predict(toy_data_with_era["X"].iloc[:10])
    assert preds.shape == (10,)


def test_catboost_train_predict_smoke(toy_data_with_era: dict[str, Any]) -> None:
    model = CatBoostModel(
        params={"verbose": 0, "allow_writing_files": False}, iterations=20
    )
    model.train(toy_data_with_era["X"], toy_data_with_era["y"], n_rounds=20)
    preds = model.predict(toy_data_with_era["X"].iloc[:10])
    assert preds.shape == (10,)


def test_model_spec_normalizes_top_level_hyperparams() -> None:
    from alphapulse.experiments.schema import ModelSpec

    spec = ModelSpec(type="XGBoost", params={"max_depth": 4, "learning_rate": 0.05})
    assert "params" in spec.params
    assert spec.params["params"]["max_depth"] == 4
    assert spec.params["params"]["learning_rate"] == 0.05


def test_instantiate_model_matches_model_factory() -> None:
    from alphapulse.hpo.builder import instantiate_model
    from alphapulse.models.era_ensemble_model import EraEnsembleModel

    params = {"params": {"max_depth": 3, "learning_rate": 0.1}, "name": "TestXGB"}
    from_builder = instantiate_model("XGBoost", params, index=0, n_subs=3)
    from_factory = ModelFactory().suggest_fixed("XGBoost", params, n_subs=3)
    assert isinstance(from_builder, EraEnsembleModel)
    assert isinstance(from_factory, EraEnsembleModel)
    assert from_builder.n_subs == from_factory.n_subs == 3


def test_apply_gpu_model_params_packboost_sets_cuda() -> None:
    from alphapulse.hpo.search_space import apply_gpu_model_params

    params = apply_gpu_model_params("Packboost", {"n_rounds_base": 100})
    assert params["device"] == "cuda"


def test_apply_gpu_model_params_lightgbm_sets_device() -> None:
    from alphapulse.hpo.search_space import apply_gpu_model_params

    params = apply_gpu_model_params("LightGBM", {"params": {"verbosity": -1}})
    inner = params["params"]
    assert inner["device"] == "gpu"
    assert inner["gpu_platform_id"] == 0
    assert inner["gpu_device_id"] == 0
    assert "n_jobs" not in inner


def test_instantiate_catboost_gpu_strips_colsample_bylevel() -> None:
    from alphapulse.hpo.builder import instantiate_model
    from alphapulse.models.catboost_model import CatBoostModel
    from alphapulse.models.era_ensemble_model import EraEnsembleModel

    model = instantiate_model(
        "CatBoost",
        {"params": {"task_type": "GPU", "verbose": 0}},
        index=0,
        n_subs=2,
    )
    assert isinstance(model, EraEnsembleModel)
    base = model.base_model_factory()
    assert isinstance(base, CatBoostModel)
    assert base.params.get("task_type") == "GPU"
    assert "colsample_bylevel" not in base.params


def test_ensemble_optimizer_fit_predict() -> None:
    rng = np.random.RandomState(0)
    n = 200
    eras = pd.Series(np.repeat(["e1", "e2", "e3", "e4"], n // 4))
    y = rng.randn(n)
    oof = np.column_stack([y + rng.randn(n) * 0.5, y + rng.randn(n) * 0.8])

    optimizer = EnsembleOptimizer(seed=0, min_weight=0.05, max_weight=0.90)
    optimizer.fit(oof, y, eras)
    assert optimizer.weights_ is not None
    assert optimizer.weights_.sum() == pytest.approx(1.0)
    assert all(0.05 <= w <= 0.90 for w in optimizer.weights_)
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

    X1, _ = aug.generate()
    X2, _ = aug.generate()
    assert len(X1) == len(X2) == 10
    assert list(X1.columns) == list("ABC")


def test_xgboost_ray_callbacks_return_training_callback_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    import xgboost as xgb

    from alphapulse.models import xgboost_model

    fake_ray = types.ModuleType("ray")
    fake_tune = types.ModuleType("ray.tune")
    fake_context = types.SimpleNamespace(get_trial_id=lambda: "trial-1")
    fake_tune.get_context = lambda: fake_context  # type: ignore[attr-defined]
    fake_tune.report = lambda metrics: None  # type: ignore[attr-defined]
    fake_ray.tune = fake_tune  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.tune", fake_tune)

    callbacks = xgboost_model._make_ray_callbacks()
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], xgb.callback.TrainingCallback)
