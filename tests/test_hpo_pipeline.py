"""Tests for HPO pipeline (run_trial with preloaded data)."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from scripts.hpo_pipeline import main as hpo_main

from alphapulse.hpo import run_trial, sample_random_config


@pytest.fixture
def toy_data_with_era() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    n_eras = 50
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


@pytest.fixture
def minimal_flat_config() -> dict[str, Any]:
    return {
        "num_models": 1,
        "model_1_type": "XGBoost",
        "model_2_type": "XGBoost",
        "model_3_type": "XGBoost",
        "scaler_type": "StandardScaler",
        "use_packboost": False,
        "packboost_n_worst_eras": 5,
        "packboost_boost_weight": 0.3,
        "packboost_n_rounds_base": 300,
        "packboost_n_rounds_boost": 100,
        "xgb_max_depth": 3,
        "xgb_learning_rate": 0.05,
        "xgb_n_rounds": 15,
        "xgb_early_stopping": 5,
        "packboost_model_n_worst_eras": 5,
        "packboost_model_boost_weight": 0.3,
        "packboost_model_n_rounds_base": 300,
        "packboost_model_n_rounds_boost": 100,
        "ensemble_method": "single",
        "stacking_meta_learner": "ridge",
    }


def test_sample_random_config_includes_targets_when_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload = {
        "feature_sets": {"medium": ["f_a"], "strength": ["f_a"]},
        "targets": ["target", "target_alpha_20"],
    }
    (data_dir / "features.json").write_text(json.dumps(payload), encoding="utf-8")
    config = sample_random_config(seed=1, fast=True, data_dir=data_dir)
    assert "target_mode" in config
    assert "primary_target" in config
    assert "use_feature_routing" in config
    assert config["use_feature_routing"] is True
    assert "active_groups" in config
    assert "routed_feature_count" in config


def test_hpo_main_default_objective_is_corr_sharpe() -> None:
    assert hpo_main.__defaults__ is not None
    assert "corr_sharpe" in hpo_main.__defaults__


def test_best_from_db_minimizes_max_drawdown(tmp_path: Path) -> None:
    from scripts.hpo_pipeline import _best_from_db

    from alphapulse.hpo.trial_db import TrialDB

    with TrialDB(tmp_path / "trials.db") as db:
        db.insert_trial(0, {"candidate": "larger_drawdown"})
        db.update_trial(
            0,
            status="completed",
            metrics={"max_drawdown": 0.20},
        )
        db.insert_trial(1, {"candidate": "smaller_drawdown"})
        db.update_trial(
            1,
            status="completed",
            metrics={"max_drawdown": 0.05},
        )
        best_score, best_config = _best_from_db(db, "max_drawdown")

    assert best_score == 0.05
    assert best_config == {"candidate": "smaller_drawdown"}


def test_persistable_flat_config_keeps_data_and_model_seeds() -> None:
    from scripts.hpo_pipeline import _persistable_flat_config

    persisted = _persistable_flat_config(
        {
            "_data_dir": "data/v5.2",
            "data_seed": 42,
            "model_seed": 47,
        }
    )

    assert persisted == {"data_seed": 42, "model_seed": 47}


def test_ray_minimizes_max_drawdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    import scripts.hpo_pipeline as hpo_pipeline

    from alphapulse.hpo import search_space

    captured: dict[str, Any] = {}
    tune_module = ModuleType("ray.tune")
    ray_module = ModuleType("ray")

    class FakeCLIReporter:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def fake_tune_run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            best_trial=SimpleNamespace(config={"candidate": "smallest_drawdown"}),
            best_result={"max_drawdown": 0.05},
        )

    tune_module.__dict__["CLIReporter"] = FakeCLIReporter
    tune_module.__dict__["run"] = fake_tune_run
    tune_module.__dict__["with_parameters"] = lambda trainable, **kwargs: trainable
    ray_module.__dict__["tune"] = tune_module
    ray_module.__dict__["init"] = lambda **kwargs: None
    ray_module.__dict__["shutdown"] = lambda: None
    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.tune", tune_module)

    X_train = pd.DataFrame({"feature": [0.0], "era": ["era_0001"]})
    y_train = pd.Series([0.0])
    monkeypatch.setattr(
        hpo_pipeline,
        "load_train_only_frame",
        lambda *args, **kwargs: (X_train, y_train, ["feature"]),
    )
    monkeypatch.setattr(search_space, "get_full_param_space", dict)

    hpo_pipeline._run_ray(
        data_dir=tmp_path,
        train_subsample=1.0,
        target_col="target",
        seed=42,
        num_trials=2,
        output_dir=tmp_path / "output",
        objective="max_drawdown",
        gpu=True,
    )

    assert captured["metric"] == "max_drawdown"
    assert captured["mode"] == "min"
    assert captured["config"]["hpo_fast"] is True
    assert captured["config"]["use_gpu"] is True
    assert captured["config"]["primary_target"] == "target"
    assert captured["config"]["auxiliary_targets"] == []
    assert captured["resources_per_trial"] == {"cpu": 1, "gpu": 1}


@pytest.mark.parametrize("objective", ["payout_score", "mmc_sharpe"])
def test_ray_rejects_objectives_without_meta_validation(
    tmp_path: Path,
    objective: str,
) -> None:
    import scripts.hpo_pipeline as hpo_pipeline

    with pytest.raises(ValueError, match="validation meta-model dataset"):
        hpo_pipeline._run_ray(
            data_dir=tmp_path,
            train_subsample=1.0,
            target_col="target",
            seed=42,
            num_trials=1,
            output_dir=tmp_path / "output",
            objective=objective,
        )


def test_ray_search_space_disables_meta_neutralization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alphapulse.hpo import search_space

    class FakeTune:
        @staticmethod
        def choice(values: list[Any]) -> tuple[str, list[Any]]:
            return "choice", values

        @staticmethod
        def uniform(low: float, high: float) -> tuple[str, float, float]:
            return "uniform", low, high

        @staticmethod
        def loguniform(low: float, high: float) -> tuple[str, float, float]:
            return "loguniform", low, high

    monkeypatch.setattr(search_space, "tune", FakeTune())
    param_space = search_space.get_full_param_space(use_gpu=True)

    assert param_space["use_meta_neutralization"] is False
    assert "meta_neutralization_proportion" not in param_space
    assert "catboost_colsample_bylevel" not in param_space


def test_best_criteria_auto_and_explicit_objective_are_distinct() -> None:
    from scripts.hpo_pipeline import _resolve_best_criteria

    assert _resolve_best_criteria("payout_score", "auto") == "robust_payout"
    assert _resolve_best_criteria("payout_score", "objective") == "objective"
    assert _resolve_best_criteria("numerai_corr_sharpe", "auto") == "objective"


def test_max_models_default_depends_on_mode_but_explicit_cap_is_preserved() -> None:
    from scripts.hpo_pipeline import _resolve_max_models

    assert _resolve_max_models(None, fast=True) == 2
    assert _resolve_max_models(None, fast=False) == 3
    assert _resolve_max_models(1, fast=False) == 1


def test_resume_rejects_changed_hpo_protocol(tmp_path: Path) -> None:
    from scripts.hpo_pipeline import _write_or_validate_protocol

    original = {"objective": "numerai_corr_sharpe", "purge_eras": 8, "fast": True}
    _write_or_validate_protocol(tmp_path, original, resume=False)

    changed = {**original, "purge_eras": 16}
    with pytest.raises(ValueError, match="protocol differs"):
        _write_or_validate_protocol(tmp_path, changed, resume=True)


def test_source_fingerprint_changes_with_source_content(tmp_path: Path) -> None:
    from scripts.hpo_pipeline import _source_tree_sha256

    source = tmp_path / "src" / "alphapulse" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = _source_tree_sha256(tmp_path)

    source.write_text("VALUE = 2\n", encoding="utf-8")

    assert _source_tree_sha256(tmp_path) != first


def test_sixty_day_auxiliary_target_enforces_sixteen_era_purge() -> None:
    from types import SimpleNamespace

    from alphapulse.hpo.objective import _effective_purge_eras

    strategy = SimpleNamespace(
        primary_target="target",
        auxiliary_targets=["target_charlie_60"],
    )

    assert _effective_purge_eras(8, strategy) == 16


def test_generic_target_enforces_minimum_eight_era_purge() -> None:
    from types import SimpleNamespace

    from alphapulse.hpo.objective import _effective_purge_eras

    strategy = SimpleNamespace(primary_target="target", auxiliary_targets=[])

    assert _effective_purge_eras(0, strategy) == 8


def test_fast_holdout_excludes_configured_purge_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alphapulse.hpo import objective as objective_module

    captured_train_eras: list[str] = []

    class Predictor:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return cast(np.ndarray, X["feature"].to_numpy(dtype=np.float64))

    def fake_fit_pipeline(
        pipeline_cfg: dict[str, Any],
        feature_cols: list[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        train_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> Predictor:
        captured_train_eras.extend(X_train["era"].unique().tolist())
        return Predictor()

    monkeypatch.setattr(objective_module, "_fit_pipeline", fake_fit_pipeline)
    era_names = [f"era_{i:04d}" for i in range(40)]
    eras = pd.Series(np.repeat(era_names, 2))
    X = pd.DataFrame(
        {
            "feature": np.linspace(0.0, 1.0, len(eras)),
            "era": eras,
        }
    )
    y = pd.Series(np.linspace(0.0, 1.0, len(eras)))

    metrics = objective_module._evaluate_holdout(
        X_train=X,
        y_train=y,
        era_train=eras,
        feature_cols=["feature"],
        pipeline_cfg={},
        train_kwargs={},
        holdout_eras=5,
        purge_eras=8,
    )

    assert captured_train_eras[-1] == "era_0026"
    assert "numerai_corr_sharpe" in metrics


@pytest.mark.parametrize("multi_target", [False, True])
def test_trial_worker_separates_data_and_model_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    multi_target: bool,
) -> None:
    import queue
    from types import SimpleNamespace

    import scripts.hpo_pipeline as hpo_pipeline

    loader_seeds: list[int] = []
    global_seeds: list[tuple[int, bool]] = []
    run_trial_calls: list[dict[str, Any]] = []
    strategy = SimpleNamespace(
        target_mode="multi_blend" if multi_target else "single",
        primary_target="target",
        auxiliary_targets=["target_aux"] if multi_target else [],
    )
    routing = SimpleNamespace(feature_columns=["feature"], feature_groups={})
    validation = SimpleNamespace(ok=True, reason=None, strategy=strategy)
    X_train = pd.DataFrame({"feature": [0.0, 1.0], "era": ["era_0001", "era_0002"]})
    y_train = pd.Series([0.0, 1.0], name="target")
    targets = pd.DataFrame({"target": y_train, "target_aux": [1.0, 0.0]})

    def load_single(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        loader_seeds.append(int(kwargs["seed"]))
        return X_train.copy(), y_train.copy(), ["feature"]

    def load_multi(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        loader_seeds.append(int(kwargs["seed"]))
        return X_train.copy(), y_train.copy(), targets.copy(), ["feature"]

    def fake_run_trial(config: dict[str, Any], **kwargs: Any) -> dict[str, float]:
        run_trial_calls.append({"config": dict(config), **kwargs})
        return {"corr_sharpe": 0.1}

    def fake_set_global_seed(seed: int, *, seed_torch: bool = True) -> None:
        global_seeds.append((seed, seed_torch))

    monkeypatch.setattr(hpo_pipeline, "load_dotenv", lambda: None)
    monkeypatch.setattr(hpo_pipeline, "set_global_seed", fake_set_global_seed)
    monkeypatch.setattr(hpo_pipeline, "load_feature_catalog", lambda path: object())
    monkeypatch.setattr(hpo_pipeline, "load_target_catalog", lambda path: object())
    monkeypatch.setattr(hpo_pipeline, "strategy_from_flat", lambda config: strategy)
    monkeypatch.setattr(hpo_pipeline, "resolve_feature_routing", lambda *args: routing)
    monkeypatch.setattr(
        hpo_pipeline,
        "validate_target_strategy_early",
        lambda *args, **kwargs: validation,
    )
    monkeypatch.setattr(
        hpo_pipeline,
        "apply_target_strategy_to_flat",
        lambda config, resolved_strategy: config,
    )
    monkeypatch.setattr(hpo_pipeline, "load_train_only_frame", load_single)
    monkeypatch.setattr(hpo_pipeline, "load_train_targets_frame", load_multi)
    monkeypatch.setattr(hpo_pipeline, "load_mmc_validation_frame", lambda *a, **k: None)
    monkeypatch.setattr(hpo_pipeline, "run_trial", fake_run_trial)
    monkeypatch.setattr(hpo_pipeline, "release_cuda_memory", lambda: None)

    for model_seed in (100, 101):
        result_queue: Any = queue.SimpleQueue()
        hpo_pipeline._trial_worker(
            flat_config={},
            data_dir=str(tmp_path),
            train_subsample=0.5,
            target_col="target",
            data_seed=17,
            model_seed=model_seed,
            result_queue=result_queue,
        )
        payload = result_queue.get_nowait()
        assert payload["ok"] is True
        assert payload["flat_config"]["data_seed"] == 17
        assert payload["flat_config"]["model_seed"] == model_seed

    assert loader_seeds == [17, 17]
    assert global_seeds == [(100, False), (101, False)]
    assert [call["seed"] for call in run_trial_calls] == [100, 101]
    assert [call["data_seed"] for call in run_trial_calls] == [17, 17]
    assert all(call["mmc_frame_preloaded"] for call in run_trial_calls)


@pytest.mark.parametrize("model_type", ["TabPFN", "TabICL", "TabPFN3", "Packboost"])
def test_foundation_and_torch_models_require_torch_seed(model_type: str) -> None:
    from scripts.hpo_pipeline import _config_requires_torch

    assert _config_requires_torch({"num_models": 1, "model_1_type": model_type})


def test_boosting_models_do_not_import_torch_for_seeding() -> None:
    from scripts.hpo_pipeline import _config_requires_torch

    assert not _config_requires_torch({"num_models": 1, "model_1_type": "XGBoost"})


def test_run_trial_returns_metrics(
    toy_data_with_era: dict[str, Any], minimal_flat_config: dict[str, Any]
) -> None:
    metrics = run_trial(
        {**minimal_flat_config, "purge_eras": 4},
        **toy_data_with_era,
    )
    assert isinstance(metrics, dict)
    assert "mean_per_era_correlation" in metrics
    assert "corr_sharpe" in metrics
    assert "max_drawdown" in metrics


def test_run_trial_fast_holdout(
    toy_data_with_era: dict[str, Any], minimal_flat_config: dict[str, Any]
) -> None:
    fast_config = {**minimal_flat_config, "hpo_fast": True}
    metrics = run_trial(fast_config, **toy_data_with_era)
    assert isinstance(metrics, dict)
    assert "corr_sharpe" in metrics


def test_sample_random_config_fast_tighter_bounds() -> None:
    config = sample_random_config(seed=42, fast=True)
    assert config.get("hpo_fast") is True
    assert config["n_subs"] <= 5
    assert config["xgb_n_rounds"] <= 400
    assert config["num_models"] <= 2
    assert config.get("foundation_max_train_rows", 10_000) <= 5_000
    assert "SyntheticDataAugmenter" not in (
        config.get("model_1_type"),
        config.get("model_2_type"),
        config.get("model_3_type"),
    )


@pytest.mark.parametrize("phase", ["phase_a", "phase_b"])
def test_random_gpu_sampling_omits_dead_catboost_colsample(phase: str) -> None:
    gpu_config = sample_random_config(seed=42, phase=phase, use_gpu=True)
    cpu_config = sample_random_config(seed=42, phase=phase, use_gpu=False)

    assert "catboost_colsample_bylevel" not in gpu_config
    assert "catboost_colsample_bylevel" in cpu_config


def test_old_gpu_catboost_config_remains_readable() -> None:
    from alphapulse.hpo.search_space import resolve_flat_config

    resolved = resolve_flat_config(
        {
            "num_models": 1,
            "model_1_type": "CatBoost",
            "use_gpu": True,
            "catboost_colsample_bylevel": 0.237,
        }
    )

    params = resolved["models"][0]["params"]["params"]
    assert params["task_type"] == "GPU"
    assert "colsample_bylevel" not in params


@pytest.mark.parametrize("phase", ["phase_a", "phase_b"])
def test_sample_random_config_disables_meta_neutralization(phase: str) -> None:
    for seed in range(10):
        config = sample_random_config(seed=seed, phase=phase)
        assert config["use_meta_neutralization"] is False
        assert "meta_neutralization_proportion" not in config


def test_local_cpu_sampling_excludes_packboost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alphapulse.hpo import search_space

    monkeypatch.setattr(
        search_space,
        "available_boosting_models",
        lambda: ["XGBoost", "LightGBM", "CatBoost", "Packboost"],
    )

    for seed in range(100):
        config = sample_random_config(seed=seed, use_gpu=False)
        model_types = [
            config.get(f"model_{i}_type")
            for i in range(1, int(config["num_models"]) + 1)
        ]
        assert "Packboost" not in model_types
        assert config["use_packboost"] is False
        assert config["use_gpu"] is False


def test_resolve_flat_config_keeps_manual_meta_neutralization() -> None:
    from alphapulse.hpo.search_space import resolve_flat_config

    config = resolve_flat_config(
        {
            "num_models": 1,
            "model_1_type": "XGBoost",
            "use_meta_neutralization": True,
            "meta_neutralization_proportion": 0.7,
        }
    )

    assert config["meta_neutralize_proportion"] == 0.7


def test_hpo_objective_does_not_leak_into_inner_weight_optimization() -> None:
    from alphapulse.hpo.search_space import resolve_flat_config

    base = {
        "num_models": 2,
        "model_1_type": "XGBoost",
        "model_2_type": "LightGBM",
        "ensemble_method": "weighted",
        "hpo_objective": "payout_score",
    }
    default_inner = resolve_flat_config(base)
    explicit_inner = resolve_flat_config({**base, "ensemble_objective": "payout_score"})

    assert default_inner["ensemble_params"]["objective"] == "corr_sharpe"
    assert explicit_inner["ensemble_params"]["objective"] == "payout_score"


def test_resolve_flat_config_fast_foundation_defaults_to_pca() -> None:
    from alphapulse.hpo.search_space import (
        _torch_available,
        resolve_flat_config,
        resolve_foundation_compression,
    )

    assert resolve_foundation_compression(None, hpo_fast=True) == "pca"
    if not _torch_available():
        assert resolve_foundation_compression("autoencoder") == "pca"

    flat = {
        "hpo_fast": True,
        "num_models": 1,
        "model_1_type": "TabPFN",
        "foundation_max_train_rows": 5000,
        "foundation_n_components": 128,
        "foundation_n_estimators": 2,
        "foundation_compression_epochs": 5,
        "scaler_type": "StandardScaler",
        "use_packboost": False,
        "ensemble_method": "single",
    }
    cfg = resolve_flat_config(flat)
    params = cfg["models"][0]["params"]
    assert params["compression"] == "pca"
    assert params["n_estimators"] == 2
    assert params["compression_epochs"] == 5
    assert cfg["neutralize_proportion"] == 0.0


def test_resolve_flat_config_boosting_respects_neutralization_flag() -> None:
    from alphapulse.hpo.search_space import (
        MIN_NEUTRALIZATION_PROPORTION,
        resolve_flat_config,
    )

    flat = {
        "num_models": 1,
        "model_1_type": "XGBoost",
        "use_neutralization": False,
        "neutralization_proportion": 0.05,
        "scaler_type": "StandardScaler",
        "use_packboost": False,
        "ensemble_method": "single",
    }
    cfg = resolve_flat_config(flat)
    assert cfg["neutralize_proportion"] == 0.0

    flat["use_neutralization"] = True
    cfg = resolve_flat_config(flat)
    assert cfg["neutralize_proportion"] >= MIN_NEUTRALIZATION_PROPORTION


def test_resolve_flat_config_mixed_ensemble_uses_neutralization() -> None:
    from alphapulse.hpo.search_space import (
        MIN_NEUTRALIZATION_PROPORTION,
        resolve_flat_config,
    )

    flat = {
        "num_models": 2,
        "model_1_type": "XGBoost",
        "model_2_type": "TabPFN",
        "use_neutralization": True,
        "neutralization_proportion": 0.2,
        "scaler_type": "StandardScaler",
        "use_packboost": False,
        "ensemble_method": "weighted",
    }
    cfg = resolve_flat_config(flat)
    assert cfg["neutralize_proportion"] >= MIN_NEUTRALIZATION_PROPORTION
    assert cfg["ensemble_params"].get("optimize_weights") is True
    assert cfg["ensemble_params"].get("min_weight") == 0.05
    assert cfg["ensemble_params"].get("max_weight") == 0.90


def test_resolve_flat_config_weighted_uses_saved_ensemble_weights() -> None:
    from alphapulse.hpo.search_space import resolve_flat_config

    flat = {
        "num_models": 2,
        "model_1_type": "XGBoost",
        "model_2_type": "LightGBM",
        "scaler_type": "StandardScaler",
        "use_packboost": False,
        "ensemble_method": "weighted",
        "ensemble_weights": [0.85, 0.15],
    }
    cfg = resolve_flat_config(flat)
    assert cfg["ensemble_params"]["weights"] == [0.85, 0.15]
    assert "optimize_weights" not in cfg["ensemble_params"]


def test_weighted_foundation_only_ensemble_has_feasible_bounds() -> None:
    from alphapulse.hpo.search_space import resolve_flat_config
    from alphapulse.pipeline.ensemble_optimizer import validate_weight_bounds_list

    flat = {
        "num_models": 2,
        "model_1_type": "TabPFN",
        "model_2_type": "TabICL",
        "scaler_type": "RobustScaler",
        "ensemble_method": "weighted",
    }
    cfg = resolve_flat_config(flat)
    params = cfg["ensemble_params"]
    validate_weight_bounds_list(params["min_weights"], params["max_weights"])


def test_sample_random_config_boosting_enables_neutralization() -> None:
    foundation_types = {"TabPFN", "TabICL", "TabPFN3"}
    for seed in range(20):
        config = sample_random_config(seed=seed, fast=True)
        types = [
            config.get("model_1_type"),
            config.get("model_2_type"),
            config.get("model_3_type"),
        ][: config["num_models"]]
        if all(t in foundation_types for t in types):
            assert config["neutralization_proportion"] == 0.0
            assert config["use_neutralization"] is False
        else:
            assert config["use_neutralization"] is True
            assert config["neutralization_proportion"] >= 0.15


def test_resolve_flat_config_skips_augmenter_model_type() -> None:
    from alphapulse.hpo.search_space import resolve_flat_config

    flat = {
        "num_models": 2,
        "model_1_type": "SyntheticDataAugmenter",
        "model_2_type": "XGBoost",
        "scaler_type": "StandardScaler",
        "use_packboost": False,
        "ensemble_method": "single",
        "xgb_n_rounds": 100,
        "xgb_early_stopping": 10,
    }
    cfg = resolve_flat_config(flat)
    assert len(cfg["models"]) == 1
    assert cfg["models"][0]["type"] == "XGBoost"


def test_sample_random_config_returns_dict() -> None:
    config = sample_random_config(seed=42)
    assert isinstance(config, dict)
    assert "num_models" in config
    assert "scaler_type" in config


def test_lightgbm_uses_lgbm_rounds() -> None:
    from alphapulse.hpo.search_space import get_train_kwargs_from_flat

    flat = {
        "num_models": 1,
        "model_1_type": "LightGBM",
        "lgbm_n_rounds": 777,
        "lgbm_early_stopping": 33,
    }
    kw = get_train_kwargs_from_flat(flat)
    assert kw["n_rounds"] == 777
    assert kw["early_stopping_rounds"] == 33


def test_mixed_ensemble_preserves_per_model_training_budgets() -> None:
    from alphapulse.hpo.search_space import get_train_kwargs_from_flat

    flat = {
        "num_models": 2,
        "model_1_type": "LightGBM",
        "model_2_type": "XGBoost",
        "lgbm_n_rounds": 600,
        "lgbm_early_stopping": 30,
        "xgb_n_rounds": 150,
        "xgb_early_stopping": 20,
    }

    kwargs = get_train_kwargs_from_flat(flat)

    assert kwargs["model_train_kwargs_by_index"] == [
        {"n_rounds": 600, "early_stopping_rounds": 30},
        {"n_rounds": 150, "early_stopping_rounds": 20},
    ]


def test_augmentation_aligns_xy_index() -> None:
    from alphapulse.hpo.objective import _apply_synthetic_augmentation, _fit_pipeline
    from alphapulse.hpo.search_space import (
        get_train_kwargs_from_flat,
        resolve_flat_config,
    )

    rng = np.random.default_rng(7)
    rows = 2000
    X = pd.DataFrame(
        rng.standard_normal((rows, 12)), columns=[f"f{i}" for i in range(12)]
    )
    X.index = rng.integers(1_000_000, 9_000_000, size=rows)
    X["era"] = np.repeat([f"era_{i:04d}" for i in range(80)], rows // 80)
    y = pd.Series(rng.standard_normal(rows), index=X.index, name="target")
    feature_cols = [f"f{i}" for i in range(12)]
    flat = {
        "num_models": 1,
        "model_1_type": "XGBoost",
        "use_augmentation": True,
        "augmenter_top_fraction": 0.1,
        "augmenter_n_synthetic": 200,
        "scaler_type": "StandardScaler",
        "use_packboost": False,
        "ensemble_method": "single",
        "n_subs": 3,
        "xgb_max_depth": 3,
        "xgb_learning_rate": 0.05,
        "xgb_n_rounds": 20,
        "xgb_early_stopping": 5,
    }
    X_aug, y_aug = _apply_synthetic_augmentation(X, y, flat, feature_cols, seed=7)
    assert X_aug.index.equals(y_aug.index)
    assert len(X_aug) == len(y_aug)
    cfg = resolve_flat_config(flat)
    kw = get_train_kwargs_from_flat(flat)
    _fit_pipeline(cfg, feature_cols, X, y, kw, flat_config=flat, seed=7)


def test_xgb_defaults_when_type_xgb() -> None:
    from alphapulse.hpo.search_space import get_train_kwargs_from_flat

    flat = {"num_models": 1, "model_1_type": "XGBoost", "xgb_n_rounds": 400}
    kw = get_train_kwargs_from_flat(flat)
    assert kw["n_rounds"] == 400


def test_multi_blend_packboost_multihead_forwards_era(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alphapulse.features.catalog import load_feature_catalog
    from alphapulse.hpo.feature_routing import (
        merge_routing_into_pipeline_config,
        resolve_feature_routing,
    )
    from alphapulse.hpo.objective import _fit_pipeline
    from alphapulse.hpo.search_space import (
        get_train_kwargs_from_flat,
        resolve_flat_config,
    )
    from alphapulse.models.foundation_models import TabPFNModel
    from alphapulse.models.packboost_model import PackboostModel
    from alphapulse.pipeline.multi_target import MultiTargetPipeline

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    feature_cols = [f"f_{i}" for i in range(12)]
    payload = {
        "feature_sets": {
            "medium": feature_cols[:8],
            "agility": feature_cols[4:10],
            "dexterity": feature_cols[6:12],
            "serenity": feature_cols[2:8],
        },
        "targets": ["target", "target_alpha_20"],
    }
    (data_dir / "features.json").write_text(json.dumps(payload), encoding="utf-8")
    catalog = load_feature_catalog(data_dir)

    flat = {
        "num_models": 2,
        "model_1_type": "TabPFN",
        "model_2_type": "Packboost",
        "target_mode": "multi_blend",
        "primary_target": "target",
        "auxiliary_targets": ["target_alpha_20"],
        "target_blend_method": "equal",
        "foundation_max_train_rows": 200,
        "foundation_compression": "pca",
        "foundation_n_components": 4,
        "scaler_type": "RobustScaler",
        "use_packboost": False,
        "ensemble_method": "single",
        "use_feature_routing": True,
        "active_groups": ["medium", "agility", "dexterity", "serenity"],
        "model_1_groups": ["medium", "serenity"],
        "model_2_groups": ["agility", "dexterity"],
        "model_1_lane": 0,
        "model_2_lane": 0,
        "lane_0_steps": [],
        "hpo_fast": True,
        "packboost_model_n_worst_eras": 2,
        "packboost_model_boost_weight": 0.3,
        "packboost_model_n_rounds_base": 20,
        "packboost_model_n_rounds_boost": 10,
    }
    routing = resolve_feature_routing(flat, catalog)
    cfg = merge_routing_into_pipeline_config(resolve_flat_config(flat), routing)

    def fast_tabpfn_train(
        self: TabPFNModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        self.is_trained = True
        self.model = type("M", (), {"predict": lambda _s, x: np.zeros(len(x))})()
        return {}

    packboost_training_eras: list[pd.Series] = []

    def fast_packboost_train(
        self: PackboostModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        assert "era" in X_train.columns
        packboost_training_eras.append(X_train["era"].copy())
        self.is_trained = True
        return {}

    def fast_packboost_predict(
        self: PackboostModel,
        X: pd.DataFrame,
    ) -> np.ndarray:
        assert self.is_trained
        return np.zeros(len(X))

    monkeypatch.setattr(TabPFNModel, "train", fast_tabpfn_train)
    monkeypatch.setattr(PackboostModel, "train", fast_packboost_train)
    monkeypatch.setattr(PackboostModel, "predict", fast_packboost_predict)

    rng = np.random.default_rng(0)
    n_eras = 30
    rows_per_era = 10
    n = n_eras * rows_per_era
    cols = routing.feature_columns
    X = pd.DataFrame(rng.standard_normal((n, len(cols))), columns=cols)
    X["era"] = np.repeat([f"era_{i:04d}" for i in range(n_eras)], rows_per_era)
    targets = pd.DataFrame(
        {
            "target": pd.Series(rng.standard_normal(n)),
            "target_alpha_20": pd.Series(rng.standard_normal(n)),
        }
    )
    pipeline = _fit_pipeline(
        cfg,
        cols,
        X,
        targets["target"],
        get_train_kwargs_from_flat(flat),
        flat_config=flat,
        seed=42,
        feature_groups=routing.feature_groups,
        targets_df=targets,
    )
    assert isinstance(pipeline, MultiTargetPipeline)
    assert packboost_training_eras
    assert all(eras.notna().all() for eras in packboost_training_eras)
    preds = pipeline.predict(X.drop(columns=["era"]))
    assert preds.shape == (n,)
    assert np.all(np.isfinite(preds))


def test_load_diagnostics_train_data_multi_blend(tmp_path: Path) -> None:
    from scripts.hpo_pipeline import _load_diagnostics_train_data

    from alphapulse.hpo.objective import _fit_pipeline
    from alphapulse.hpo.search_space import (
        get_train_kwargs_from_flat,
        resolve_flat_config,
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    feature_cols = [f"f_{i}" for i in range(6)]
    payload = {
        "feature_sets": {"medium": feature_cols},
        "targets": ["target", "target_alpha_20"],
    }
    (data_dir / "features.json").write_text(json.dumps(payload), encoding="utf-8")

    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(rng.standard_normal((n, len(feature_cols))), columns=feature_cols)
    df["era"] = np.repeat([f"era_{i:04d}" for i in range(8)], 10)
    df["target"] = rng.standard_normal(n)
    df["target_alpha_20"] = rng.standard_normal(n)
    df.to_parquet(data_dir / "train.parquet")

    best_config = {
        "num_models": 1,
        "model_1_type": "XGBoost",
        "target_mode": "multi_blend",
        "primary_target": "target",
        "auxiliary_targets": ["target_alpha_20"],
        "target_blend_method": "equal",
        "scaler_type": "StandardScaler",
        "ensemble_method": "single",
        "use_feature_routing": False,
        "xgb_n_rounds": 5,
        "xgb_early_stopping": 3,
        "xgb_max_depth": 2,
        "xgb_learning_rate": 0.1,
    }
    X_train, y_train, targets_df, feat_cols, feature_groups = (
        _load_diagnostics_train_data(
            best_config,
            data_dir,
            train_subsample=1.0,
            target_col="target",
            seed=42,
        )
    )
    assert targets_df is not None
    assert "target_alpha_20" in targets_df.columns

    era_train = X_train["era"]
    eras_sorted = sorted(era_train.unique(), key=str)
    train_mask = ~era_train.isin(set(eras_sorted[-2:]))
    targets_train = targets_df.loc[train_mask]

    pipeline = _fit_pipeline(
        resolve_flat_config(best_config),
        feat_cols,
        X_train.loc[train_mask],
        y_train.loc[train_mask],
        get_train_kwargs_from_flat(best_config),
        flat_config=best_config,
        seed=42,
        feature_groups=feature_groups or None,
        targets_df=targets_train,
    )
    preds = pipeline.predict(X_train.drop(columns=["era"]).iloc[:5])
    assert len(preds) == 5
    assert np.all(np.isfinite(preds))


def test_persistable_flat_config_strips_runtime_keys() -> None:
    from scripts.hpo_pipeline import _persistable_flat_config

    flat = {
        "model_1_type": "XGBoost",
        "_data_dir": "/tmp/data",
        "_train_subsample": 0.125,
        "log_wandb_diagnostics": True,
        "target_mode": "single",
    }
    persisted = _persistable_flat_config(flat)
    assert persisted == {"model_1_type": "XGBoost", "target_mode": "single"}


def test_multi_target_loader_returns_unique_aligned_row_index(tmp_path: Path) -> None:
    from alphapulse.experiments.data import load_train_targets_frame

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "era": ["era1", "era1", "era2", "era2"],
            "target": [0.1, 0.2, 0.3, 0.4],
            "target_aux": [0.4, 0.3, 0.2, 0.1],
        },
        index=["duplicate", "duplicate", "row3", "row4"],
    )
    train.to_parquet(data_dir / "train.parquet")

    X, y, targets, features = load_train_targets_frame(
        data_dir,
        train_subsample=1.0,
        primary_target="target",
        auxiliary_targets=["target_aux"],
        seed=42,
        feature_columns=["feature_a"],
        need_era=True,
    )

    assert not X.index.has_duplicates
    assert X.index.equals(y.index)
    assert X.index.equals(targets.index)
    assert features == ["feature_a"]


def test_parquet_loader_restores_id_without_dataframe_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alphapulse.experiments.data import read_parquet_frame

    path = tmp_path / "frame.parquet"
    pd.DataFrame({"id": ["a", "b"], "feature_a": [1, 2]}).to_parquet(path)

    def reject_set_index(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("set_index would copy the complete feature frame")

    monkeypatch.setattr(pd.DataFrame, "set_index", reject_set_index)
    frame = read_parquet_frame(path, columns=["feature_a"])

    assert frame.index.tolist() == ["a", "b"]
    assert frame.index.name == "id"
    assert frame.columns.tolist() == ["feature_a"]


def test_parquet_loader_retries_transient_arrow_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alphapulse.experiments import data as experiment_data

    path = tmp_path / "frame.parquet"
    pd.DataFrame({"id": ["a", "b"], "feature_a": [1, 2]}).to_parquet(path)
    original = experiment_data.pq.ParquetFile
    attempts = 0

    def flaky_parquet_file(path_arg: Path) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise experiment_data.pa.ArrowInvalid("transient snappy decoder failure")
        return original(path_arg)

    monkeypatch.setattr(experiment_data.pq, "ParquetFile", flaky_parquet_file)

    frame = experiment_data.read_parquet_frame(path, columns=["feature_a"])

    assert attempts == 2
    assert frame.index.tolist() == ["a", "b"]


def test_worker_wandb_requires_diagnostics() -> None:
    from scripts.hpo_pipeline import _worker_wandb_enabled

    assert not _worker_wandb_enabled(
        project="project",
        group="group",
        trial_number=3,
        diagnostics=False,
    )
    assert _worker_wandb_enabled(
        project="project",
        group="group",
        trial_number=3,
        diagnostics=True,
    )
