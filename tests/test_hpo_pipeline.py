"""Tests for HPO pipeline (run_trial with preloaded data)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphapulse.hpo import run_trial, sample_random_config


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


def test_run_trial_returns_metrics(
    toy_data_with_era: dict[str, Any], minimal_flat_config: dict[str, Any]
) -> None:
    metrics = run_trial(minimal_flat_config, **toy_data_with_era)
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
    assert "SyntheticDataAugmenter" not in (
        config.get("model_1_type"),
        config.get("model_2_type"),
        config.get("model_3_type"),
    )


def test_resolve_flat_config_fast_foundation_uses_autoencoder() -> None:
    from alphapulse.hpo.search_space import resolve_flat_config

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
    assert params["compression"] == "autoencoder"
    assert params["n_estimators"] == 2
    assert params["compression_epochs"] == 5
    assert cfg["neutralize_proportion"] == 0.0


def test_resolve_flat_config_boosting_uses_neutralization() -> None:
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
    assert cfg["neutralize_proportion"] >= MIN_NEUTRALIZATION_PROPORTION


def test_sample_random_config_boosting_enables_neutralization() -> None:
    for seed in range(20):
        config = sample_random_config(seed=seed, fast=True)
        types = [
            config.get("model_1_type"),
            config.get("model_2_type"),
            config.get("model_3_type"),
        ][: config["num_models"]]
        if "TabPFN" in types or "TabICL" in types:
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
