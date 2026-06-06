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


def test_xgb_defaults_when_type_xgb() -> None:
    from alphapulse.hpo.search_space import get_train_kwargs_from_flat

    flat = {"num_models": 1, "model_1_type": "XGBoost", "xgb_n_rounds": 400}
    kw = get_train_kwargs_from_flat(flat)
    assert kw["n_rounds"] == 400
