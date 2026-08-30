import numpy as np
import pandas as pd
import pytest

from alphapulse.models import XGBoostModel
from alphapulse.pipeline import Pipeline
from alphapulse.pipeline.ensemble import EnsembleStrategy
from alphapulse.pipeline.ensemble_optimizer import (
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MIN_WEIGHT,
    EnsembleOptimizer,
    project_weights_to_bounds,
    project_weights_to_bounds_list,
    validate_weight_bounds,
    validate_weight_bounds_list,
)


def test_validate_weight_bounds_infeasible() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        validate_weight_bounds(2, min_weight=0.6, max_weight=0.9)


def test_project_weights_respects_bounds() -> None:
    w = project_weights_to_bounds(
        np.array([0.99, 0.01]), min_weight=0.05, max_weight=0.90
    )
    assert w.sum() == pytest.approx(1.0)
    assert all(DEFAULT_MIN_WEIGHT <= x <= DEFAULT_MAX_WEIGHT for x in w)


def test_ensemble_optimizer_bounded_weights_stay_in_box() -> None:
    rng = np.random.RandomState(0)
    n = 200
    eras = pd.Series(np.repeat(["e1", "e2", "e3", "e4"], n // 4))
    y = rng.randn(n)
    oof = np.column_stack([rng.randn(n), y + rng.randn(n) * 0.05])

    bounded = EnsembleOptimizer(
        seed=0, min_weight=0.05, max_weight=0.90, objective="corr_sharpe"
    )
    bounded.fit(oof, y, eras)
    assert bounded.weights_ is not None
    assert bounded.weights_.sum() == pytest.approx(1.0)
    assert all(0.05 <= float(w) <= 0.90 for w in bounded.weights_)


def test_payout_weight_optimization_requires_meta_model() -> None:
    optimizer = EnsembleOptimizer(objective="payout_score")
    matrix = np.ones((20, 2))

    with pytest.raises(ValueError, match="requires aligned meta-model"):
        optimizer.fit(
            matrix,
            np.ones(20),
            pd.Series(["era"] * 20),
        )


def test_pipeline_optimize_weights_post_fit() -> None:
    rng = np.random.RandomState(1)
    n = 160
    eras = np.repeat([f"e{i}" for i in range(8)], n // 8)
    X = pd.DataFrame(rng.randn(n, 3), columns=list("ABC"))
    X["era"] = eras
    y = pd.Series(X["A"] * 0.4 + rng.randn(n) * 0.2)

    n_train = 120
    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_val = X.iloc[n_train:]
    y_val = y.iloc[n_train:]
    era_val = X_val["era"]

    model_a = XGBoostModel(
        params={"max_depth": 2, "learning_rate": 0.1, "tree_method": "hist"},
        name="m_a",
    )
    model_b = XGBoostModel(
        params={"max_depth": 2, "learning_rate": 0.1, "tree_method": "hist"},
        name="m_b",
    )
    pipeline = Pipeline(
        preprocessors=[],
        models=[model_a, model_b],
        ensemble_method="weighted",
        ensemble_params={
            "optimize_weights": True,
            "objective": "corr_sharpe",
            "min_weight": 0.05,
            "max_weight": 0.90,
        },
    )
    pipeline.fit(
        X_train.drop(columns=["era"]),
        y_train,
        X_val=X_val.drop(columns=["era"]),
        y_val=y_val,
        era_val=era_val,
        n_rounds=30,
    )
    weights = pipeline.ensemble_weights
    assert weights is not None
    assert len(weights) == 2
    assert sum(weights) == pytest.approx(1.0)
    assert all(0.05 <= w <= 0.90 for w in weights)


def test_ensemble_optimizer_per_model_bounds() -> None:
    rng = np.random.RandomState(3)
    n = 200
    eras = pd.Series(np.repeat(["e1", "e2", "e3", "e4"], n // 4))
    y = rng.randn(n)
    oof = np.column_stack([rng.randn(n), y + rng.randn(n) * 0.05])

    optimizer = EnsembleOptimizer(seed=0, objective="corr_sharpe")
    optimizer.fit(
        oof,
        y,
        eras,
        min_weights=[0.05, 0.05],
        max_weights=[0.35, 0.90],
    )
    assert optimizer.weights_ is not None
    assert optimizer.weights_[0] <= 0.35 + 1e-6
    assert optimizer.weights_[1] <= 0.90 + 1e-6


def test_project_weights_to_bounds_list() -> None:
    w = project_weights_to_bounds_list(
        np.array([0.9, 0.1]),
        [0.05, 0.05],
        [0.35, 0.90],
    )
    validate_weight_bounds_list([0.05, 0.05], [0.35, 0.90])
    assert w.sum() == pytest.approx(1.0)
    assert w[0] <= 0.35 + 1e-6


def test_ensemble_strategy_fixed_weights_skip_optimizer() -> None:
    strategy = EnsembleStrategy(
        method="weighted",
        params={"weights": [0.9, 0.1]},
    )
    strategy.fit(n_models=2)
    assert strategy.weights is not None
    assert strategy.weights[0] == pytest.approx(0.9)


def test_ensemble_strategy_rejects_optimize_and_fixed_weights() -> None:
    strategy = EnsembleStrategy(
        method="weighted",
        params={"optimize_weights": True, "weights": [0.9, 0.1]},
    )
    with pytest.raises(ValueError, match="both optimize_weights and fixed weights"):
        strategy.fit(
            n_models=2,
            get_val_predictions=lambda: np.zeros((10, 2)),
            y_val=pd.Series(np.zeros(10)),
            eras_val=pd.Series(["e"] * 10),
        )
