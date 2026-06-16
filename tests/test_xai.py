"""Tests for universal feature importance extraction (XAI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphapulse.evaluation.shap_report import (
    compute_universal_feature_importance,
    log_universal_feature_importance,
)
from alphapulse.models.sklearn_models import RandomForestModel
from alphapulse.models.xgboost_model import XGBoostModel
from alphapulse.pipeline.pipeline import Pipeline
from alphapulse.preprocessors.base import BasePreprocessor

N_ROWS = 200
N_FEATURES = 10
FEATURE_COLS = [f"f{i}" for i in range(N_FEATURES)]


class _IdentityPreprocessor(BasePreprocessor):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> _IdentityPreprocessor:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


def _make_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(N_ROWS, N_FEATURES), columns=FEATURE_COLS)
    y = pd.Series(rng.randn(N_ROWS), name="target")
    return X, y


def _xgb_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    model = XGBoostModel()
    pipe = Pipeline(preprocessors=[_IdentityPreprocessor()], model=model)
    pipe.fit(X, y, n_rounds=10, early_stopping_rounds=5)
    return pipe


def _lgbm_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pytest.importorskip("lightgbm")
    from alphapulse.models.lightgbm_model import LightGBMModel

    model = LightGBMModel(
        params={
            "objective": "regression",
            "metric": "rmse",
            "max_depth": 3,
            "learning_rate": 0.1,
            "num_leaves": 8,
            "min_child_samples": 5,
            "verbosity": -1,
            "n_jobs": 1,
        },
        n_estimators=20,
    )
    pipe = Pipeline(preprocessors=[_IdentityPreprocessor()], model=model)
    pipe.fit(X, y, n_rounds=20)
    return pipe


def _catboost_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pytest.importorskip("catboost")
    from alphapulse.models.catboost_model import CatBoostModel

    model = CatBoostModel(iterations=20)
    pipe = Pipeline(preprocessors=[_IdentityPreprocessor()], model=model)
    pipe.fit(X, y, n_rounds=20)
    return pipe


def _rf_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    model = RandomForestModel(
        params={"n_estimators": 10, "n_jobs": 1, "random_state": 0}
    )
    pipe = Pipeline(preprocessors=[_IdentityPreprocessor()], model=model)
    pipe.fit(X, y)
    return pipe


def _ridge_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    from alphapulse.models.sklearn_models import RidgeModel

    model = RidgeModel()
    pipe = Pipeline(preprocessors=[_IdentityPreprocessor()], model=model)
    pipe.fit(X, y)
    return pipe


class TestComputeUniversalFeatureImportance:
    def test_xgboost_returns_nonempty_dict(self) -> None:
        X, y = _make_data()
        pipe = _xgb_pipeline(X, y)
        imp, label = compute_universal_feature_importance(
            pipe, X, feature_cols=FEATURE_COLS, top_n=5
        )
        assert len(imp) > 0
        assert len(imp) <= 5
        assert label == "XGBoost"

    def test_xgboost_scores_are_finite_and_nonneg(self) -> None:
        X, y = _make_data()
        pipe = _xgb_pipeline(X, y)
        imp, _ = compute_universal_feature_importance(
            pipe, X, feature_cols=FEATURE_COLS
        )
        assert all(np.isfinite(v) and v >= 0 for v in imp.values())

    def test_lgbm_returns_nonempty_dict(self) -> None:
        X, y = _make_data()
        pipe = _lgbm_pipeline(X, y)
        imp, label = compute_universal_feature_importance(
            pipe, X, feature_cols=FEATURE_COLS, top_n=5
        )
        assert len(imp) > 0
        assert "LightGBM" in label

    def test_catboost_returns_nonempty_dict(self) -> None:
        X, y = _make_data()
        pipe = _catboost_pipeline(X, y)
        imp, label = compute_universal_feature_importance(
            pipe, X, feature_cols=FEATURE_COLS, top_n=5
        )
        assert len(imp) > 0
        assert "CatBoost" in label

    def test_random_forest_returns_nonempty_dict(self) -> None:
        X, y = _make_data()
        pipe = _rf_pipeline(X, y)
        imp, label = compute_universal_feature_importance(
            pipe, X, feature_cols=FEATURE_COLS, top_n=5
        )
        assert len(imp) > 0
        assert "SklearnTree" in label

    def test_ridge_only_returns_empty(self) -> None:
        X, y = _make_data()
        pipe = _ridge_pipeline(X, y)
        imp, label = compute_universal_feature_importance(
            pipe, X, feature_cols=FEATURE_COLS
        )
        assert imp == {}
        assert label == "none"

    def test_top_n_limits_result_size(self) -> None:
        X, y = _make_data()
        pipe = _xgb_pipeline(X, y)
        for top_n in (3, 7, N_FEATURES):
            imp, _ = compute_universal_feature_importance(
                pipe, X, feature_cols=FEATURE_COLS, top_n=top_n
            )
            assert len(imp) <= top_n

    def test_result_sorted_descending(self) -> None:
        X, y = _make_data()
        pipe = _xgb_pipeline(X, y)
        imp, _ = compute_universal_feature_importance(
            pipe, X, feature_cols=FEATURE_COLS
        )
        scores = list(imp.values())
        assert scores == sorted(scores, reverse=True)

    def test_keys_are_subset_of_feature_cols(self) -> None:
        X, y = _make_data()
        pipe = _xgb_pipeline(X, y)
        imp, _ = compute_universal_feature_importance(
            pipe, X, feature_cols=FEATURE_COLS
        )
        assert set(imp.keys()).issubset(set(FEATURE_COLS))

    def test_mixed_xgb_rf_pipeline_nonempty(self) -> None:
        X, y = _make_data()
        xgb_model = XGBoostModel(name="xgb1")
        rf_model = RandomForestModel(
            params={"n_estimators": 5, "n_jobs": 1, "random_state": 0}, name="rf1"
        )
        pipe = Pipeline(
            preprocessors=[_IdentityPreprocessor()],
            models=[xgb_model, rf_model],
            ensemble_method="weighted",
            ensemble_params={"weights": [0.5, 0.5]},
        )
        pipe.fit(X, y, n_rounds=5)
        imp, label = compute_universal_feature_importance(
            pipe, X, feature_cols=FEATURE_COLS
        )
        assert len(imp) > 0
        assert "XGBoost" in label
        assert "SklearnTree" in label


class TestLogUniversalFeatureImportance:
    def test_returns_empty_when_wandb_inactive(self) -> None:
        X, y = _make_data()
        pipe = _xgb_pipeline(X, y)
        result = log_universal_feature_importance(pipe, X, feature_cols=FEATURE_COLS)
        assert result == {}


def test_log_wandb_figure_accepts_matplotlib_figure() -> None:
    from unittest.mock import MagicMock

    import numpy as np

    from alphapulse.evaluation.wandb_diagnostics import (
        _log_wandb_figure,
        _new_figure,
    )

    fig = _new_figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    mock_wandb = MagicMock()
    _log_wandb_figure(mock_wandb, "diagnostics/test", fig)
    mock_wandb.log.assert_called_once()
    mock_wandb.Image.assert_called_once()
