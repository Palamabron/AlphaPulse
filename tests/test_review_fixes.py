"""Tests for code-review fixes."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphapulse.autoresearch import mutations
from alphapulse.autoresearch.state import ResearchState, TrialRecord
from alphapulse.experiments.split import internal_val_split
from alphapulse.hpo.search_space import get_train_kwargs_from_flat
from alphapulse.models import XGBoostModel
from alphapulse.models.era_ensemble_model import EraEnsembleModel
from alphapulse.pipeline import Pipeline
from alphapulse.pipeline.ensemble import EnsembleStrategy
from alphapulse.preprocessors import PCAPreprocessor, StandardScalerPreprocessor


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(0)
    n = 200
    X = pd.DataFrame(
        np.random.randn(n, 4).astype(np.float64),
        columns=["a", "b", "c", "d"],
    )
    y = pd.Series(X["a"] * 0.5 + np.random.randn(n) * 0.2)
    return X, y


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


class TestEnsembleStackingGuard:
    def test_unfitted_stacking_raises(self) -> None:
        es = EnsembleStrategy(method="stacking")
        preds = np.random.randn(10, 2)
        with pytest.raises(RuntimeError, match="meta-learner is not fitted"):
            es.combine(preds)


class TestTrainKwargsFromFlat:
    def test_lightgbm_uses_lgbm_rounds(self) -> None:
        flat = {
            "num_models": 1,
            "model_1_type": "LightGBM",
            "lgbm_n_rounds": 777,
            "lgbm_early_stopping": 33,
        }
        kw = get_train_kwargs_from_flat(flat)
        assert kw["n_rounds"] == 777
        assert kw["early_stopping_rounds"] == 33

    def test_xgb_defaults_when_type_xgb(self) -> None:
        flat = {"num_models": 1, "model_1_type": "XGBoost", "xgb_n_rounds": 400}
        kw = get_train_kwargs_from_flat(flat)
        assert kw["n_rounds"] == 400


class TestInternalValSplit:
    def test_era_aware_split_uses_last_eras(self) -> None:
        n_eras = 20
        rows_per_era = 5
        X = pd.DataFrame(
            np.random.randn(n_eras * rows_per_era, 2),
            columns=["a", "b"],
        )
        y = pd.Series(np.random.randn(len(X)))
        era = pd.Series(np.repeat([f"e{i:03d}" for i in range(n_eras)], rows_per_era))
        X_tr, y_tr, X_va, y_va = internal_val_split(
            X, y, era_train=era, force_internal=True
        )
        assert len(X_tr) + len(X_va) == len(X)
        assert len(X_va) > 0
        val_eras = set(era.loc[X_va.index])
        train_eras = set(era.loc[X_tr.index])
        assert val_eras.isdisjoint(train_eras)

    def test_stacking_forces_internal_split_on_small_train(self) -> None:
        X = pd.DataFrame({"a": np.arange(100.0)})
        y = pd.Series(np.arange(100.0))
        era = pd.Series(np.repeat([f"e{i}" for i in range(10)], 10))
        _, _, X_va, _ = internal_val_split(X, y, era_train=era, force_internal=True)
        assert X_va is not None
        assert len(X_va) > 0


class TestPipelineSaveLoad:
    def test_roundtrip(
        self, toy_data: tuple[pd.DataFrame, pd.Series], tmp_path: Path
    ) -> None:
        X, y = toy_data
        pipe = Pipeline(
            preprocessors=[StandardScalerPreprocessor()],
            model=_xgb(),
        )
        pipe.fit(X, y, n_rounds=5)
        path = tmp_path / "pipe.pkl"
        pipe.save_pipeline(path)
        loaded = Pipeline.load_pipeline(path)
        np.testing.assert_allclose(loaded.predict(X), pipe.predict(X))


class TestEraPreservation:
    def test_era_survives_pca_for_era_ensemble(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = toy_data
        era = pd.Series(np.repeat([f"e{i:03d}" for i in range(20)], 10), index=X.index)
        X = X.assign(era=era)

        def factory() -> XGBoostModel:
            return _xgb("sub")

        model = EraEnsembleModel(base_model_factory=factory, n_subs=4)
        pipe = Pipeline(
            preprocessors=[
                StandardScalerPreprocessor(),
                PCAPreprocessor(n_components=2),
            ],
            model=model,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipe.fit(X, y, n_rounds=5)
        degrade_warnings = [w for w in caught if "falling back" in str(w.message)]
        assert not degrade_warnings
        assert len(model._sub_models) > 1


class TestEmptyRowPredict:
    def test_all_nan_rows_returns_finite(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = toy_data
        pipe = Pipeline(
            preprocessors=[StandardScalerPreprocessor()],
            model=_xgb(),
        )
        pipe.fit(X, y, n_rounds=5)
        X_bad = X.copy()
        X_bad[:] = np.nan
        preds = pipe.predict(X_bad)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()


class TestNumeraiPredictEras:
    def test_passes_eras_when_neutralization_enabled(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
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
        predict_fn = pipe.to_numerai_predict()
        out = predict_fn(X, pd.DataFrame())
        assert out.shape == (len(X), 1)


class TestMutations:
    def _base_config(self) -> dict[str, Any]:
        return {
            "preprocessors": [{"type": "StandardScaler", "params": {}}],
            "models": [{"type": "XGBoost", "params": {"max_depth": 3}}],
            "ensemble_method": "single",
            "ensemble_params": {},
        }

    def test_add_model_switches_to_weighted(self) -> None:
        cfg = mutations.add_model(self._base_config(), "LightGBM", {})
        assert len(cfg["models"]) == 2
        assert cfg["ensemble_method"] == "weighted"

    def test_remove_only_model_raises(self) -> None:
        with pytest.raises(ValueError, match="only model"):
            mutations.remove_model(self._base_config(), 0)

    def test_set_neutralization_bounds(self) -> None:
        with pytest.raises(ValueError, match="proportion"):
            mutations.set_neutralization(self._base_config(), 1.5)


class TestResearchState:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        state = ResearchState(
            trials=[
                TrialRecord(
                    trial_number=0,
                    sharpe=1.0,
                    metrics={"sharpe": 1.0},
                    config={"models": []},
                    model_types=["XGBoost"],
                    elapsed_seconds=1.0,
                    action_taken="initial",
                    agent_reasoning="test",
                )
            ],
            current_config={"models": []},
        )
        path = tmp_path / "state.json"
        state.save(path)
        loaded = ResearchState.load(path)
        assert len(loaded.trials) == 1
        assert loaded.trials[0].sharpe == 1.0
        assert loaded.current_config == {"models": []}
