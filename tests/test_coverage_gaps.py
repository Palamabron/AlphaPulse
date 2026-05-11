"""Tests covering previously uncovered code paths."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphapulse.experiments.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentV1,
    FeatureConfig,
    ModelSpec,
    PreprocessorStep,
    TrainConfig,
)
from alphapulse.hpo.builder import build_pipeline_or_multi, build_preprocessors
from alphapulse.hpo.search_space import resolve_flat_config
from alphapulse.models import XGBoostModel
from alphapulse.pipeline.ensemble import EnsembleStrategy
from alphapulse.pipeline.multihead import HeadSpec, MultiHeadPipeline
from alphapulse.preprocessors import StandardScalerPreprocessor
from alphapulse.preprocessors.grouped import GroupedPreprocessor
from alphapulse.validation.purged_cv import PurgedEraCV


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(0)
    n = 200
    X = pd.DataFrame(
        np.random.randn(n, 6).astype(np.float64),
        columns=["a", "b", "c", "d", "e", "f"],
    )
    y = pd.Series(X["a"] * 0.5 + X["b"] * 0.3 + np.random.randn(n) * 0.2)
    return X, y


@pytest.fixture
def toy_data_with_era(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X, y = toy_data
    era = pd.Series(np.repeat([f"era_{i:03d}" for i in range(10)], 20), index=X.index)
    return X, y, era


def _xgb(name: str = "xgb", depth: int = 2) -> XGBoostModel:
    return XGBoostModel(
        params={
            "max_depth": depth,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "objective": "reg:squarederror",
        },
        name=name,
    )


class TestMultiHeadPipeline:
    def test_single_head_fit_predict(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = toy_data
        head = HeadSpec(
            model=_xgb("h1"),
            input_columns=["a", "b", "c"],
            input_group=None,
            local_preprocessors=[],
            feature_groups={},
        )
        pipe = MultiHeadPipeline(global_preprocessors=[], heads=[head])
        pipe.fit(X, y, n_rounds=5)
        preds = pipe.predict(X)
        assert preds.shape == (len(X),)

    def test_two_heads_disjoint_groups(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = toy_data
        feature_groups = {"group_ab": ["a", "b"], "group_cd": ["c", "d"]}
        heads = [
            HeadSpec(
                model=_xgb("h1"),
                input_columns=None,
                input_group="group_ab",
                local_preprocessors=[StandardScalerPreprocessor()],
                feature_groups=feature_groups,
            ),
            HeadSpec(
                model=_xgb("h2"),
                input_columns=None,
                input_group="group_cd",
                local_preprocessors=[],
                feature_groups=feature_groups,
            ),
        ]
        pipe = MultiHeadPipeline(
            global_preprocessors=[],
            heads=heads,
            ensemble_method="weighted",
            ensemble_params={"weights": [0.5, 0.5]},
        )
        pipe.fit(X, y, n_rounds=5)
        preds = pipe.predict(X)
        assert preds.shape == (len(X),)
        assert isinstance(preds, np.ndarray)

    def test_global_preprocessor_applied(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = toy_data
        head = HeadSpec(
            model=_xgb("h1"),
            input_columns=["a", "b"],
            input_group=None,
            local_preprocessors=[],
            feature_groups={},
        )
        pipe = MultiHeadPipeline(
            global_preprocessors=[StandardScalerPreprocessor()],
            heads=[head],
        )
        pipe.fit(X, y, n_rounds=5)
        preds = pipe.predict(X)
        assert preds.shape == (len(X),)

    def test_missing_group_raises(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = toy_data
        head = HeadSpec(
            model=_xgb("h1"),
            input_columns=None,
            input_group="nonexistent_group",
            local_preprocessors=[],
            feature_groups={},
        )
        pipe = MultiHeadPipeline(global_preprocessors=[], heads=[head])
        with pytest.raises(ValueError, match="No columns from input_group"):
            pipe.fit(X, y, n_rounds=5)

    def test_to_numerai_predict(self, toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = toy_data
        head = HeadSpec(
            model=_xgb("h1"),
            input_columns=list(X.columns),
            input_group=None,
            local_preprocessors=[],
            feature_groups={},
        )
        pipe = MultiHeadPipeline(global_preprocessors=[], heads=[head])
        pipe.fit(X, y, n_rounds=5)
        predict_fn = pipe.to_numerai_predict()
        out = predict_fn(X, pd.DataFrame())
        assert isinstance(out, pd.DataFrame)
        assert "prediction" in out.columns
        assert out["prediction"].between(0.0, 1.0).all()

    def test_save_load_roundtrip(
        self, toy_data: tuple[pd.DataFrame, pd.Series], tmp_path: Path
    ) -> None:
        X, y = toy_data
        head = HeadSpec(
            model=_xgb("h1"),
            input_columns=["a", "b"],
            input_group=None,
            local_preprocessors=[],
            feature_groups={},
        )
        pipe = MultiHeadPipeline(global_preprocessors=[], heads=[head])
        pipe.fit(X, y, n_rounds=5)
        preds_before = pipe.predict(X)

        save_path = tmp_path / "multihead.pkl"
        pipe.save_pipeline(save_path)
        loaded = MultiHeadPipeline.load_pipeline(save_path)
        preds_after = loaded.predict(X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)


class TestGroupedPreprocessor:
    def test_fit_transform_shape(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = toy_data
        gpp = GroupedPreprocessor(
            groups={"grp1": ["a", "b", "c"], "grp2": ["d", "e", "f"]},
            group_preprocessors={
                "grp1": [StandardScalerPreprocessor()],
                "grp2": [StandardScalerPreprocessor()],
            },
        )
        out = gpp.fit_transform(X, y)
        assert out.shape == X.shape

    def test_column_prefix_applied(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = toy_data
        gpp = GroupedPreprocessor(
            groups={"g1": ["a", "b"], "g2": ["c", "d"]},
            group_preprocessors={"g1": [], "g2": []},
            column_prefix=True,
        )
        gpp.fit(X, y)
        out = gpp.transform(X)
        assert all(col.startswith(("g1__", "g2__")) for col in out.columns)

    def test_column_prefix_off(self, toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = toy_data
        gpp = GroupedPreprocessor(
            groups={"g1": ["a", "b"]},
            group_preprocessors={"g1": []},
            column_prefix=False,
        )
        gpp.fit(X, y)
        out = gpp.transform(X)
        assert list(out.columns) == ["a", "b"]

    def test_mismatched_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="same keys"):
            GroupedPreprocessor(
                groups={"g1": ["a"]},
                group_preprocessors={"g2": []},
            )

    def test_missing_column_raises(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = toy_data
        gpp = GroupedPreprocessor(
            groups={"g1": ["a", "nonexistent"]},
            group_preprocessors={"g1": []},
        )
        with pytest.raises(ValueError, match="missing columns"):
            gpp.fit(X, y)

    def test_transform_before_fit_raises(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, _ = toy_data
        gpp = GroupedPreprocessor(
            groups={"g1": ["a"]},
            group_preprocessors={"g1": []},
        )
        with pytest.raises(ValueError, match="not fitted"):
            gpp.transform(X)


class TestEnsembleStrategyStacking:
    def _make_val_preds(self, n: int = 100, k: int = 2) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.standard_normal((n, k))

    def test_stacking_ridge_combine(self) -> None:
        n, k = 100, 2
        rng = np.random.default_rng(1)
        val_preds = self._make_val_preds(n, k)
        y_val = pd.Series(rng.standard_normal(n))

        es = EnsembleStrategy(method="stacking", params={"meta_learner": "ridge"})
        es.fit(
            n_models=k,
            get_val_predictions=lambda: val_preds,
            y_val=y_val,
        )
        test_preds = self._make_val_preds(50, k)
        out = es.combine(test_preds)
        assert out.shape == (50,)
        assert np.isfinite(out).all()

    def test_stacking_requires_val_data(self) -> None:
        es = EnsembleStrategy(method="stacking")
        with pytest.raises(ValueError, match="Stacking requires"):
            es.fit(n_models=2, get_val_predictions=None, y_val=None)

    def test_weighted_normalizes_weights(self) -> None:
        es = EnsembleStrategy(method="weighted", params={"weights": [2.0, 2.0]})
        es.fit(n_models=2)
        assert es._weights is not None
        np.testing.assert_allclose(es._weights.sum(), 1.0)

    def test_weighted_wrong_length_raises(self) -> None:
        es = EnsembleStrategy(method="weighted", params={"weights": [0.5, 0.3, 0.2]})
        with pytest.raises(ValueError, match="weights length"):
            es.fit(n_models=2)


# ---------------------------------------------------------------------------
# build_pipeline_or_multi / resolve_flat_config
# ---------------------------------------------------------------------------


class TestBuilderAndSearchSpace:
    def _base_flat_config(self) -> dict[str, Any]:
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
            "xgb_n_rounds": 10,
            "xgb_early_stopping": 5,
            "packboost_model_n_worst_eras": 5,
            "packboost_model_boost_weight": 0.3,
            "packboost_model_n_rounds_base": 300,
            "packboost_model_n_rounds_boost": 100,
            "ensemble_method": "single",
            "stacking_meta_learner": "ridge",
        }

    def test_resolve_flat_single_xgb(self) -> None:
        cfg = resolve_flat_config(self._base_flat_config())
        assert len(cfg["models"]) == 1
        assert cfg["models"][0]["type"] == "XGBoost"
        assert cfg["ensemble_method"] == "single"
        assert len(cfg["preprocessors"]) == 1

    def test_resolve_flat_with_packboost_preprocessor(self) -> None:
        flat = self._base_flat_config()
        flat["use_packboost"] = True
        cfg = resolve_flat_config(flat)
        types = [p["type"] for p in cfg["preprocessors"]]
        assert "Packboost" in types

    def test_resolve_flat_two_models_weighted(self) -> None:
        flat = self._base_flat_config()
        flat["num_models"] = 2
        flat["ensemble_method"] = "weighted"
        cfg = resolve_flat_config(flat)
        assert len(cfg["models"]) == 2
        assert cfg["ensemble_method"] == "weighted"
        assert "weights" in cfg["ensemble_params"]

    def test_resolve_flat_one_model_forces_single_ensemble(self) -> None:
        flat = self._base_flat_config()
        flat["num_models"] = 1
        flat["ensemble_method"] = "stacking"
        cfg = resolve_flat_config(flat)
        assert cfg["ensemble_method"] == "single"

    def test_build_pipeline_or_multi_returns_pipeline(
        self, toy_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        from alphapulse.pipeline.pipeline import Pipeline

        cfg = resolve_flat_config(self._base_flat_config())
        pipe = build_pipeline_or_multi(cfg)
        assert isinstance(pipe, Pipeline)
        X, y = toy_data
        pipe.fit(X, y, n_rounds=5)
        preds = pipe.predict(X)
        assert preds.shape == (len(X),)

    def test_build_preprocessors_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preprocessor type"):
            build_preprocessors([{"type": "DoesNotExist"}])


class TestExperimentV1Schema:
    def _make_experiment(self, tmp_path: Path) -> ExperimentV1:
        return ExperimentV1(
            data=DataConfig(data_dir=tmp_path),
            features=FeatureConfig(
                columns=["a", "b"],
                groups={"grp": ["a", "b"]},
            ),
            preprocessing=[PreprocessorStep(type="StandardScaler")],
            models=[ModelSpec(type="XGBoost", params={"max_depth": 3})],
            ensemble_method="single",
            train=TrainConfig(n_rounds=10, early_stopping_rounds=5),
            evaluation=EvaluationConfig(primary_metric="sharpe"),
        )

    def test_to_pipeline_config_keys(self, tmp_path: Path) -> None:
        exp = self._make_experiment(tmp_path)
        cfg = exp.to_pipeline_config()
        assert "preprocessors" in cfg
        assert "models" in cfg
        assert "ensemble_method" in cfg
        assert "feature_groups" in cfg

    def test_to_pipeline_config_values(self, tmp_path: Path) -> None:
        exp = self._make_experiment(tmp_path)
        cfg = exp.to_pipeline_config()
        assert cfg["ensemble_method"] == "single"
        assert cfg["feature_groups"] == {"grp": ["a", "b"]}
        assert cfg["preprocessors"][0]["type"] == "StandardScaler"
        assert cfg["models"][0]["type"] == "XGBoost"

    def test_default_feature_config(self, tmp_path: Path) -> None:
        exp = ExperimentV1(
            data=DataConfig(data_dir=tmp_path),
            models=[ModelSpec(type="XGBoost")],
        )
        cfg = exp.to_pipeline_config()
        assert cfg["feature_groups"] == {}
        assert cfg["preprocessors"] == []


class TestPurgedErasCVSplitEras:
    def _make_eras(self, n_eras: int, rows_per_era: int = 5) -> pd.Series:
        return pd.Series(
            np.repeat([f"era_{i:04d}" for i in range(n_eras)], rows_per_era)
        )

    def test_split_eras_yields_non_overlapping(self) -> None:
        eras = self._make_eras(20)
        cv = PurgedEraCV(n_splits=3, n_purge=1, n_embargo=1, min_train_eras=4)
        for train_eras, test_eras in cv.split_eras(eras):
            assert set(train_eras).isdisjoint(set(test_eras))

    def test_split_eras_raises_on_too_few_eras(self) -> None:
        eras = self._make_eras(3)
        cv = PurgedEraCV(n_splits=3, n_purge=2, n_embargo=2, min_train_eras=10)
        with pytest.raises(ValueError, match="Not enough eras"):
            list(cv.split_eras(eras))

    def test_split_eras_summary_consistent(self) -> None:
        eras = self._make_eras(20)
        cv = PurgedEraCV(n_splits=3, n_purge=1, n_embargo=1, min_train_eras=4)
        summaries = cv.summary(eras)
        assert len(summaries) > 0
        for s in summaries:
            assert s["n_train_eras"] >= cv.min_train_eras
            assert s["n_test_eras"] >= 1
