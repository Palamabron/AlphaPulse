"""Tests for AlphaPulse feature implementations (v0.4.0 and v0.5.0)."""

import random
import tempfile
import warnings
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from alphapulse.evaluation.export_validation import smoke_test_predict_fn
from alphapulse.evaluation.metrics import rank_normalize, rank_normalize_per_era
from alphapulse.hpo.trial_db import TrialDB
from alphapulse.models import RidgeModel
from alphapulse.pipeline import Pipeline
from alphapulse.utils import set_global_seed


class TestSetGlobalSeed:
    def test_is_idempotent_for_numpy(self) -> None:
        set_global_seed(42)
        v1 = np.random.random()
        set_global_seed(42)
        v2 = np.random.random()
        assert v1 == v2

    def test_seeds_python_random(self) -> None:
        set_global_seed(99)
        v1 = random.random()
        set_global_seed(99)
        v2 = random.random()
        assert v1 == v2

    def test_negative_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            set_global_seed(-1)

    def test_zero_seed_is_valid(self) -> None:
        set_global_seed(0)


class TestRankNormalizePerEra:
    def test_output_same_length(self) -> None:
        preds = np.array([0.1, 0.5, 0.9, 0.2, 0.8, 0.4])
        eras = pd.Series(["e1", "e1", "e1", "e2", "e2", "e2"])
        out = rank_normalize_per_era(preds, eras)
        assert len(out) == len(preds)

    def test_values_in_zero_one(self) -> None:
        preds = np.array([0.1, 0.5, 0.9, 0.2, 0.8, 0.4])
        eras = pd.Series(["e1", "e1", "e1", "e2", "e2", "e2"])
        out = rank_normalize_per_era(preds, eras)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_per_era_not_global(self) -> None:
        preds = np.array([0.1, 0.9, 0.5, 0.6])
        eras = pd.Series(["e1", "e1", "e2", "e2"])
        out = rank_normalize_per_era(preds, eras)
        np.testing.assert_allclose(out[0], 0.0)
        np.testing.assert_allclose(out[1], 1.0)
        np.testing.assert_allclose(out[2], 0.0)
        np.testing.assert_allclose(out[3], 1.0)

    def test_differs_from_global_normalization(self) -> None:
        preds = np.array([0.1, 0.9, 0.5, 0.6])
        eras = pd.Series(["e1", "e1", "e2", "e2"])
        per_era_out = rank_normalize_per_era(preds, eras)
        global_out = rank_normalize(preds)
        assert not np.allclose(per_era_out, global_out)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            rank_normalize_per_era(np.array([1.0, 2.0]), pd.Series(["e1"]))

    def test_exported_from_evaluation_package(self) -> None:
        from alphapulse.evaluation import rank_normalize_per_era as rnpe

        assert callable(rnpe)


class TestToNumeraiPredictColumnAlignment:
    def _make_fitted_pipeline(self, feature_columns: list[str]) -> Pipeline:
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            rng.standard_normal((30, len(feature_columns))),
            columns=feature_columns,
        )
        y = pd.Series(rng.standard_normal(30))
        pipe = Pipeline(
            preprocessors=[], model=RidgeModel(), feature_columns=feature_columns
        )
        pipe.fit(X, y)
        return pipe

    def test_handles_missing_columns(self) -> None:
        cols = ["f0", "f1", "f2"]
        pipe = self._make_fitted_pipeline(cols)
        predict_fn = pipe.to_numerai_predict()
        live = pd.DataFrame({"f0": [0.1, 0.2], "f1": [0.3, 0.4]})
        bench = pd.DataFrame({"v2": [0.5, 0.5]})
        result = predict_fn(live, bench)
        assert isinstance(result, pd.DataFrame)
        assert "prediction" in result.columns
        assert len(result) == 2

    def test_handles_extra_columns(self) -> None:
        cols = ["f0", "f1"]
        pipe = self._make_fitted_pipeline(cols)
        predict_fn = pipe.to_numerai_predict()
        live = pd.DataFrame({"f0": [0.1], "f1": [0.2], "f_extra": [0.9]})
        bench = pd.DataFrame({"v2": [0.5]})
        result = predict_fn(live, bench)
        assert len(result) == 1

    def test_predictions_in_unit_interval(self) -> None:
        cols = ["f0", "f1", "f2"]
        pipe = self._make_fitted_pipeline(cols)
        predict_fn = pipe.to_numerai_predict()
        rng = np.random.default_rng(42)
        live = pd.DataFrame(rng.standard_normal((50, 3)), columns=cols)
        bench = pd.DataFrame({"v2": rng.random(50)})
        result = predict_fn(live, bench)
        preds = result["prediction"].to_numpy()
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


class TestValidateFeatureSchema:
    def test_warns_on_missing_columns(self) -> None:
        cols = ["f0", "f1", "f2"]
        pipe = Pipeline(preprocessors=[], model=RidgeModel(), feature_columns=cols)
        X_missing = pd.DataFrame({"f0": [1.0], "f1": [2.0]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe.validate_feature_schema(X_missing)
        assert len(w) == 1
        assert "f2" in str(w[0].message)

    def test_no_warning_on_exact_match(self) -> None:
        cols = ["f0", "f1"]
        pipe = Pipeline(preprocessors=[], model=RidgeModel(), feature_columns=cols)
        X_ok = pd.DataFrame({"f0": [1.0], "f1": [2.0]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe.validate_feature_schema(X_ok)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_no_warning_when_no_feature_columns_set(self) -> None:
        pipe = Pipeline(preprocessors=[], model=RidgeModel(), feature_columns=None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe.validate_feature_schema(pd.DataFrame({"f0": [1.0]}))
        assert len(w) == 0


class TestBenchmarkColumnsExcluded:
    def test_benchmark_columns_excluded_by_resolve(self) -> None:
        from alphapulse.experiments.data import resolve_feature_columns

        df = pd.DataFrame(
            {
                "f0": [1.0, 2.0],
                "f1": [3.0, 4.0],
                "v2_equivalent_return": [0.1, 0.2],
                "era": ["e1", "e2"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cols = resolve_feature_columns(
                df,
                Path(tmpdir),
                explicit=None,
                benchmark_columns=["v2_equivalent_return"],
            )
        assert "v2_equivalent_return" not in cols
        assert "f0" in cols
        assert "f1" in cols

    def test_benchmark_columns_excluded_from_explicit(self) -> None:
        from alphapulse.experiments.data import resolve_feature_columns

        df = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0], "bm": [0.1, 0.2]})
        with tempfile.TemporaryDirectory() as tmpdir:
            cols = resolve_feature_columns(
                df,
                Path(tmpdir),
                explicit=["f0", "f1", "bm"],
                benchmark_columns=["bm"],
            )
        assert "bm" not in cols
        assert "f0" in cols

    def test_data_config_benchmark_columns_field(self) -> None:
        from alphapulse.experiments.schema import DataConfig

        cfg = DataConfig(
            data_dir=Path("/tmp"),
            benchmark_columns=["v2_equivalent_return"],
        )
        assert cfg.benchmark_columns == ["v2_equivalent_return"]

    def test_data_config_benchmark_columns_default_empty(self) -> None:
        from alphapulse.experiments.schema import DataConfig

        cfg = DataConfig(data_dir=Path("/tmp"))
        assert cfg.benchmark_columns == []


class TestSmokeTestPredictFn:
    def _write_valid_pkl(self, path: Path, feature_columns: list[str]) -> None:
        def predict_fn(
            live_features: pd.DataFrame,
            live_benchmark_models: pd.DataFrame,
        ) -> pd.DataFrame:
            preds = rank_normalize(np.random.default_rng(0).random(len(live_features)))
            return pd.DataFrame({"prediction": preds}, index=live_features.index)

        with open(path, "wb") as f:
            cloudpickle.dump(predict_fn, f)

    def test_passes_on_valid_pkl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl = Path(tmpdir) / "predict.pkl"
            cols = ["f0", "f1", "f2"]
            self._write_valid_pkl(pkl, cols)
            smoke_test_predict_fn(pkl, cols)

    def test_raises_on_corrupt_pkl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl = Path(tmpdir) / "predict.pkl"
            pkl.write_bytes(b"not a valid pickle")
            with pytest.raises(RuntimeError, match="smoke_test"):
                smoke_test_predict_fn(pkl, ["f0"])

    def test_raises_when_prediction_out_of_range(self) -> None:
        def bad_fn(
            live_features: pd.DataFrame,
            live_benchmark_models: pd.DataFrame,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {"prediction": [2.0] * len(live_features)},
                index=live_features.index,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl = Path(tmpdir) / "predict.pkl"
            with open(pkl, "wb") as f:
                cloudpickle.dump(bad_fn, f)
            with pytest.raises(RuntimeError, match="smoke_test"):
                smoke_test_predict_fn(pkl, ["f0"])

    def test_raises_when_missing_prediction_column(self) -> None:
        def bad_fn(
            live_features: pd.DataFrame,
            live_benchmark_models: pd.DataFrame,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {"wrong_col": [0.5] * len(live_features)},
                index=live_features.index,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl = Path(tmpdir) / "predict.pkl"
            with open(pkl, "wb") as f:
                cloudpickle.dump(bad_fn, f)
            with pytest.raises(RuntimeError, match="smoke_test"):
                smoke_test_predict_fn(pkl, ["f0"])

    def test_tolerates_extra_feature_columns_in_live(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl = Path(tmpdir) / "predict.pkl"
            cols = ["f0", "f1"]
            self._write_valid_pkl(pkl, cols)
            smoke_test_predict_fn(pkl, cols)


class TestCanonicalArtifactNaming:
    def test_stem_format(self) -> None:
        import re

        from scripts.export_numerai_pickle import _artifact_stem

        flat = {"model_1_type": "XGBoost", "num_models": 1}
        stem = _artifact_stem(flat, "target")
        pattern = r"^\d{8}T\d{6}_XGBoost_target_[0-9a-f]{8}$"
        assert re.match(pattern, stem), f"Unexpected stem format: {stem}"

    def test_stem_uses_model_type(self) -> None:
        from scripts.export_numerai_pickle import _artifact_stem

        flat = {"model_1_type": "LightGBM"}
        stem = _artifact_stem(flat, "target_victor")
        assert "LightGBM" in stem
        assert "target_victor" in stem

    def test_stem_falls_back_on_missing_model_type(self) -> None:
        from scripts.export_numerai_pickle import _artifact_stem

        stem = _artifact_stem({}, "target")
        assert "unknown" in stem

    def test_different_configs_produce_different_hashes(self) -> None:
        from scripts.export_numerai_pickle import _artifact_stem

        s1 = _artifact_stem({"model_1_type": "XGBoost", "lr": 0.1}, "target")
        s2 = _artifact_stem({"model_1_type": "XGBoost", "lr": 0.2}, "target")
        assert s1.split("_")[-1] != s2.split("_")[-1]


class TestProvenanceArtifact:
    def test_provenance_has_required_keys(self) -> None:
        from scripts.export_numerai_pickle import _provenance

        prov = _provenance({"model_1_type": "XGBoost"}, {"models": []}, "target")
        assert {
            "git_commit",
            "dependencies",
            "flat_config",
            "resolved_config",
            "target_col",
        } <= prov.keys()

    def test_provenance_embeds_configs(self) -> None:
        from scripts.export_numerai_pickle import _provenance

        flat = {"model_1_type": "LightGBM", "lr": 0.05}
        pipeline_cfg = {"models": [{"type": "LightGBM"}]}
        prov = _provenance(flat, pipeline_cfg, "target_victor")
        assert prov["flat_config"] == flat
        assert prov["resolved_config"] == pipeline_cfg
        assert prov["target_col"] == "target_victor"

    def test_git_commit_is_string(self) -> None:
        from scripts.export_numerai_pickle import _provenance

        prov = _provenance({}, {}, "target")
        assert isinstance(prov["git_commit"], str)
        assert len(prov["git_commit"]) > 0


class TestTrialDB:
    def test_insert_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trials.db"
            with TrialDB(db_path) as db:
                db.insert_trial(0, {"model_1_type": "XGBoost"})
                rows = db.load_all_trials()
            assert len(rows) == 1
            assert rows[0]["trial_number"] == 0
            assert rows[0]["status"] == "running"

    def test_update_to_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trials.db"
            with TrialDB(db_path) as db:
                db.insert_trial(1, {"model_1_type": "LightGBM"})
                db.update_trial(
                    1,
                    status="completed",
                    metrics={"corr_sharpe": 1.2},
                    elapsed_seconds=42.0,
                )
                rows = db.load_all_trials()
            assert rows[0]["status"] == "completed"
            assert rows[0]["metrics"]["corr_sharpe"] == pytest.approx(1.2)
            assert rows[0]["elapsed_seconds"] == pytest.approx(42.0)

    def test_completed_trials_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trials.db"
            with TrialDB(db_path) as db:
                db.insert_trial(0, {})
                db.insert_trial(1, {})
                db.insert_trial(2, {})
                db.update_trial(0, status="completed")
                db.update_trial(2, status="completed")
                done = db.completed_trials()
            assert done == {0, 2}
            assert 1 not in done

    def test_insert_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trials.db"
            with TrialDB(db_path) as db:
                db.insert_trial(0, {"lr": 0.1})
                db.insert_trial(0, {"lr": 0.9})
                rows = db.load_all_trials()
            assert len(rows) == 1
            assert rows[0]["flat_config"]["lr"] == pytest.approx(0.9)

    def test_insert_skips_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trials.db"
            with TrialDB(db_path) as db:
                db.insert_trial(0, {"lr": 0.1})
                db.update_trial(0, status="completed")
                db.insert_trial(0, {"lr": 0.9})
                rows = db.load_all_trials()
            assert len(rows) == 1
            assert rows[0]["flat_config"]["lr"] == pytest.approx(0.1)

    def test_persists_across_connections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trials.db"
            with TrialDB(db_path) as db:
                db.insert_trial(5, {"model_1_type": "Ridge"})
                db.update_trial(5, status="completed", elapsed_seconds=10.0)
            with TrialDB(db_path) as db2:
                rows = db2.load_all_trials()
            assert len(rows) == 1
            assert rows[0]["trial_number"] == 5
            assert rows[0]["status"] == "completed"


class TestNeutralizerInBacktester:
    def _make_data(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        rng = np.random.default_rng(0)
        n = 60
        X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["f0", "f1", "f2", "f3"])
        y = pd.Series(rng.standard_normal(n))
        era = pd.Series(["e1"] * 20 + ["e2"] * 20 + ["e3"] * 20)
        return X, y, era

    def test_backtester_accepts_neutralizer(self) -> None:
        from alphapulse.evaluation.backtester import Backtester
        from alphapulse.pipeline.neutralizer import FeatureNeutralizer

        X, y, era = self._make_data()

        class ConstantPredictor:
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.random.default_rng(1).standard_normal(len(X))

        bt = Backtester(
            ConstantPredictor(), neutralizer=FeatureNeutralizer(proportion=0.5)
        )
        assert "corr_sharpe" in bt.evaluate(X, y, era)

    def test_neutralizer_changes_metrics(self) -> None:
        from alphapulse.evaluation.backtester import Backtester
        from alphapulse.pipeline.neutralizer import FeatureNeutralizer

        rng = np.random.default_rng(42)
        n = 60
        X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["f0", "f1", "f2", "f3"])
        y = pd.Series(rng.standard_normal(n))
        era = pd.Series(["e1"] * 20 + ["e2"] * 20 + ["e3"] * 20)

        class FixedPredictor:
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.random.default_rng(7).standard_normal(len(X))

        m_plain = Backtester(FixedPredictor()).evaluate(X, y, era)
        m_neutralized = Backtester(
            FixedPredictor(), neutralizer=FeatureNeutralizer(1.0)
        ).evaluate(X, y, era)
        assert m_plain["corr_sharpe"] != m_neutralized["corr_sharpe"]


class TestMaskedAuxTargets:
    def test_trains_with_partial_nan_aux_target(self) -> None:
        from alphapulse.models.sklearn_models import RidgeModel
        from alphapulse.pipeline.multi_target import MultiTargetPipeline

        rng = np.random.default_rng(0)
        n = 40
        X = pd.DataFrame(
            rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)]
        )
        X_val = pd.DataFrame(rng.standard_normal((20, 5)), columns=X.columns)
        targets = pd.DataFrame(
            {
                "target": pd.Series(rng.standard_normal(n)),
                "aux": pd.Series(
                    [rng.standard_normal() if i % 2 == 0 else np.nan for i in range(n)]
                ),
            }
        )
        targets_val = pd.DataFrame(
            {
                "target": pd.Series(rng.standard_normal(20)),
                "aux": pd.Series(
                    [rng.standard_normal() if i % 2 == 0 else np.nan for i in range(20)]
                ),
            }
        )
        pipeline = MultiTargetPipeline(
            preprocessors=[],
            model_factory=RidgeModel,
            target_columns=["target", "aux"],
            primary_target="target",
        )
        pipeline.fit(X, targets, X_val=X_val, targets_val=targets_val)
        preds = pipeline.predict(X)
        assert len(preds) == n
        assert np.all(np.isfinite(preds))

    def test_skips_fully_nan_aux_target(self) -> None:
        from alphapulse.models.sklearn_models import RidgeModel
        from alphapulse.pipeline.multi_target import MultiTargetPipeline

        rng = np.random.default_rng(1)
        n = 40
        X = pd.DataFrame(
            rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)]
        )
        targets = pd.DataFrame(
            {
                "target": pd.Series(rng.standard_normal(n)),
                "all_nan": pd.Series([np.nan] * n),
            }
        )
        pipeline = MultiTargetPipeline(
            preprocessors=[],
            model_factory=RidgeModel,
            target_columns=["target", "all_nan"],
            primary_target="target",
        )
        pipeline.fit(X, targets)
        assert "all_nan" not in pipeline._models
        assert "target" in pipeline._models

    def test_three_targets_one_skipped_no_dimension_mismatch(self) -> None:
        from alphapulse.models.sklearn_models import RidgeModel
        from alphapulse.pipeline.multi_target import MultiTargetPipeline

        rng = np.random.default_rng(3)
        n = 40
        X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["f0", "f1", "f2"])
        targets = pd.DataFrame(
            {
                "target": pd.Series(rng.standard_normal(n)),
                "aux1": pd.Series(rng.standard_normal(n)),
                "all_nan": pd.Series([np.nan] * n),
            }
        )
        pipeline = MultiTargetPipeline(
            preprocessors=[],
            model_factory=RidgeModel,
            target_columns=["target", "aux1", "all_nan"],
            primary_target="target",
        )
        pipeline.fit(X, targets)
        assert "all_nan" not in pipeline._models
        assert pipeline._weights is not None
        assert len(pipeline._weights) == 2
        preds = pipeline.predict(X)
        assert len(preds) == n
        assert np.all(np.isfinite(preds))

    def test_sharpe_blend_skips_nan_target_no_keyerror(self) -> None:
        from alphapulse.models.sklearn_models import RidgeModel
        from alphapulse.pipeline.multi_target import MultiTargetPipeline

        rng = np.random.default_rng(4)
        n = 60
        X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["f0", "f1", "f2"])
        era = pd.Series(["e1"] * 20 + ["e2"] * 20 + ["e3"] * 20)
        targets = pd.DataFrame(
            {
                "target": pd.Series(rng.standard_normal(n)),
                "aux1": pd.Series(rng.standard_normal(n)),
                "all_nan": pd.Series([np.nan] * n),
            }
        )
        pipeline = MultiTargetPipeline(
            preprocessors=[],
            model_factory=RidgeModel,
            target_columns=["target", "aux1", "all_nan"],
            primary_target="target",
            blend_method="sharpe",
        )
        pipeline.fit(X, targets, era_train=era)
        assert "all_nan" not in pipeline._models
        preds = pipeline.predict(X)
        assert len(preds) == n
        assert np.all(np.isfinite(preds))
