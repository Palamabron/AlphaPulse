"""Tests for v0.5.0 production hardening features."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphapulse.hpo.trial_db import TrialDB


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
        hash1 = s1.split("_")[-1]
        hash2 = s2.split("_")[-1]
        assert hash1 != hash2


class TestProvenanceArtifact:
    def test_provenance_has_required_keys(self) -> None:
        from scripts.export_numerai_pickle import _provenance

        flat = {"model_1_type": "XGBoost"}
        pipeline_cfg = {"models": [{"type": "XGBoost"}]}
        prov = _provenance(flat, pipeline_cfg, "target")
        assert "git_commit" in prov
        assert "dependencies" in prov
        assert "flat_config" in prov
        assert "resolved_config" in prov
        assert "target_col" in prov

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
                db.insert_trial(0, {"lr": 0.9})  # OR IGNORE, should not overwrite
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
    def _make_data(self) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        rng = np.random.default_rng(0)
        n = 60
        X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["f0", "f1", "f2", "f3"])
        y = pd.Series(rng.standard_normal(n))
        era = pd.Series(["e1"] * 20 + ["e2"] * 20 + ["e3"] * 20)
        preds = pd.Series(rng.standard_normal(n))
        return X, y, era, preds

    def test_backtester_accepts_neutralizer(self) -> None:
        from alphapulse.evaluation.backtester import Backtester
        from alphapulse.pipeline.neutralizer import FeatureNeutralizer

        X, y, era, _ = self._make_data()

        class ConstantPredictor:
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.random.default_rng(1).standard_normal(len(X))

        neutralizer = FeatureNeutralizer(proportion=0.5)
        bt = Backtester(ConstantPredictor(), neutralizer=neutralizer)
        metrics = bt.evaluate(X, y, era)
        assert "corr_sharpe" in metrics

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

        bt_plain = Backtester(FixedPredictor())
        bt_neutralized = Backtester(
            FixedPredictor(), neutralizer=FeatureNeutralizer(1.0)
        )

        m_plain = bt_plain.evaluate(X, y, era)
        m_neutralized = bt_neutralized.evaluate(X, y, era)
        assert m_plain["corr_sharpe"] != m_neutralized["corr_sharpe"]

    def test_backtester_without_neutralizer_unchanged(self) -> None:
        from alphapulse.evaluation.backtester import Backtester

        X, y, era, _ = self._make_data()

        class FixedPredictor:
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.random.default_rng(3).standard_normal(len(X))

        bt = Backtester(FixedPredictor(), neutralizer=None)
        metrics = bt.evaluate(X, y, era)
        assert "corr_sharpe" in metrics


class TestMaskedAuxTargets:
    def _make_multi_target_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(0)
        n = 40
        X = pd.DataFrame(
            rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)]
        )
        X_val = pd.DataFrame(rng.standard_normal((20, 5)), columns=X.columns)
        primary = pd.Series(rng.standard_normal(n), name="target")
        aux_partial = pd.Series(
            [rng.standard_normal() if i % 2 == 0 else np.nan for i in range(n)],
            name="aux",
        )
        targets = pd.DataFrame({"target": primary, "aux": aux_partial})
        targets_val = pd.DataFrame(
            {
                "target": pd.Series(rng.standard_normal(20)),
                "aux": pd.Series(
                    [rng.standard_normal() if i % 2 == 0 else np.nan for i in range(20)]
                ),
            }
        )
        return X, X_val, targets, targets_val

    def test_trains_with_partial_nan_aux_target(self) -> None:
        from alphapulse.models.sklearn_models import RidgeModel
        from alphapulse.pipeline.multi_target import MultiTargetPipeline

        X, X_val, targets, targets_val = self._make_multi_target_data()

        pipeline = MultiTargetPipeline(
            preprocessors=[],
            model_factory=RidgeModel,
            target_columns=["target", "aux"],
            primary_target="target",
        )
        pipeline.fit(X, targets, X_val=X_val, targets_val=targets_val)
        preds = pipeline.predict(X)
        assert len(preds) == len(X)
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

    def test_predict_works_when_aux_skipped(self) -> None:
        from alphapulse.models.sklearn_models import RidgeModel
        from alphapulse.pipeline.multi_target import MultiTargetPipeline

        rng = np.random.default_rng(2)
        n = 40
        X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["f0", "f1", "f2"])
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
        preds = pipeline.predict(X)
        assert len(preds) == n

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
