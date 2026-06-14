"""Tests for HPO trial database persistence and resume behavior."""

from pathlib import Path

from alphapulse.hpo.trial_db import TrialDB


def test_insert_trial_replaces_failed_config_on_retry(tmp_path: Path) -> None:
    db_path = tmp_path / "trials.db"
    first_config = {"model_1_type": "XGBoost", "xgb_n_rounds": 100}
    second_config = {"model_1_type": "CatBoost", "catboost_iterations": 200}

    with TrialDB(db_path) as db:
        db.insert_trial(0, first_config)
        db.update_trial(0, status="failed", error="boom")
        db.insert_trial(0, second_config)
        row = db.load_all_trials()[0]

    assert row["status"] == "running"
    assert row["flat_config"] == second_config
    assert row["error"] is None


def test_insert_trial_does_not_overwrite_completed(tmp_path: Path) -> None:
    db_path = tmp_path / "trials.db"
    completed_config = {"model_1_type": "XGBoost", "xgb_n_rounds": 300}
    new_config = {"model_1_type": "LightGBM", "lgbm_n_rounds": 400}

    with TrialDB(db_path) as db:
        db.insert_trial(1, completed_config)
        db.update_trial(
            1,
            status="completed",
            metrics={"corr_sharpe": 1.5},
            elapsed_seconds=12.0,
        )
        db.insert_trial(1, new_config)
        row = db.load_all_trials()[0]

    assert row["status"] == "completed"
    assert row["flat_config"] == completed_config


def test_completed_trials_skips_only_successful(tmp_path: Path) -> None:
    db_path = tmp_path / "trials.db"
    with TrialDB(db_path) as db:
        db.insert_trial(0, {"a": 1})
        db.update_trial(0, status="completed", metrics={"corr_sharpe": 1.0})
        db.insert_trial(1, {"b": 2})
        db.update_trial(1, status="failed", error="timeout")
        assert db.completed_trials() == {0}


def test_wandb_group_file_persists_across_resume(tmp_path: Path) -> None:
    from scripts.hpo_pipeline import _load_or_create_wandb_group

    output_dir = tmp_path / "hpo_run"
    output_dir.mkdir()
    first = _load_or_create_wandb_group(output_dir, "alphapulse-hpo")
    second = _load_or_create_wandb_group(output_dir, "alphapulse-hpo")
    assert first == second
    assert (output_dir / "wandb_group.txt").read_text(encoding="utf-8") == first


def test_all_results_from_db_includes_all_trials(tmp_path: Path) -> None:
    from scripts.hpo_pipeline import _all_results_from_db

    db_path = tmp_path / "trials.db"
    with TrialDB(db_path) as db:
        db.insert_trial(0, {"model_1_type": "XGBoost"})
        db.update_trial(
            0,
            status="completed",
            metrics={"corr_sharpe": 1.2},
            elapsed_seconds=5.0,
        )
        db.insert_trial(1, {"model_1_type": "CatBoost"})
        db.update_trial(1, status="failed", error="timeout", elapsed_seconds=3.0)
        results = _all_results_from_db(db)

    assert len(results) == 2
    assert results[0].sharpe == 1.2
    assert results[1].error == "timeout"
    assert results[0].params == {"model_1_type": "XGBoost"}
