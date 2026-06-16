import json
from pathlib import Path

import optuna

from alphapulse.hpo.optuna_search import (
    create_hpo_study,
    suggest_flat_config,
    tell_trial_result,
)


def test_suggest_flat_config_returns_routing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload = {
        "feature_sets": {
            "small": ["f_a", "f_b"],
            "medium": ["f_a", "f_b", "f_c"],
            "strength": ["f_a", "f_c"],
        },
        "targets": ["target", "target_alpha_20"],
    }
    (data_dir / "features.json").write_text(json.dumps(payload), encoding="utf-8")

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    cfg = suggest_flat_config(trial, fast=True, data_dir=data_dir)
    assert cfg["use_feature_routing"] is True
    assert cfg["active_groups"]
    assert cfg["routed_feature_count"] <= 1000
    tell_trial_result(study, trial, 0.5)


def test_create_hpo_study_persists_storage(tmp_path: Path) -> None:
    out = tmp_path / "hpo"
    out.mkdir()
    study = create_hpo_study(out, seed=1, sampler="tpe", resume=False)
    trial = study.ask()
    tell_trial_result(study, trial, 1.0)
    assert (out / "optuna.db").exists()
