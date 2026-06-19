import json
from pathlib import Path
from unittest.mock import MagicMock

import optuna

from alphapulse.hpo.optuna_search import (
    DEFAULT_N_STARTUP_TRIALS,
    _suggest_model_hyperparams,
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


def test_create_hpo_study_uses_n_startup_trials(tmp_path: Path) -> None:
    out = tmp_path / "hpo"
    out.mkdir()
    study = create_hpo_study(
        out, seed=1, sampler="tpe", resume=False, n_startup_trials=25
    )
    sampler = study.sampler
    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert sampler._n_startup_trials == 25


def test_default_n_startup_trials_constant() -> None:
    assert DEFAULT_N_STARTUP_TRIALS == 25


def test_suggest_model_hyperparams_skips_unused_families() -> None:
    trial = MagicMock()
    params = _suggest_model_hyperparams(trial, {"CatBoost"}, fast=True)
    assert "catboost_depth" in params
    assert "lgbm_num_leaves" not in params
    assert "xgb_max_depth" not in params
    assert "foundation_max_train_rows" not in params
    trial.suggest_categorical.assert_called()
    trial.suggest_float.assert_called()


def test_suggest_model_hyperparams_includes_foundation_when_active() -> None:
    trial = MagicMock()
    params = _suggest_model_hyperparams(trial, {"TabPFN"}, fast=True)
    assert "foundation_max_train_rows" in params
    assert "catboost_depth" not in params


def test_suggest_flat_config_omits_lgbm_when_not_selected() -> None:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_flat_config(trial, fast=True)
        types = [
            cfg.get("model_1_type"),
            cfg.get("model_2_type"),
        ][: int(cfg.get("num_models", 1))]
        if "LightGBM" not in types:
            assert not any(k.startswith("lgbm_") for k in cfg)
        if "XGBoost" not in types:
            assert not any(k.startswith("xgb_") for k in cfg)
        if "CatBoost" not in types:
            assert not any(k.startswith("catboost_") for k in cfg)
        return 0.0

    study.optimize(objective, n_trials=30)


def test_suggest_flat_config_meta_neutralization_conditional() -> None:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=1),
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_flat_config(trial, fast=True)
        if not cfg.get("use_meta_neutralization"):
            assert "meta_neutralization_proportion" not in cfg
        return 0.0

    study.optimize(objective, n_trials=20)


def test_suggest_flat_config_single_model_no_ensemble_method_suggest() -> None:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=2),
    )
    saw_single = False
    for _ in range(40):
        trial = study.ask()
        cfg = suggest_flat_config(trial, fast=True)
        tell_trial_result(study, trial, 0.0)
        if int(cfg.get("num_models", 1)) == 1:
            saw_single = True
            assert cfg["ensemble_method"] == "single"
    assert saw_single
