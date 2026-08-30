"""Tests for AutoResearch loop, mutations, and state."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphapulse.autoresearch import mutations
from alphapulse.autoresearch.state import ResearchState, TrialRecord


@pytest.fixture
def toy_data_with_era() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    n_eras = 50
    rows_per_era = 8
    n = n_eras * rows_per_era
    X = pd.DataFrame(
        rng.standard_normal((n, 4)).astype(np.float64), columns=list("ABCD")
    )
    X["era"] = np.repeat([f"era_{i:04d}" for i in range(n_eras)], rows_per_era)
    y = pd.Series(X["A"] * 0.5 + X["B"] * 0.3 + rng.standard_normal(n) * 0.2)
    return {
        "X_train": X,
        "y_train": y,
        "era_train": X["era"],
        "feature_cols": list("ABCD"),
    }


def _base_config() -> dict[str, Any]:
    return {
        "preprocessors": [{"type": "StandardScaler", "params": {}}],
        "models": [{"type": "XGBoost", "params": {"max_depth": 3}}],
        "ensemble_method": "single",
        "ensemble_params": {},
    }


def test_autoresearch_uses_sixteen_era_purge_for_sixty_day_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import alphapulse.autoresearch.loop as loop

    observed: dict[str, int] = {}

    class FakeEvaluator:
        def __init__(self, **kwargs: Any) -> None:
            observed["n_purge"] = int(kwargs["n_purge"])

        def evaluate_walk_forward(self, *args: Any, **kwargs: Any) -> dict[str, float]:
            return {"corr_sharpe": 0.0}

    monkeypatch.setattr(loop, "EraSplitEvaluator", FakeEvaluator)
    X = pd.DataFrame({"feature": [0.0], "era": ["era_0001"]})
    y = pd.Series([0.0], index=X.index)

    loop._run_one_trial(
        _base_config(),
        X_train=X,
        y_train=y,
        era_train=X["era"],
        feature_cols=["feature"],
        seed=42,
        target_col="target_ender_60",
    )

    assert observed["n_purge"] == 16


class TestMutations:
    def test_add_model_switches_to_weighted(self) -> None:
        cfg = mutations.add_model(_base_config(), "LightGBM", {})
        assert len(cfg["models"]) == 2
        assert cfg["ensemble_method"] == "weighted"

    def test_remove_only_model_raises(self) -> None:
        with pytest.raises(ValueError, match="only model"):
            mutations.remove_model(_base_config(), 0)

    def test_set_neutralization_bounds(self) -> None:
        with pytest.raises(ValueError, match="proportion"):
            mutations.set_neutralization(_base_config(), 1.5)

    def test_tune_model_params_updates_nested_xgb_params(self) -> None:
        cfg = {
            "preprocessors": [],
            "models": [
                {
                    "type": "XGBoost",
                    "params": {
                        "params": {
                            "max_depth": 3,
                            "learning_rate": 0.01,
                        }
                    },
                }
            ],
            "ensemble_method": "single",
            "ensemble_params": {},
        }
        updated = mutations.tune_model_params(cfg, 0, {"max_depth": 7})
        inner = updated["models"][0]["params"]["params"]
        assert inner["max_depth"] == 7
        assert inner["learning_rate"] == 0.01


class TestResearchState:
    def test_save_load_roundtrip(self, tmp_path: Any) -> None:
        state = ResearchState(
            trials=[
                TrialRecord(
                    trial_number=0,
                    sharpe=1.0,
                    metrics={"corr_sharpe": 1.0},
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


def test_autoresearch_run_one_trial(toy_data_with_era: dict[str, Any]) -> None:
    from alphapulse.autoresearch.loop import _run_one_trial

    config = {
        "preprocessors": [{"type": "StandardScaler", "params": {}}],
        "models": [
            {
                "type": "XGBoost",
                "params": {
                    "params": {
                        "max_depth": 3,
                        "learning_rate": 0.1,
                        "tree_method": "hist",
                        "objective": "reg:squarederror",
                    }
                },
            }
        ],
        "ensemble_method": "single",
        "ensemble_params": {},
    }
    metrics, elapsed = _run_one_trial(
        config,
        seed=0,
        X_train=toy_data_with_era["X_train"],
        y_train=toy_data_with_era["y_train"],
        era_train=toy_data_with_era["era_train"],
        feature_cols=toy_data_with_era["feature_cols"],
    )
    assert elapsed > 0
    assert "corr_sharpe" in metrics
