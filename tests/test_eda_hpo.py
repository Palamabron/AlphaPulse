import json
from pathlib import Path

from eda.utils.hpo import load_hpo_trials, rank_trials, recipe_summary


def test_hpo_trials_preserve_validation_and_holdout_metrics(tmp_path: Path) -> None:
    path = tmp_path / "all_trials.json"
    path.write_text(
        json.dumps(
            [
                {
                    "trial": 7,
                    "sharpe": 0.4,
                    "metrics": {
                        "holdout_corr_sharpe": 0.4,
                        "val_corr_sharpe": 0.2,
                        "payout_score": 0.8,
                    },
                    "params": {
                        "num_models": 2,
                        "model_1_type": "LightGBM",
                        "model_2_type": "XGBoost",
                        "ensemble_method": "weighted",
                    },
                    "elapsed_seconds": 12.0,
                    "error": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    trials = load_hpo_trials(path)

    assert trials.loc[0, "holdout_corr_sharpe"] == 0.4
    assert trials.loc[0, "payout_score"] == 0.8
    assert trials.loc[0, "recipe"] == "LightGBM+XGBoost · weighted"


def test_hpo_rankings_can_select_different_winners(tmp_path: Path) -> None:
    path = tmp_path / "all_trials.json"
    base = {
        "params": {
            "num_models": 1,
            "model_1_type": "XGBoost",
            "ensemble_method": "single",
        },
        "elapsed_seconds": 1.0,
        "error": None,
    }
    path.write_text(
        json.dumps(
            [
                {
                    **base,
                    "trial": 1,
                    "sharpe": 1.0,
                    "metrics": {
                        "holdout_corr_sharpe": 1.0,
                        "val_corr_sharpe": 0.0,
                        "payout_score": -0.2,
                    },
                },
                {
                    **base,
                    "trial": 2,
                    "sharpe": -0.1,
                    "metrics": {
                        "holdout_corr_sharpe": -0.1,
                        "val_corr_sharpe": 0.4,
                        "payout_score": 0.9,
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    trials = load_hpo_trials(path)

    assert rank_trials(trials, "holdout").loc[0, "trial"] == 1
    assert rank_trials(trials, "validation").loc[0, "trial"] == 2
    assert len(recipe_summary(trials)) == 1
