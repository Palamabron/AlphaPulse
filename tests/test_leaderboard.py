import json
from pathlib import Path

from alphapulse.logging_.leaderboard import (
    TrialLeaderboardEntry,
    compute_robust_payout_score,
    format_leaderboard,
    save_leaderboard,
    selection_score_from_metrics,
)


def test_leaderboard_format_and_sort_by_sharpe(tmp_path: Path) -> None:
    entries = [
        TrialLeaderboardEntry(
            trial_number=1,
            sharpe=0.5,
            mean_per_era_correlation=0.02,
            std_per_era_correlation=0.01,
            max_drawdown=0.1,
            model_types="XGBoost",
            elapsed_seconds=10.0,
            holdout_corr_sharpe=0.5,
        ),
        TrialLeaderboardEntry(
            trial_number=2,
            sharpe=1.2,
            mean_per_era_correlation=0.04,
            std_per_era_correlation=0.02,
            max_drawdown=0.05,
            model_types="LightGBM+CatBoost",
            elapsed_seconds=25.0,
            holdout_corr_sharpe=1.2,
        ),
    ]
    text = format_leaderboard(entries, current_trial=2)
    assert "LEADERBOARD" in text
    assert "holdout corr_sharpe" in text
    assert "HoldoutSharpe" in text
    assert "LightGBM+CatBoost" in text
    assert "*" in text

    path = tmp_path / "leaderboard.json"
    save_leaderboard(path, entries)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[0]["trial_number"] == 2


def test_leaderboard_format_and_sort_by_payout_score(tmp_path: Path) -> None:
    entries = [
        TrialLeaderboardEntry(
            trial_number=1,
            sharpe=1.5,
            mean_per_era_correlation=0.03,
            std_per_era_correlation=0.01,
            max_drawdown=0.1,
            model_types="XGBoost",
            elapsed_seconds=10.0,
            payout_score=0.8,
            mmc_sharpe=0.2,
            val_corr_sharpe=0.1,
            val_mean_per_era_correlation=0.01,
            holdout_corr_sharpe=1.5,
            robust_payout_score=0.8,
        ),
        TrialLeaderboardEntry(
            trial_number=2,
            sharpe=0.9,
            mean_per_era_correlation=0.04,
            std_per_era_correlation=0.02,
            max_drawdown=0.05,
            model_types="LightGBM",
            elapsed_seconds=25.0,
            payout_score=1.1,
            mmc_sharpe=0.4,
            val_corr_sharpe=0.2,
            val_mean_per_era_correlation=0.02,
            holdout_corr_sharpe=0.9,
            robust_payout_score=1.1,
        ),
    ]
    text = format_leaderboard(entries)
    assert "by legacy proxy on validation" in text
    assert "ValidationMmcSharpe" in text
    assert "ValidationSharpe" in text
    assert "HoldoutSharpe" in text
    assert text.index("LightGBM") < text.index("XGBoost")

    path = tmp_path / "leaderboard.json"
    save_leaderboard(path, entries)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[0]["trial_number"] == 2
    assert data[0]["payout_score"] == 1.1
    assert data[0]["mmc_sharpe"] == 0.4


def test_robust_payout_penalizes_negative_holdout() -> None:
    payout = 1.0
    robust = compute_robust_payout_score(
        payout, val_corr_sharpe=0.4, holdout_corr_sharpe=-0.02
    )
    assert robust is not None
    assert robust == 0.25


def test_robust_payout_does_not_improve_negative_payout() -> None:
    robust = compute_robust_payout_score(
        -0.4, val_corr_sharpe=0.3, holdout_corr_sharpe=-0.1
    )
    assert robust is not None
    assert robust == -1.6


def test_robust_leaderboard_prefers_consistent_trial() -> None:
    entries = [
        TrialLeaderboardEntry(
            trial_number=35,
            sharpe=-0.02,
            mean_per_era_correlation=-0.001,
            std_per_era_correlation=0.04,
            max_drawdown=0.2,
            model_types="XGBoost",
            elapsed_seconds=300.0,
            payout_score=0.997,
            mmc_sharpe=0.32,
            val_corr_sharpe=0.37,
            val_mean_per_era_correlation=0.01,
            holdout_corr_sharpe=-0.02,
            robust_payout_score=compute_robust_payout_score(0.997, 0.37, -0.02),
        ),
        TrialLeaderboardEntry(
            trial_number=104,
            sharpe=0.62,
            mean_per_era_correlation=0.024,
            std_per_era_correlation=0.03,
            max_drawdown=0.1,
            model_types="LightGBM+XGBoost",
            elapsed_seconds=900.0,
            payout_score=0.916,
            mmc_sharpe=0.25,
            val_corr_sharpe=0.46,
            val_mean_per_era_correlation=0.016,
            holdout_corr_sharpe=0.615,
            robust_payout_score=compute_robust_payout_score(0.916, 0.46, 0.615),
        ),
    ]
    text = format_leaderboard(entries)
    assert "robust legacy proxy" in text
    assert text.index("104") < text.index("35", text.index("robust legacy proxy"))


def test_selection_score_from_metrics_uses_robust_payout() -> None:
    metrics = {
        "payout_score": 0.997,
        "val_corr_sharpe": 0.37,
        "holdout_corr_sharpe": -0.02,
    }
    objective_score = selection_score_from_metrics(
        metrics, objective="payout_score", criteria="objective"
    )
    robust_score = selection_score_from_metrics(
        metrics, objective="payout_score", criteria="robust_payout"
    )
    assert objective_score == 0.997
    assert robust_score == 0.997 * 0.25
