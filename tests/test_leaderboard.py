import json
from pathlib import Path

from alphapulse.logging_.leaderboard import (
    TrialLeaderboardEntry,
    format_leaderboard,
    save_leaderboard,
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
        ),
        TrialLeaderboardEntry(
            trial_number=2,
            sharpe=1.2,
            mean_per_era_correlation=0.04,
            std_per_era_correlation=0.02,
            max_drawdown=0.05,
            model_types="LightGBM+CatBoost",
            elapsed_seconds=25.0,
        ),
    ]
    text = format_leaderboard(entries, current_trial=2)
    assert "LEADERBOARD" in text
    assert "by sharpe" in text
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
        ),
    ]
    text = format_leaderboard(entries)
    assert "by payout_score" in text
    assert "Payout" in text
    assert text.index("LightGBM") < text.index("XGBoost")

    path = tmp_path / "leaderboard.json"
    save_leaderboard(path, entries)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[0]["trial_number"] == 2
    assert data[0]["payout_score"] == 1.1
