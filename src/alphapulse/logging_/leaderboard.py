from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..autoresearch.state import TrialRecord
    from ..hpo.objective import TrialResult


@dataclass
class TrialLeaderboardEntry:
    trial_number: int
    sharpe: float
    mean_per_era_correlation: float
    std_per_era_correlation: float | None
    max_drawdown: float | None
    model_types: str
    elapsed_seconds: float
    error: str | None = None


def _model_types_from_flat(flat: dict[str, Any]) -> str:
    num = flat.get("num_models", 1)
    types = [
        flat.get("model_1_type", "?"),
        flat.get("model_2_type", "?"),
        flat.get("model_3_type", "?"),
    ][:num]
    return "+".join(str(t) for t in types)


def entry_from_hpo_result(result: TrialResult) -> TrialLeaderboardEntry:
    metrics = result.metrics
    return TrialLeaderboardEntry(
        trial_number=result.trial_number,
        sharpe=result.sharpe,
        mean_per_era_correlation=float(metrics.get("mean_per_era_correlation", 0.0)),
        std_per_era_correlation=metrics.get("std_per_era_correlation"),
        max_drawdown=metrics.get("max_drawdown"),
        model_types=_model_types_from_flat(result.params),
        elapsed_seconds=result.elapsed_seconds,
        error=result.error,
    )


def entry_from_trial_record(record: TrialRecord) -> TrialLeaderboardEntry:
    return TrialLeaderboardEntry(
        trial_number=record.trial_number,
        sharpe=record.sharpe,
        mean_per_era_correlation=float(
            record.metrics.get("mean_per_era_correlation", 0.0)
        ),
        std_per_era_correlation=record.metrics.get("std_per_era_correlation"),
        max_drawdown=record.metrics.get("max_drawdown"),
        model_types="+".join(record.model_types),
        elapsed_seconds=record.elapsed_seconds,
        error=record.error,
    )


def format_leaderboard(
    entries: list[TrialLeaderboardEntry],
    *,
    top_n: int = 10,
    current_trial: int | None = None,
) -> str:
    sorted_entries = sorted(entries, key=lambda e: e.sharpe, reverse=True)[:top_n]
    lines = [
        f"--- LEADERBOARD (top {top_n} by sharpe) ---",
        " Rank | Trial |  Sharpe |    Corr |  StdCorr |  MaxDD | "
        "Models              | Time",
    ]
    for rank, entry in enumerate(sorted_entries, start=1):
        std_corr = (
            f"{entry.std_per_era_correlation:8.4f}"
            if entry.std_per_era_correlation is not None
            else "     N/A"
        )
        max_dd = (
            f"{entry.max_drawdown:6.3f}" if entry.max_drawdown is not None else "   N/A"
        )
        marker = (
            " *"
            if current_trial is not None and entry.trial_number == current_trial
            else ""
        )
        lines.append(
            f" {rank:4d} | {entry.trial_number:5d} | "
            f"{entry.sharpe:7.4f} | {entry.mean_per_era_correlation:7.4f} | "
            f"{std_corr} | {max_dd} | "
            f"{entry.model_types[:19]:<19} | {entry.elapsed_seconds:4.0f}s{marker}"
        )
    if current_trial is not None:
        lines.append("* = current trial")
    return "\n".join(lines)


def print_leaderboard(
    logger: Any,
    entries: list[TrialLeaderboardEntry],
    *,
    top_n: int = 10,
    current_trial: int | None = None,
) -> None:
    logger.info(format_leaderboard(entries, top_n=top_n, current_trial=current_trial))


def save_leaderboard(path: Path, entries: list[TrialLeaderboardEntry]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_entries = sorted(entries, key=lambda e: e.sharpe, reverse=True)
    payload = [asdict(e) for e in sorted_entries]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
