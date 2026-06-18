from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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
    payout_score: float | None = None
    mmc_sharpe: float | None = None
    val_corr_sharpe: float | None = None
    val_mean_per_era_correlation: float | None = None
    holdout_corr_sharpe: float | None = None


def _model_types_from_flat(flat: dict[str, Any]) -> str:
    num = flat.get("num_models", 1)
    types = [
        flat.get("model_1_type", "?"),
        flat.get("model_2_type", "?"),
        flat.get("model_3_type", "?"),
    ][:num]
    return "+".join(str(t) for t in types)


def _finite_optional(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _payout_from_result(
    payout_score: float | None,
    metrics: dict[str, Any],
) -> float | None:
    if payout_score is not None:
        return _finite_optional(payout_score)
    return _finite_optional(metrics.get("payout_score"))


def _mmc_from_result(
    mmc_sharpe: float | None,
    metrics: dict[str, Any],
) -> float | None:
    if mmc_sharpe is not None:
        return _finite_optional(mmc_sharpe)
    return _finite_optional(metrics.get("mmc_sharpe"))


def _rank_score(entry: TrialLeaderboardEntry) -> float:
    if entry.payout_score is not None:
        return entry.payout_score
    return entry.sharpe


def _uses_payout_score(entries: list[TrialLeaderboardEntry]) -> bool:
    return any(e.payout_score is not None for e in entries)


def _fmt_metric(value: float | None, width: int = 7, precision: int = 4) -> str:
    if value is None:
        return "N/A".rjust(width)
    if value == float("-inf"):
        return "-inf".rjust(width)
    if value == float("inf"):
        return " inf".rjust(width)
    if not np.isfinite(value):
        return "N/A".rjust(width)
    return f"{value:{width}.{precision}f}"


def entry_from_hpo_result(result: TrialResult) -> TrialLeaderboardEntry:
    metrics = result.metrics
    holdout_sharpe = _finite_optional(metrics.get("holdout_corr_sharpe"))
    if holdout_sharpe is None:
        holdout_sharpe = _finite_optional(result.sharpe)
    holdout_corr = _finite_optional(metrics.get("holdout_mean_per_era_correlation"))
    if holdout_corr is None:
        holdout_corr = float(metrics.get("mean_per_era_correlation", 0.0))
    holdout_std = _finite_optional(metrics.get("holdout_std_per_era_correlation"))
    if holdout_std is None:
        holdout_std = _finite_optional(metrics.get("std_per_era_correlation"))
    holdout_dd = _finite_optional(metrics.get("holdout_max_drawdown"))
    if holdout_dd is None:
        holdout_dd = _finite_optional(metrics.get("max_drawdown"))
    return TrialLeaderboardEntry(
        trial_number=result.trial_number,
        sharpe=holdout_sharpe if holdout_sharpe is not None else result.sharpe,
        mean_per_era_correlation=holdout_corr,
        std_per_era_correlation=holdout_std,
        max_drawdown=holdout_dd,
        model_types=_model_types_from_flat(result.params),
        elapsed_seconds=result.elapsed_seconds,
        error=result.error,
        payout_score=_payout_from_result(result.payout_score, metrics),
        mmc_sharpe=_mmc_from_result(result.mmc_sharpe, metrics),
        val_corr_sharpe=_finite_optional(metrics.get("val_corr_sharpe")),
        val_mean_per_era_correlation=_finite_optional(
            metrics.get("val_mean_per_era_correlation")
        ),
        holdout_corr_sharpe=holdout_sharpe,
    )


def entry_from_trial_record(record: TrialRecord) -> TrialLeaderboardEntry:
    metrics = record.metrics
    holdout_sharpe = _finite_optional(metrics.get("holdout_corr_sharpe"))
    if holdout_sharpe is None:
        holdout_sharpe = _finite_optional(record.sharpe)
    holdout_corr = _finite_optional(metrics.get("holdout_mean_per_era_correlation"))
    if holdout_corr is None:
        holdout_corr = float(metrics.get("mean_per_era_correlation", 0.0))
    holdout_std = _finite_optional(metrics.get("holdout_std_per_era_correlation"))
    if holdout_std is None:
        holdout_std = _finite_optional(metrics.get("std_per_era_correlation"))
    holdout_dd = _finite_optional(metrics.get("holdout_max_drawdown"))
    if holdout_dd is None:
        holdout_dd = _finite_optional(metrics.get("max_drawdown"))
    return TrialLeaderboardEntry(
        trial_number=record.trial_number,
        sharpe=holdout_sharpe if holdout_sharpe is not None else record.sharpe,
        mean_per_era_correlation=holdout_corr,
        std_per_era_correlation=holdout_std,
        max_drawdown=holdout_dd,
        model_types="+".join(record.model_types),
        elapsed_seconds=record.elapsed_seconds,
        error=record.error,
        payout_score=_payout_from_result(record.payout_score, metrics),
        mmc_sharpe=_mmc_from_result(record.mmc_sharpe, metrics),
        val_corr_sharpe=_finite_optional(metrics.get("val_corr_sharpe")),
        val_mean_per_era_correlation=_finite_optional(
            metrics.get("val_mean_per_era_correlation")
        ),
        holdout_corr_sharpe=holdout_sharpe,
    )


def format_leaderboard(
    entries: list[TrialLeaderboardEntry],
    *,
    top_n: int = 10,
    current_trial: int | None = None,
) -> str:
    by_payout = _uses_payout_score(entries)
    sorted_entries = sorted(entries, key=_rank_score, reverse=True)[:top_n]
    if by_payout:
        header = (
            f"--- LEADERBOARD (top {top_n} by payout on validation) ---\n"
            " Payout = 0.75*ValidationSharpe + 2.25*ValidationMmcSharpe\n"
            " Rank | Trial |   Payout | ValidationSharpe | ValidationMmcSharpe | "
            "ValidationMeanCorr | HoldoutSharpe | HoldoutMeanCorr | "
            "Models              | Time"
        )
    else:
        header = (
            f"--- LEADERBOARD (top {top_n} by holdout corr_sharpe) ---\n"
            " Rank | Trial | HoldoutSharpe | HoldoutMeanCorr | HoldoutStdCorr | "
            "HoldoutMaxDrawdown | Models              | Time"
        )
    lines = [header]
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
        if by_payout:
            lines.append(
                f" {rank:4d} | {entry.trial_number:5d} | "
                f"{_fmt_metric(entry.payout_score)} | "
                f"{_fmt_metric(entry.val_corr_sharpe)} | "
                f"{_fmt_metric(entry.mmc_sharpe)} | "
                f"{_fmt_metric(entry.val_mean_per_era_correlation)} | "
                f"{_fmt_metric(entry.holdout_corr_sharpe)} | "
                f"{_fmt_metric(entry.mean_per_era_correlation)} | "
                f"{entry.model_types[:19]:<19} | {entry.elapsed_seconds:4.0f}s{marker}"
            )
        else:
            lines.append(
                f" {rank:4d} | {entry.trial_number:5d} | "
                f"{_fmt_metric(entry.holdout_corr_sharpe)} | "
                f"{_fmt_metric(entry.mean_per_era_correlation)} | "
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
    sorted_entries = sorted(entries, key=_rank_score, reverse=True)
    payload = [asdict(e) for e in sorted_entries]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
