from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from ..autoresearch.state import TrialRecord
    from ..hpo.objective import TrialResult

BestCriteria = Literal["objective", "robust_payout"]
NEGATIVE_HOLDOUT_PAYOUT_FACTOR = 0.25
ROBUST_CONSISTENCY_FLOOR = 0.5


def compute_robust_payout_score(
    payout_score: float | None,
    val_corr_sharpe: float | None,
    holdout_corr_sharpe: float | None,
) -> float | None:
    """Down-rank high validation payout when holdout CORR does not confirm it."""
    payout = _finite_optional(payout_score)
    if payout is None:
        return None
    holdout = _finite_optional(holdout_corr_sharpe)
    if holdout is None:
        return payout
    if holdout <= 0.0:
        return _apply_score_penalty(payout, NEGATIVE_HOLDOUT_PAYOUT_FACTOR)
    val = _finite_optional(val_corr_sharpe)
    if val is None or val <= 0.0:
        return _apply_score_penalty(payout, min(1.0, holdout / 0.2))
    consistency = min(1.0, holdout / val)
    return _apply_score_penalty(
        payout,
        ROBUST_CONSISTENCY_FLOOR + (1.0 - ROBUST_CONSISTENCY_FLOOR) * consistency,
    )


def _apply_score_penalty(score: float, factor: float) -> float:
    bounded_factor = max(np.finfo(float).eps, min(1.0, factor))
    if score >= 0.0:
        return score * bounded_factor
    return score / bounded_factor


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
    robust_payout_score: float | None = None


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


def _robust_rank_score(entry: TrialLeaderboardEntry) -> float:
    if entry.robust_payout_score is not None:
        return entry.robust_payout_score
    return _rank_score(entry)


def _uses_payout_score(entries: list[TrialLeaderboardEntry]) -> bool:
    return any(e.payout_score is not None for e in entries)


def _uses_robust_payout(entries: list[TrialLeaderboardEntry]) -> bool:
    return any(
        e.robust_payout_score is not None and e.holdout_corr_sharpe is not None
        for e in entries
    )


def selection_score_from_metrics(
    metrics: dict[str, Any],
    *,
    objective: str,
    criteria: BestCriteria = "objective",
) -> float:
    raw_objective = metrics.get(objective)
    if raw_objective is None:
        return float("nan")
    objective_score = float(raw_objective)
    if criteria != "robust_payout" or objective != "payout_score":
        return objective_score
    robust = compute_robust_payout_score(
        _finite_optional(metrics.get("payout_score")),
        _finite_optional(metrics.get("val_corr_sharpe")),
        _finite_optional(metrics.get("holdout_corr_sharpe")),
    )
    if robust is not None:
        return robust
    return objective_score


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
    payout = _payout_from_result(result.payout_score, metrics)
    val_corr_sharpe = _finite_optional(metrics.get("val_corr_sharpe"))
    return TrialLeaderboardEntry(
        trial_number=result.trial_number,
        sharpe=holdout_sharpe if holdout_sharpe is not None else result.sharpe,
        mean_per_era_correlation=holdout_corr,
        std_per_era_correlation=holdout_std,
        max_drawdown=holdout_dd,
        model_types=_model_types_from_flat(result.params),
        elapsed_seconds=result.elapsed_seconds,
        error=result.error,
        payout_score=payout,
        mmc_sharpe=_mmc_from_result(result.mmc_sharpe, metrics),
        val_corr_sharpe=val_corr_sharpe,
        val_mean_per_era_correlation=_finite_optional(
            metrics.get("val_mean_per_era_correlation")
        ),
        holdout_corr_sharpe=holdout_sharpe,
        robust_payout_score=compute_robust_payout_score(
            payout, val_corr_sharpe, holdout_sharpe
        ),
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
    payout = _payout_from_result(record.payout_score, metrics)
    val_corr_sharpe = _finite_optional(metrics.get("val_corr_sharpe"))
    return TrialLeaderboardEntry(
        trial_number=record.trial_number,
        sharpe=holdout_sharpe if holdout_sharpe is not None else record.sharpe,
        mean_per_era_correlation=holdout_corr,
        std_per_era_correlation=holdout_std,
        max_drawdown=holdout_dd,
        model_types="+".join(record.model_types),
        elapsed_seconds=record.elapsed_seconds,
        error=record.error,
        payout_score=payout,
        mmc_sharpe=_mmc_from_result(record.mmc_sharpe, metrics),
        val_corr_sharpe=val_corr_sharpe,
        val_mean_per_era_correlation=_finite_optional(
            metrics.get("val_mean_per_era_correlation")
        ),
        holdout_corr_sharpe=holdout_sharpe,
        robust_payout_score=compute_robust_payout_score(
            payout, val_corr_sharpe, holdout_sharpe
        ),
    )


def _format_payout_table(
    entries: list[TrialLeaderboardEntry],
    *,
    top_n: int,
    current_trial: int | None,
    title: str,
    sort_key: Callable[[TrialLeaderboardEntry], float],
    score_label: str,
    score_getter: Callable[[TrialLeaderboardEntry], float | None],
) -> list[str]:
    header = (
        f"--- {title} ---\n"
        " LegacyProxy = 0.75*ValidationSharpe + 2.25*ValidationMmcSharpe\n"
        f" Rank | Trial | {score_label:>11} | ValidationSharpe | "
        "ValidationMmcSharpe | ValidationMeanCorr | HoldoutSharpe | "
        "HoldoutMeanCorr | Models              | Time"
    )
    lines = [header]
    sorted_entries = sorted(entries, key=sort_key, reverse=True)[:top_n]
    for rank, entry in enumerate(sorted_entries, start=1):
        marker = (
            " *"
            if current_trial is not None and entry.trial_number == current_trial
            else ""
        )
        lines.append(
            f" {rank:4d} | {entry.trial_number:5d} | "
            f"{_fmt_metric(score_getter(entry))} | "
            f"{_fmt_metric(entry.val_corr_sharpe)} | "
            f"{_fmt_metric(entry.mmc_sharpe)} | "
            f"{_fmt_metric(entry.val_mean_per_era_correlation)} | "
            f"{_fmt_metric(entry.holdout_corr_sharpe)} | "
            f"{_fmt_metric(entry.mean_per_era_correlation)} | "
            f"{entry.model_types[:19]:<19} | {entry.elapsed_seconds:4.0f}s{marker}"
        )
    return lines


def format_leaderboard(
    entries: list[TrialLeaderboardEntry],
    *,
    top_n: int = 10,
    current_trial: int | None = None,
) -> str:
    by_payout = _uses_payout_score(entries)
    lines: list[str] = []
    if by_payout:
        lines.extend(
            _format_payout_table(
                entries,
                top_n=top_n,
                current_trial=current_trial,
                title=f"LEADERBOARD (top {top_n} by legacy proxy on validation)",
                sort_key=_rank_score,
                score_label="LegacyProxy",
                score_getter=lambda entry: entry.payout_score,
            )
        )
        if _uses_robust_payout(entries):
            lines.append("")
            lines.extend(
                _format_payout_table(
                    entries,
                    top_n=top_n,
                    current_trial=current_trial,
                    title=(
                        f"LEADERBOARD (top {top_n} by robust legacy proxy: "
                        "proxy penalized when holdout CORR is weak vs validation)"
                    ),
                    sort_key=_robust_rank_score,
                    score_label="RobustProxy",
                    score_getter=lambda entry: entry.robust_payout_score,
                )
            )
    else:
        header = (
            f"--- LEADERBOARD (top {top_n} by holdout corr_sharpe) ---\n"
            " Rank | Trial | HoldoutSharpe | HoldoutMeanCorr | HoldoutStdCorr | "
            "HoldoutMaxDrawdown | Models              | Time"
        )
        lines = [header]
        sorted_entries = sorted(entries, key=_rank_score, reverse=True)[:top_n]
        for rank, entry in enumerate(sorted_entries, start=1):
            std_corr = (
                f"{entry.std_per_era_correlation:8.4f}"
                if entry.std_per_era_correlation is not None
                else "     N/A"
            )
            max_dd = (
                f"{entry.max_drawdown:6.3f}"
                if entry.max_drawdown is not None
                else "   N/A"
            )
            marker = (
                " *"
                if current_trial is not None and entry.trial_number == current_trial
                else ""
            )
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
