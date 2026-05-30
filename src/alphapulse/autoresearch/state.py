from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrialRecord:
    trial_number: int
    sharpe: float
    metrics: dict[str, float]
    config: dict[str, Any]
    model_types: list[str]
    elapsed_seconds: float
    action_taken: str
    agent_reasoning: str
    error: str | None = None
    mmc_sharpe: float | None = None
    payout_score: float | None = None


def _dominates(a: TrialRecord, b: TrialRecord) -> bool:
    """Return True if a Pareto-dominates b on (corr_sharpe, mmc_sharpe).

    a dominates b iff a is at least as good on both objectives AND strictly
    better on at least one. Trials missing mmc_sharpe are treated as -inf.
    """
    a_corr = a.sharpe
    b_corr = b.sharpe
    a_mmc = a.mmc_sharpe if a.mmc_sharpe is not None else float("-inf")
    b_mmc = b.mmc_sharpe if b.mmc_sharpe is not None else float("-inf")
    return (a_corr >= b_corr and a_mmc >= b_mmc) and (a_corr > b_corr or a_mmc > b_mmc)


@dataclass
class ParetoFront:
    """Non-dominated set of trials on the (corr_sharpe, mmc_sharpe) objectives."""

    members: list[TrialRecord] = field(default_factory=list)

    def update(self, trial: TrialRecord) -> None:
        """Add trial if it is not dominated; remove any members it dominates."""
        if trial.error is not None:
            return
        # Check if this trial is dominated by any current member
        for member in self.members:
            if _dominates(member, trial):
                return  # dominated — don't add
        # Remove members this trial dominates
        self.members = [m for m in self.members if not _dominates(trial, m)]
        self.members.append(trial)

    def best_payout(
        self, corr_weight: float = 0.75, mmc_weight: float = 2.25
    ) -> TrialRecord | None:
        """Return the Pareto member maximising corr_weight*corr + mmc_weight*mmc."""
        if not self.members:
            return None
        return max(
            self.members,
            key=lambda t: (
                corr_weight * t.sharpe
                + mmc_weight * (t.mmc_sharpe if t.mmc_sharpe is not None else 0.0)
            ),
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [
            {
                "trial_number": m.trial_number,
                "corr_sharpe": m.sharpe,
                "mmc_sharpe": m.mmc_sharpe,
                "payout_score": m.payout_score,
                "model_types": m.model_types,
            }
            for m in sorted(self.members, key=lambda t: t.sharpe, reverse=True)
        ]


@dataclass
class ResearchState:
    trials: list[TrialRecord] = field(default_factory=list)
    best_trial: TrialRecord | None = None
    current_config: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    pareto_front: ParetoFront = field(default_factory=ParetoFront)

    def add_trial(self, record: TrialRecord) -> None:
        self.trials.append(record)
        self.pareto_front.update(record)
        if record.error is None:
            # Best trial = maximise payout_score when available, else corr_sharpe
            objective = (
                record.payout_score
                if record.payout_score is not None
                else record.sharpe
            )
            best_objective = float("-inf")
            if self.best_trial is not None:
                best_objective = (
                    self.best_trial.payout_score
                    if self.best_trial.payout_score is not None
                    else self.best_trial.sharpe
                )
            if objective > best_objective:
                self.best_trial = record

    def save(self, path: Path) -> None:
        data = {
            "trials": [_to_dict(t) for t in self.trials],
            "best_trial": _to_dict(self.best_trial) if self.best_trial else None,
            "current_config": self.current_config,
            "pareto_front": self.pareto_front.to_list(),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> ResearchState:
        data = json.loads(path.read_text())
        state = cls()
        state.trials = [_from_dict(t) for t in data.get("trials", [])]
        state.best_trial = (
            _from_dict(data["best_trial"]) if data.get("best_trial") else None
        )
        state.current_config = data.get("current_config", {})
        # Rebuild Pareto front from saved trials
        for t in state.trials:
            state.pareto_front.update(t)
        return state


def _to_dict(t: TrialRecord) -> dict[str, Any]:
    return {
        "trial_number": t.trial_number,
        "sharpe": t.sharpe,
        "metrics": t.metrics,
        "config": t.config,
        "model_types": t.model_types,
        "elapsed_seconds": t.elapsed_seconds,
        "action_taken": t.action_taken,
        "agent_reasoning": t.agent_reasoning,
        "error": t.error,
        "mmc_sharpe": t.mmc_sharpe,
        "payout_score": t.payout_score,
    }


def _from_dict(d: dict[str, Any]) -> TrialRecord:
    return TrialRecord(
        trial_number=d["trial_number"],
        sharpe=d["sharpe"],
        metrics=d.get("metrics", {}),
        config=d.get("config", {}),
        model_types=d.get("model_types", []),
        elapsed_seconds=d.get("elapsed_seconds", 0.0),
        action_taken=d.get("action_taken", "initial"),
        agent_reasoning=d.get("agent_reasoning", ""),
        error=d.get("error"),
        mmc_sharpe=d.get("mmc_sharpe"),
        payout_score=d.get("payout_score"),
    )
