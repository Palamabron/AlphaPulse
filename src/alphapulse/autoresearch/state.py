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


@dataclass
class ResearchState:
    trials: list[TrialRecord] = field(default_factory=list)
    best_trial: TrialRecord | None = None
    current_config: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    def add_trial(self, record: TrialRecord) -> None:
        self.trials.append(record)
        if record.error is None and (
            self.best_trial is None or record.sharpe > self.best_trial.sharpe
        ):
            self.best_trial = record

    def save(self, path: Path) -> None:
        data = {
            "trials": [_to_dict(t) for t in self.trials],
            "best_trial": _to_dict(self.best_trial) if self.best_trial else None,
            "current_config": self.current_config,
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
    )
