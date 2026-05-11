"""Research agent backed by the Claude Code CLI (`claude -p`).

No API key required — uses the claude binary available in the Claude Code
terminal session. The agent receives trial history and returns a structured
JSON mutation decision.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from .mutations import VALID_ENSEMBLE_METHODS, VALID_MODELS, VALID_PREPROCESSORS
from .state import ResearchState

_SYSTEM_PROMPT = f"""You are an expert ML researcher specializing in Numerai stock market prediction.

You are running an automated research loop. Analyze the trial history and decide \
what single pipeline change to make next to maximize the validation Sharpe ratio.

## Pipeline Config Format
```json
{{
  "preprocessors": [
    {{"type": "StandardScaler", "params": {{}}}},
    {{"type": "GaussianNoise", "params": {{"sigma": 0.01}}}}
  ],
  "models": [
    {{"type": "XGBoost", "params": {{"max_depth": 5, "learning_rate": 0.01}}}},
    {{"type": "LightGBM", "params": {{"num_leaves": 31, "learning_rate": 0.01}}}}
  ],
  "ensemble_method": "weighted",
  "ensemble_params": {{"weights": [0.5, 0.5]}},
  "neutralize_proportion": 0.0
}}
```

## Available Models: {", ".join(VALID_MODELS)}
- XGBoost: max_depth (3–7), learning_rate (0.001–0.1), subsample (0.5–1.0)
- LightGBM: num_leaves (16–127), learning_rate (0.005–0.05), min_child_samples (100–500), colsample_bytree (0.3–0.8)
- CatBoost: depth (4–8), learning_rate (0.01–0.1), l2_leaf_reg (1–10)
- Packboost: n_worst_eras (3–10), boost_weight (0.1–0.5), n_rounds_base (300–1000), n_rounds_boost (100–300)

## Available Preprocessors: {", ".join(VALID_PREPROCESSORS)}
- StandardScaler/RobustScaler: no important params
- GaussianNoise: sigma (0.001–0.05)
- PCA/TruncatedSVD: n_components (int or float 0–1)
- VarianceSelector: keep_fraction (0.5–0.95), mode "quantile"
- LGBMImportanceSelector: keep_fraction (0.5–0.9)

## Ensemble Methods: {", ".join(VALID_ENSEMBLE_METHODS)}
- single: one model only
- weighted: ensemble_params = {{"weights": [w1, w2, ...]}} — must sum to 1.0
- stacking: ensemble_params = {{"meta_learner": "ridge" or "xgboost"}}

## Scoring
Primary: **sharpe** (mean per-era Spearman correlation ÷ std). Higher = better.
Typical good Sharpe on Numerai v5.2: 0.3–0.8.

## Strategy
- Trials 1–5: Explore broadly (different model types, CatBoost, ensemble combos).
- Trials 6–15: Build diverse ensembles combining different model families.
- Trials 15+: Exploit — tune hyperparams around the best config found.
- If stuck (no improvement in 5+ trials): use try_random_config.
- Neutralization (0.2–0.5) can reduce era noise and improve Sharpe consistency.
- Stacking with ridge often beats weighted averaging when models are diverse.

## Response Format
You MUST respond with ONLY a valid JSON object — no prose, no markdown fences.
The JSON must have exactly these keys:
  "action": one of [tune_model_params, add_model, remove_model, change_ensemble,
                    add_preprocessor, remove_preprocessor, set_neutralization,
                    try_random_config]
  "params": dict of action-specific parameters (see below)
  "reasoning": short explanation of why you chose this action

### Action params:
tune_model_params:   {{"model_index": int, "param_updates": {{...}}}}
add_model:           {{"model_type": str, "params": {{...}}}}
remove_model:        {{"model_index": int}}
change_ensemble:     {{"method": str, "params": {{...}}}}
add_preprocessor:    {{"preprocessor_type": str, "params": {{...}}, "position": int}}
remove_preprocessor: {{"position": int}}
set_neutralization:  {{"proportion": float}}
try_random_config:   {{}}

Example valid response:
{{"action": "add_model", "params": {{"model_type": "CatBoost", "params": {{"depth": 6, "learning_rate": 0.05}}}}, "reasoning": "Adding CatBoost to increase ensemble diversity; LightGBM alone is plateauing."}}"""


@dataclass
class MutationDecision:
    action_name: str
    action_kwargs: dict[str, Any]
    reasoning: str


def _format_history(state: ResearchState, max_trials: int = 30) -> str:
    trials = state.trials[-max_trials:]
    if not trials:
        return "No trials completed yet."

    lines = [
        f"{'Trial':>6} | {'Sharpe':>7} | {'Corr':>7} | {'Models':<35} | Action",
        "-" * 90,
    ]
    for t in trials:
        models_str = "+".join(t.model_types)
        corr = t.metrics.get("mean_per_era_correlation", 0.0)
        err = " [ERR]" if t.error else ""
        lines.append(
            f"{t.trial_number:>6} | {t.sharpe:>7.4f} | {corr:>7.4f} | "
            f"{models_str:<35} | {t.action_taken}{err}"
        )

    if state.best_trial:
        b = state.best_trial
        lines.append(
            f"\nBEST → trial #{b.trial_number}: sharpe={b.sharpe:.4f}, "
            f"corr={b.metrics.get('mean_per_era_correlation', 0.0):.4f}, "
            f"models={'+'.join(b.model_types)}"
        )
        lines.append(f"Best config:\n{json.dumps(b.config, indent=2)}")

    lines.append(f"\nCurrent config:\n{json.dumps(state.current_config, indent=2)}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from text output."""
    text = text.strip()
    try:
        return dict[str, Any](json.loads(text))
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return dict[str, Any](json.loads(match.group()))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in agent output:\n{text[:500]}")


def decide_next_action(
    state: ResearchState,
    model: str = "claude-sonnet-4-6",
) -> MutationDecision:
    history = _format_history(state)
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"---\n"
        f"Trial history (last {min(30, len(state.trials))} trials):\n\n"
        f"{history}\n\n"
        f"Total trials so far: {len(state.trials)}\n\n"
        f"Respond with ONLY a JSON object (no markdown, no prose)."
    )

    result = subprocess.run(
        ["claude", "-p", "--model", model, prompt],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited with code {result.returncode}: {result.stderr[:300]}"
        )

    raw = result.stdout.strip()
    decision = _extract_json(raw)

    action_name = decision.get("action", "try_random_config")
    action_kwargs = decision.get("params", {})
    reasoning = decision.get("reasoning", "")

    return MutationDecision(
        action_name=action_name,
        action_kwargs=action_kwargs,
        reasoning=reasoning,
    )
