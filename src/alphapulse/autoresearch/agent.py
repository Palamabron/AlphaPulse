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
what single pipeline change to make next to maximize the **legacy AlphaPulse HPO proxy**.

## Historical AlphaPulse proxy (not official Numerai payout)
payout_score = 0.75 * CORR_sharpe + 2.25 * MMC_sharpe

where:
- CORR_sharpe = Sharpe ratio of legacy per-era Spearman correlations
- MMC_sharpe = Sharpe ratio of the legacy residual-correlation MMC approximation

This compatibility objective is not official Numerai CORR, MMC, Season score, or payout.

When optimizing, consider both objectives. The Pareto front shows configs that are \
non-dominated on (CORR_sharpe, MMC_sharpe). MMC rewards models that are different \
from the Numerai meta model — diversity matters more than raw accuracy.

## Tactics for each objective
**To push CORR_sharpe:**
- Add/tune gradient boosting models (XGBoost, LightGBM, CatBoost)
- Use PackboostModel to focus on worst-performing eras
- Use EraEnsembleModel for era-specific sub-models
- Reduce neutralization proportion

**To push MMC_sharpe (originality):**
- Increase neutralization proportion (0.3–0.7) — removes common factor exposure
- Add models from different families (Ridge, RandomForest alongside GBMs)
- Use EraStableSelector to focus on uniquely informative features
- Stacking with diverse base learners

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
- Ridge: alpha (1–1000) — linear model, very different from tree models
- RandomForest/ExtraTrees: n_estimators (100–500), min_samples_leaf (50–500)

## Available Preprocessors: {", ".join(VALID_PREPROCESSORS)}
- StandardScaler/RobustScaler: no important params
- GaussianNoise: sigma (0.001–0.05)
- PCA/TruncatedSVD: n_components (int or float 0–1)
- VarianceSelector: keep_fraction (0.5–0.95), mode "quantile"
- LGBMImportanceSelector: keep_fraction (0.5–0.9)
- EraStableSelector: keep_fraction (0.3–0.7), stability_weight (0–1) — selects features stable across eras

## Ensemble Methods: {", ".join(VALID_ENSEMBLE_METHODS)}
- single: one model only
- weighted: ensemble_params = {{"weights": [w1, w2, ...]}} — must sum to 1.0
- stacking: ensemble_params = {{"meta_learner": "ridge" or "xgboost"}}

## Strategy
- Trials 1–5: Explore broadly. Try CORR-focused config AND an MMC-focused config.
- Trials 6–15: Build diverse ensembles. Mix tree models with Ridge/RF for originality.
- Trials 15+: Expand the Pareto front. If CORR is high but MMC is low, add neutralization. If MMC is high but CORR is low, add stronger GBMs.
- If stuck (no legacy-proxy improvement in 5+ trials): use try_random_config.
- Neutralization (0.2–0.5) typically improves MMC at slight cost to CORR.

## Response Format
You MUST respond with ONLY a valid JSON object — no prose, no markdown fences.
The JSON must have exactly these keys:
  "action": one of [tune_model_params, add_model, remove_model, change_ensemble,
                    add_preprocessor, remove_preprocessor, set_neutralization,
                    try_random_config]
  "params": dict of action-specific parameters (see below)
  "reasoning": short explanation of why you chose this action
  "target_objective": one of ["corr", "mmc", "balanced"] — which axis you are targeting

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
{{"action": "set_neutralization", "params": {{"proportion": 0.4}}, "reasoning": "CORR is strong but MMC is low; adding neutralization should improve originality.", "target_objective": "mmc"}}"""


@dataclass
class MutationDecision:
    action_name: str
    action_kwargs: dict[str, Any]
    reasoning: str
    target_objective: str = "balanced"


def _format_history(state: ResearchState, max_trials: int = 30) -> str:
    trials = state.trials[-max_trials:]
    if not trials:
        return "No trials completed yet."

    lines = [
        f"{'Trial':>6} | {'Sharpe':>7} | {'MMC_S':>7} | {'LegacyProxy':>11} | {'Models':<30} | Action",
        "-" * 95,
    ]
    for t in trials:
        models_str = "+".join(t.model_types)
        mmc_s = f"{t.mmc_sharpe:>7.4f}" if t.mmc_sharpe is not None else "    N/A"
        payout = f"{t.payout_score:>7.4f}" if t.payout_score is not None else "    N/A"
        err = " [ERR]" if t.error else ""
        lines.append(
            f"{t.trial_number:>6} | {t.sharpe:>7.4f} | {mmc_s} | {payout} | "
            f"{models_str:<30} | {t.action_taken}{err}"
        )

    if state.best_trial:
        b = state.best_trial
        payout_str = (
            f", legacy_proxy={b.payout_score:.4f}" if b.payout_score is not None else ""
        )
        mmc_str = f", mmc_sharpe={b.mmc_sharpe:.4f}" if b.mmc_sharpe is not None else ""
        lines.append(
            f"\nBEST → trial #{b.trial_number}: corr_sharpe={b.sharpe:.4f}"
            f"{mmc_str}{payout_str}, models={'+'.join(b.model_types)}"
        )
        lines.append(f"Best config:\n{json.dumps(b.config, indent=2)}")

    pareto = state.pareto_front.members
    if pareto:
        lines.append(
            f"\nPARETO FRONT ({len(pareto)} configs — non-dominated on CORR+MMC):"
        )
        lines.append(
            f"{'Trial':>6} | {'CORR_S':>7} | {'MMC_S':>7} | {'LegacyProxy':>11}"
        )
        lines.append("-" * 40)
        for m in sorted(pareto, key=lambda t: t.sharpe, reverse=True):
            mmc_s = f"{m.mmc_sharpe:>7.4f}" if m.mmc_sharpe is not None else "    N/A"
            payout = (
                f"{m.payout_score:>7.4f}" if m.payout_score is not None else "    N/A"
            )
            lines.append(f"{m.trial_number:>6} | {m.sharpe:>7.4f} | {mmc_s} | {payout}")

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


_MAX_RETRIES = 2


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

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            result = subprocess.run(
                ["claude", "-p", "--model", model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"claude CLI exited with code {result.returncode}: "
                    f"{result.stderr[:300]}"
                )

            raw = result.stdout.strip()
            decision = _extract_json(raw)

            action_name = decision.get("action", "try_random_config")
            action_kwargs = decision.get("params", {})
            reasoning = decision.get("reasoning", "")
            target_objective = decision.get("target_objective", "balanced")

            return MutationDecision(
                action_name=action_name,
                action_kwargs=action_kwargs,
                reasoning=reasoning,
                target_objective=str(target_objective),
            )
        except (subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                import time

                time.sleep(2**attempt)

    raise RuntimeError(
        f"Agent failed after {_MAX_RETRIES + 1} attempts: {last_error}"
    ) from last_error
