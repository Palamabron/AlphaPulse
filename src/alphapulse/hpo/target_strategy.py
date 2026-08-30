import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd

from ..features.catalog import TargetCatalog, load_target_catalog
from ..pipeline.multi_target import _MIN_TRAIN_ROWS

if TYPE_CHECKING:
    import optuna

TargetMode = Literal["single", "multi_blend"]
MAX_AUXILIARY_RESAMPLE_ATTEMPTS = 3
MAX_NAN_FRACTION = 0.5

PRIMARY_TARGET = "target"


@dataclass
class TargetStrategy:
    target_mode: TargetMode
    primary_target: str
    auxiliary_targets: list[str] = field(default_factory=list)
    target_blend_method: Literal["equal", "sharpe"] = "equal"


@dataclass(frozen=True)
class TargetValidationResult:
    strategy: TargetStrategy
    ok: bool
    reason: str | None = None


def _sample_auxiliary(
    rng: random.Random,
    catalog: TargetCatalog,
    primary: str,
    *,
    max_aux: int,
) -> list[str]:
    pool = [t for t in catalog.targets if t != primary]
    if not pool or max_aux <= 0:
        return []
    n = rng.randint(1, min(max_aux, len(pool)))
    return rng.sample(pool, n)


def sample_target_strategy(
    rng: random.Random,
    catalog: TargetCatalog,
    *,
    fast: bool = False,
    primary_target: str = PRIMARY_TARGET,
) -> TargetStrategy:
    max_aux = 1 if fast else rng.randint(1, 3)
    multi_prob = 0.30 if fast else 0.35
    primary = primary_target
    if primary not in catalog.targets:
        raise ValueError(f"Primary tournament target {primary!r} missing from catalog")
    if rng.random() < multi_prob:
        aux = _sample_auxiliary(rng, catalog, primary, max_aux=max_aux)
        return TargetStrategy(
            target_mode="multi_blend",
            primary_target=primary,
            auxiliary_targets=aux,
            target_blend_method=rng.choice(["equal", "sharpe"]),
        )
    return TargetStrategy(
        target_mode="single",
        primary_target=primary,
        auxiliary_targets=[],
        target_blend_method="equal",
    )


def suggest_target_strategy(
    trial: "optuna.Trial",
    catalog: TargetCatalog,
    *,
    fast: bool = False,
    primary_target: str = PRIMARY_TARGET,
) -> TargetStrategy:
    primary = primary_target
    if primary not in catalog.targets:
        raise ValueError(f"Primary tournament target {primary!r} missing from catalog")
    mode = trial.suggest_categorical("target_mode", ["single", "multi_blend"])
    pool = [t for t in catalog.targets if t != primary]
    max_slots = 1 if fast else 3
    aux: list[str] = []
    if pool:
        for i in range(max_slots):
            choice = trial.suggest_categorical(f"auxiliary_target_{i}", ["none", *pool])
            if choice != "none" and choice not in aux:
                aux.append(choice)
    if mode == "single" or not aux:
        return TargetStrategy(
            target_mode="single",
            primary_target=primary,
            auxiliary_targets=[],
            target_blend_method="equal",
        )
    blend_method = cast(
        Literal["equal", "sharpe"],
        trial.suggest_categorical("target_blend_method", ["equal", "sharpe"]),
    )
    return TargetStrategy(
        target_mode="multi_blend",
        primary_target=primary,
        auxiliary_targets=aux,
        target_blend_method=blend_method,
    )


def _target_stats(series: pd.Series) -> tuple[int, float]:
    valid = int(series.notna().sum())
    total = len(series)
    nan_fraction = 1.0 - (valid / total) if total else 1.0
    return valid, nan_fraction


def _auxiliary_is_valid(series: pd.Series) -> bool:
    valid, nan_fraction = _target_stats(series)
    return valid >= _MIN_TRAIN_ROWS and nan_fraction <= MAX_NAN_FRACTION


def validate_target_strategy_early(
    targets_df: pd.DataFrame,
    strategy: TargetStrategy,
    *,
    catalog: TargetCatalog | None = None,
    rng: random.Random | None = None,
) -> TargetValidationResult:
    if strategy.primary_target not in targets_df.columns:
        return TargetValidationResult(
            strategy=strategy,
            ok=False,
            reason=f"primary target {strategy.primary_target!r} missing from data",
        )

    primary_valid, _ = _target_stats(targets_df[strategy.primary_target])
    if primary_valid < _MIN_TRAIN_ROWS:
        return TargetValidationResult(
            strategy=strategy,
            ok=False,
            reason=f"primary target has only {primary_valid} valid rows",
        )

    if strategy.target_mode == "single" or not strategy.auxiliary_targets:
        return TargetValidationResult(strategy=strategy, ok=True)

    invalid_aux = [
        col
        for col in strategy.auxiliary_targets
        if col not in targets_df.columns or not _auxiliary_is_valid(targets_df[col])
    ]
    if not invalid_aux:
        return TargetValidationResult(strategy=strategy, ok=True)

    if catalog is None or rng is None:
        downgraded = TargetStrategy(
            target_mode="single",
            primary_target=strategy.primary_target,
            auxiliary_targets=[],
            target_blend_method=strategy.target_blend_method,
        )
        return TargetValidationResult(
            strategy=downgraded,
            ok=True,
            reason="downgraded multi_blend to single (no resample context)",
        )

    current = strategy
    for _ in range(MAX_AUXILIARY_RESAMPLE_ATTEMPTS):
        pool = [
            t
            for t in catalog.targets
            if t != current.primary_target
            and t not in current.auxiliary_targets
            and t in targets_df.columns
        ]
        if not pool:
            break
        replacement = rng.choice(pool)
        new_aux = [
            replacement if col in invalid_aux else col
            for col in current.auxiliary_targets
        ]
        current = TargetStrategy(
            target_mode="multi_blend",
            primary_target=current.primary_target,
            auxiliary_targets=new_aux,
            target_blend_method=current.target_blend_method,
        )
        invalid_aux = [
            col
            for col in current.auxiliary_targets
            if col not in targets_df.columns or not _auxiliary_is_valid(targets_df[col])
        ]
        if not invalid_aux:
            return TargetValidationResult(strategy=current, ok=True)

    downgraded = TargetStrategy(
        target_mode="single",
        primary_target=current.primary_target,
        auxiliary_targets=[],
        target_blend_method=current.target_blend_method,
    )
    return TargetValidationResult(
        strategy=downgraded,
        ok=True,
        reason="downgraded multi_blend to single after auxiliary resample attempts",
    )


def apply_target_strategy_to_flat(
    flat: dict[str, Any], strategy: TargetStrategy
) -> dict[str, Any]:
    out = dict(flat)
    out["target_mode"] = strategy.target_mode
    out["primary_target"] = strategy.primary_target
    out["auxiliary_targets"] = list(strategy.auxiliary_targets)
    out["target_blend_method"] = strategy.target_blend_method
    return out


def strategy_from_flat(flat: dict[str, Any]) -> TargetStrategy:
    mode = flat.get("target_mode", "single")
    if mode not in ("single", "multi_blend"):
        mode = "single"
    aux = flat.get("auxiliary_targets") or []
    if not isinstance(aux, list):
        aux = []
    blend = flat.get("target_blend_method", "equal")
    if blend not in ("equal", "sharpe"):
        blend = "equal"
    return TargetStrategy(
        target_mode=mode,
        primary_target=str(flat.get("primary_target", "target")),
        auxiliary_targets=[str(a) for a in aux],
        target_blend_method=blend,
    )


def load_target_catalog_for_data(data_dir: str | Path) -> TargetCatalog:
    return load_target_catalog(data_dir)
