from typing import Literal

OptimizationDirection = Literal["minimize", "maximize"]
OptimizationMode = Literal["min", "max"]

_MINIMIZED_OBJECTIVES = frozenset({"max_drawdown"})


def optimization_direction(objective: str) -> OptimizationDirection:
    return "minimize" if objective in _MINIMIZED_OBJECTIVES else "maximize"


def optimization_mode(objective: str) -> OptimizationMode:
    return "min" if optimization_direction(objective) == "minimize" else "max"


def worst_optimization_score(objective: str) -> float:
    if optimization_direction(objective) == "minimize":
        return float("inf")
    return float("-inf")


def is_better_optimization_score(
    candidate: float,
    incumbent: float,
    objective: str,
) -> bool:
    if optimization_direction(objective) == "minimize":
        return candidate < incumbent
    return candidate > incumbent
