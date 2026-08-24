from collections.abc import Iterable

MIN_PURGE_20_DAY = 8
MIN_PURGE_60_DAY = 16


def required_purge_eras(targets: Iterable[str]) -> int:
    normalized = [str(target).lower() for target in targets]
    if any(target.endswith("_60") for target in normalized):
        return MIN_PURGE_60_DAY
    return MIN_PURGE_20_DAY


def effective_purge_eras(configured: int, targets: Iterable[str]) -> int:
    if configured < 0:
        raise ValueError("configured purge must be >= 0")
    return max(configured, required_purge_eras(targets))
