import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

LEGACY_EXCLUDED = frozenset(
    {"v2_equivalent_features", "v3_equivalent_features", "fncv3_features"}
)
SIZE_GROUPS = frozenset({"small", "medium", "all"})
STAT_GROUPS = frozenset(
    {
        "intelligence",
        "charisma",
        "strength",
        "dexterity",
        "constitution",
        "wisdom",
        "agility",
        "serenity",
        "sunshine",
        "rain",
        "midnight",
        "faith",
    }
)
_HORIZON_RE = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class FeatureCatalog:
    feature_sets: dict[str, list[str]]
    searchable_names: list[str]

    def columns(self, name: str) -> list[str]:
        if name not in self.feature_sets:
            raise KeyError(f"Unknown feature group: {name!r}")
        return list(self.feature_sets[name])

    def union(self, names: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for name in names:
            for col in self.columns(name):
                if col not in seen:
                    seen.add(col)
                    out.append(col)
        return out

    def category(self, name: str) -> str:
        if name in SIZE_GROUPS:
            return "size"
        if name in STAT_GROUPS:
            return "stat"
        return "other"


@dataclass(frozen=True)
class TargetCatalog:
    targets: list[str]

    def parse_horizon(self, target: str) -> int | None:
        if target == "target":
            return 20
        match = _HORIZON_RE.search(target)
        if match:
            return int(match.group(1))
        return None

    def valid_targets(self, df: pd.DataFrame, cols: list[str]) -> list[str]:
        return [c for c in cols if c in df.columns and c in self.targets]


def _load_features_json(data_dir: Path) -> dict:
    path = data_dir / "features.json"
    if not path.exists():
        raise FileNotFoundError(f"Expected {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid features.json at {path}")
    return data


@lru_cache(maxsize=8)
def load_feature_catalog(data_dir: str | Path) -> FeatureCatalog:
    data = _load_features_json(Path(data_dir))
    raw_sets = data.get("feature_sets", {})
    if not isinstance(raw_sets, dict):
        raise ValueError("features.json missing feature_sets dict")

    feature_sets: dict[str, list[str]] = {}
    for name, cols in raw_sets.items():
        if name in LEGACY_EXCLUDED:
            continue
        if not isinstance(name, str) or not isinstance(cols, list):
            continue
        if all(isinstance(c, str) for c in cols):
            feature_sets[name] = cols

    searchable = sorted(
        name for name in feature_sets if name in SIZE_GROUPS or name in STAT_GROUPS
    )
    return FeatureCatalog(feature_sets=feature_sets, searchable_names=searchable)


@lru_cache(maxsize=8)
def load_target_catalog(data_dir: str | Path) -> TargetCatalog:
    data = _load_features_json(Path(data_dir))
    raw_targets = data.get("targets", [])
    if not isinstance(raw_targets, list):
        raise ValueError("features.json missing targets list")
    targets = [t for t in raw_targets if isinstance(t, str)]
    return TargetCatalog(targets=targets)
