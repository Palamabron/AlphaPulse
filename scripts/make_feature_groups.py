"""Generate YAML ``features.groups`` blocks from Numerai's ``features.json``.

Usage examples::

    # Print groups for the 'small' and 'medium' sets
    uv run python scripts/make_feature_groups.py --feature-sets small medium

    # Write directly into an existing experiment YAML (replaces the groups section)
    uv run python scripts/make_feature_groups.py \\
        --feature-sets small \\
        --output-yaml experiments/my_experiment.yaml

    # List all available feature sets
    uv run python scripts/make_feature_groups.py --list-sets
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tyro
import yaml


def _load_feature_sets(features_json: Path) -> dict[str, list[str]]:
    with open(features_json, encoding="utf-8") as f:
        data = json.load(f)
    if "feature_sets" in data:
        return {k: list(v) for k, v in data["feature_sets"].items()}
    # Older format: top-level dict of set_name → list
    return {k: list(v) for k, v in data.items() if isinstance(v, list)}


def _build_groups_yaml(
    selected: dict[str, list[str]],
    indent: int = 4,
) -> str:
    lines = ["features:", "  groups:"]
    for group_name, columns in selected.items():
        lines.append(f"    {group_name}:")
        for col in columns:
            lines.append(f"      - {col}")
    return "\n".join(lines)


def _patch_experiment_yaml(output_yaml: Path, groups: dict[str, list[str]]) -> None:
    with open(output_yaml, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    if "features" not in doc or not isinstance(doc["features"], dict):
        doc["features"] = {}
    doc["features"]["groups"] = groups

    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main(
    feature_sets: list[str] | None = None,
    features_json: Path = Path("data/v5.2/features.json"),
    output_yaml: Path | None = None,
    list_sets: bool = False,
) -> None:
    """Print or inject YAML ``features.groups`` from Numerai's features.json.

    Args:
        feature_sets: One or more set names to include (e.g. small medium).
            When omitted and --list-sets is False, all sets are printed.
        features_json: Path to the Numerai features.json file.
        output_yaml: Optional experiment YAML to patch in-place with the groups.
        list_sets: Print available set names and exit.
    """
    if not features_json.exists():
        print(f"ERROR: {features_json} not found.", file=sys.stderr)
        print(
            "Download the dataset first: uv run python scripts/download_dataset.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    all_sets = _load_feature_sets(features_json)

    if list_sets:
        print("Available feature sets:")
        for name, cols in all_sets.items():
            print(f"  {name}: {len(cols)} features")
        return

    chosen_names = feature_sets or list(all_sets.keys())
    missing = [n for n in chosen_names if n not in all_sets]
    if missing:
        print(f"ERROR: Unknown feature set(s): {missing}", file=sys.stderr)
        print(f"Available: {list(all_sets.keys())}", file=sys.stderr)
        raise SystemExit(1)

    groups = {name: all_sets[name] for name in chosen_names}

    if output_yaml is not None:
        if not output_yaml.exists():
            print(f"ERROR: {output_yaml} not found.", file=sys.stderr)
            raise SystemExit(1)
        _patch_experiment_yaml(output_yaml, groups)
        total = sum(len(v) for v in groups.values())
        print(f"Patched {output_yaml}: {len(groups)} group(s), {total} total columns.")
    else:
        print(_build_groups_yaml(groups))


if __name__ == "__main__":
    tyro.cli(main)
