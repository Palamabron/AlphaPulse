import json
from pathlib import Path

import pytest

from alphapulse.features.catalog import (
    LEGACY_EXCLUDED,
    load_feature_catalog,
    load_target_catalog,
)


@pytest.fixture
def features_json(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload = {
        "feature_sets": {
            "small": ["f_a", "f_b"],
            "medium": ["f_a", "f_b", "f_c"],
            "all": ["f_a", "f_b", "f_c", "f_d"],
            "strength": ["f_a", "f_c"],
            "v2_equivalent_features": ["legacy_1"],
        },
        "targets": ["target", "target_alpha_20", "target_cyrusd_60"],
    }
    (data_dir / "features.json").write_text(json.dumps(payload), encoding="utf-8")
    return data_dir


def test_feature_catalog_excludes_legacy(features_json: Path) -> None:
    catalog = load_feature_catalog(features_json)
    assert "v2_equivalent_features" not in catalog.feature_sets
    assert "v2_equivalent_features" not in catalog.searchable_names
    assert LEGACY_EXCLUDED.isdisjoint(catalog.searchable_names)
    assert set(catalog.searchable_names) == {
        "small",
        "medium",
        "all",
        "strength",
    }


def test_feature_catalog_union_dedup(features_json: Path) -> None:
    catalog = load_feature_catalog(features_json)
    union = catalog.union(["small", "strength"])
    assert union == ["f_a", "f_b", "f_c"]


def test_target_catalog_load(features_json: Path) -> None:
    catalog = load_target_catalog(features_json)
    assert "target" in catalog.targets
    assert catalog.parse_horizon("target_alpha_20") == 20
    assert catalog.parse_horizon("target") == 20
