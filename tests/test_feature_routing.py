import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphapulse.features.catalog import load_feature_catalog
from alphapulse.hpo.builder import build_pipeline_or_multi
from alphapulse.hpo.feature_routing import (
    MAX_ROUTED_FEATURES,
    merge_routing_into_pipeline_config,
    resolve_feature_routing,
    sample_feature_routing,
)
from alphapulse.hpo.search_space import resolve_flat_config
from alphapulse.pipeline.multihead import MultiHeadPipeline


@pytest.fixture
def catalog_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload = {
        "feature_sets": {
            "small": ["f_a", "f_b"],
            "medium": ["f_a", "f_b", "f_c"],
            "strength": ["f_a", "f_c"],
            "constitution": ["f_b", "f_c"],
        },
        "targets": ["target"],
    }
    (data_dir / "features.json").write_text(json.dumps(payload), encoding="utf-8")
    return data_dir


def _base_flat(num_models: int = 1) -> dict:
    return {
        "num_models": num_models,
        "model_1_type": "XGBoost",
        "model_2_type": "LightGBM",
        "model_3_type": "XGBoost",
        "scaler_type": "StandardScaler",
        "use_packboost": False,
        "ensemble_method": "single" if num_models == 1 else "weighted",
        "xgb_max_depth": 3,
        "xgb_learning_rate": 0.05,
        "xgb_n_rounds": 10,
        "xgb_early_stopping": 5,
        "lgbm_num_leaves": 16,
        "lgbm_learning_rate": 0.05,
        "lgbm_n_rounds": 10,
        "lgbm_early_stopping": 5,
        "use_neutralization": False,
    }


def test_resolve_simple_path(catalog_dir: Path) -> None:
    catalog = load_feature_catalog(catalog_dir)
    flat = {
        **_base_flat(1),
        "use_feature_routing": True,
        "active_groups": ["small", "strength"],
        "model_1_groups": ["small", "strength"],
        "model_1_lane": 0,
        "lane_0_steps": [],
    }
    routing = resolve_feature_routing(flat, catalog)
    assert routing.build_path == "simple"
    assert routing.feature_columns == ["f_a", "f_b", "f_c"]


def test_resolve_grouped_path_single_model_with_lane_steps(catalog_dir: Path) -> None:
    catalog = load_feature_catalog(catalog_dir)
    flat = {
        **_base_flat(1),
        "use_feature_routing": True,
        "active_groups": ["small"],
        "model_1_groups": ["small"],
        "model_1_lane": 0,
        "lane_0_steps": ["VarianceFeatureSelector"],
    }
    routing = resolve_feature_routing(flat, catalog)
    assert routing.build_path == "grouped"
    assert routing.pipeline_config_patch["preprocessors"][0]["type"] == "Grouped"
    assert routing.pipeline_config_patch["models"] == [{}]

    cfg = merge_routing_into_pipeline_config(resolve_flat_config(flat), routing)
    pipeline = build_pipeline_or_multi(
        cfg,
        feature_columns=routing.feature_columns,
        feature_groups=routing.feature_groups,
    )
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 2)), columns=["f_a", "f_b"])
    y = pd.Series(rng.normal(size=40), index=X.index)
    pipeline.fit(X, y)
    assert np.isfinite(pipeline.predict(X)).all()


def test_resolve_multihead_path(catalog_dir: Path) -> None:
    catalog = load_feature_catalog(catalog_dir)
    flat = {
        **_base_flat(2),
        "use_feature_routing": True,
        "active_groups": ["small", "strength"],
        "model_1_groups": ["small"],
        "model_2_groups": ["strength"],
        "model_1_lane": 0,
        "model_2_lane": 0,
        "lane_0_steps": [],
    }
    routing = resolve_feature_routing(flat, catalog)
    assert routing.build_path == "multihead"
    cfg = merge_routing_into_pipeline_config(resolve_flat_config(flat), routing)
    pipeline = build_pipeline_or_multi(
        cfg,
        feature_columns=routing.feature_columns,
        feature_groups=routing.feature_groups,
    )
    assert isinstance(pipeline, MultiHeadPipeline)


def test_sample_feature_routing_fragment(catalog_dir: Path) -> None:
    catalog = load_feature_catalog(catalog_dir)
    fragment = sample_feature_routing(random.Random(0), catalog, 2, fast=True)
    assert fragment["use_feature_routing"] is True
    assert fragment["active_groups"]
    assert fragment["active_groups_count"] == len(fragment["active_groups"])
    assert fragment["routed_feature_count"] <= MAX_ROUTED_FEATURES


def test_sample_feature_routing_respects_feature_limit(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload = {
        "feature_sets": {
            "small": [f"f_{i}" for i in range(100)],
            "medium": [f"f_{i}" for i in range(800)],
            "all": [f"f_{i}" for i in range(1200)],
            "strength": [f"f_{i}" for i in range(400, 900)],
            "rain": [f"f_{i}" for i in range(850, 1150)],
        },
        "targets": ["target"],
    }
    (data_dir / "features.json").write_text(json.dumps(payload), encoding="utf-8")
    catalog = load_feature_catalog(data_dir)
    fragment = sample_feature_routing(random.Random(7), catalog, 2, fast=True)
    routing = resolve_feature_routing({**_base_flat(2), **fragment}, catalog)
    assert len(routing.feature_columns) <= MAX_ROUTED_FEATURES
