import json
import random
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

from alphapulse.evaluation.export_validation import smoke_test_predict_fn
from alphapulse.features.catalog import TargetCatalog
from alphapulse.hpo.export import (
    build_hpo_pipeline_from_flat,
    prepare_hpo_flat,
    resolve_hpo_build_context,
)
from alphapulse.hpo.target_strategy import sample_target_strategy
from alphapulse.pipeline.multi_target import MultiTargetPipeline
from alphapulse.pipeline.multihead import MultiHeadPipeline
from alphapulse.pipeline.pipeline import Pipeline


def _write_toy_dataset(data_dir: Path, *, n: int = 400) -> list[str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    eras = np.repeat([f"era_{i:04d}" for i in range(20)], n // 20)
    df = pd.DataFrame(
        {
            "era": eras,
            "target": rng.standard_normal(n),
            "target_alpha_20": rng.standard_normal(n),
            "f_a": rng.standard_normal(n),
            "f_b": rng.standard_normal(n),
            "f_c": rng.standard_normal(n),
        }
    )
    df.to_parquet(data_dir / "train.parquet", index=False)
    features = {
        "feature_sets": {
            "small": ["f_a", "f_b"],
            "medium": ["f_a", "f_b", "f_c"],
            "strength": ["f_a", "f_c"],
        },
        "targets": ["target", "target_alpha_20"],
    }
    (data_dir / "features.json").write_text(json.dumps(features), encoding="utf-8")
    return ["f_a", "f_b", "f_c"]


def _minimal_flat() -> dict:
    return {
        "num_models": 1,
        "model_1_type": "XGBoost",
        "model_2_type": "XGBoost",
        "model_3_type": "XGBoost",
        "scaler_type": "StandardScaler",
        "use_packboost": False,
        "ensemble_method": "single",
        "use_neutralization": False,
        "xgb_max_depth": 3,
        "xgb_learning_rate": 0.1,
        "xgb_n_rounds": 10,
        "xgb_early_stopping": 5,
        "n_subs": 3,
        "target_mode": "single",
        "primary_target": "target",
        "auxiliary_targets": [],
        "use_feature_routing": False,
    }


def test_hpo_primary_is_target() -> None:
    catalog = TargetCatalog(targets=["target", "target_alpha_20"])
    for seed in range(30):
        strategy = sample_target_strategy(random.Random(seed), catalog, fast=True)
        assert strategy.primary_target == "target"


def test_export_matches_worker_build_context(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_toy_dataset(data_dir)
    flat = prepare_hpo_flat(
        {
            **_minimal_flat(),
            "use_feature_routing": True,
            "active_groups": ["small", "strength"],
            "model_1_groups": ["small", "strength"],
            "model_1_lane": 0,
            "lane_0_steps": [],
        },
        data_dir,
    )
    ctx = resolve_hpo_build_context(flat)
    assert ctx.routing.build_path == "simple"
    assert set(ctx.feature_columns) == {"f_a", "f_b", "f_c"}


def test_export_routed_config(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_toy_dataset(data_dir)
    flat = {
        **_minimal_flat(),
        "use_feature_routing": True,
        "active_groups": ["small"],
        "model_1_groups": ["small"],
        "model_1_lane": 0,
        "lane_0_steps": [],
    }
    result = build_hpo_pipeline_from_flat(
        flat,
        data_dir,
        train_subsample=0.5,
        seed=42,
    )
    assert isinstance(result.pipeline, Pipeline | MultiHeadPipeline)
    pkl = tmp_path / "predict.pkl"
    with open(pkl, "wb") as f:
        cloudpickle.dump(result.pipeline.to_numerai_predict(), f)
    smoke_test_predict_fn(pkl, result.feature_columns)


def test_export_multi_blend_config(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_toy_dataset(data_dir)
    flat = {
        **_minimal_flat(),
        "target_mode": "multi_blend",
        "primary_target": "target",
        "auxiliary_targets": ["target_alpha_20"],
        "target_blend_method": "equal",
    }
    result = build_hpo_pipeline_from_flat(
        flat,
        data_dir,
        train_subsample=0.5,
        seed=7,
    )
    assert isinstance(result.pipeline, MultiTargetPipeline)
    pkl = tmp_path / "predict_multi.pkl"
    with open(pkl, "wb") as f:
        cloudpickle.dump(result.pipeline.to_numerai_predict(), f)
    smoke_test_predict_fn(pkl, result.feature_columns)


def test_multitarget_predict_reindex_missing_columns() -> None:
    from alphapulse.models.xgboost_model import XGBoostModel
    from alphapulse.preprocessors.scaling import StandardScalerPreprocessor

    rng = np.random.default_rng(1)
    n = 80
    X = pd.DataFrame(
        {
            "f_a": rng.standard_normal(n),
            "f_b": rng.standard_normal(n),
        }
    )
    targets = pd.DataFrame(
        {
            "target": rng.standard_normal(n),
            "target_alpha_20": rng.standard_normal(n),
        }
    )

    def factory() -> XGBoostModel:
        return XGBoostModel(
            params={
                "max_depth": 3,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "objective": "reg:squarederror",
            }
        )

    pipeline = MultiTargetPipeline(
        preprocessors=[StandardScalerPreprocessor()],
        model_factory=factory,
        target_columns=["target", "target_alpha_20"],
        primary_target="target",
    )
    pipeline.fit(X, targets, n_rounds=5)
    live = pd.DataFrame(
        {
            "f_a": rng.standard_normal(10),
            "extra_col": rng.standard_normal(10),
        }
    )
    predict_fn = pipeline.to_numerai_predict()
    out = predict_fn(live, pd.DataFrame())
    assert "prediction" in out.columns
    assert out["prediction"].between(0.0, 1.0).all()
