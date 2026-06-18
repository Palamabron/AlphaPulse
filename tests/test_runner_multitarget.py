import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphapulse.experiments.runner import run_experiment
from alphapulse.experiments.schema import ExperimentV1


@pytest.fixture
def multitarget_dataset_dir(tmp_path: Path) -> Path:
    rng = np.random.RandomState(0)
    n = 240
    eras = np.repeat([f"era_{i:04d}" for i in range(6)], n // 6)
    df = pd.DataFrame(
        {
            "feature_a": rng.randn(n).astype(np.float32),
            "feature_b": rng.randn(n).astype(np.float32),
            "era": eras,
            "target": rng.randn(n).astype(np.float32),
            "target_aux": rng.randn(n).astype(np.float32),
            "id": [f"id_{i}" for i in range(n)],
        }
    )
    df.to_parquet(tmp_path / "train.parquet", index=False)
    (tmp_path / "validation.parquet").write_bytes(
        (tmp_path / "train.parquet").read_bytes()
    )
    (tmp_path / "features.json").write_text(
        json.dumps(
            {
                "feature_sets": {
                    "small": ["feature_a"],
                    "all": ["feature_a", "feature_b"],
                },
                "targets": ["target", "target_aux"],
            }
        )
    )
    return tmp_path


def test_run_experiment_multitarget_builds_pipeline(
    multitarget_dataset_dir: Path,
) -> None:
    exp = ExperimentV1.model_validate(
        {
            "version": "1",
            "data": {
                "data_dir": str(multitarget_dataset_dir),
                "train_subsample": 1.0,
                "target_col": "target",
                "auxiliary_targets": ["target_aux"],
                "target_blend_method": "equal",
                "seed": 42,
            },
            "features": {"columns": ["feature_a", "feature_b"], "groups": {}},
            "preprocessing": [],
            "models": [
                {
                    "type": "XGBoost",
                    "params": {
                        "params": {
                            "max_depth": 2,
                            "learning_rate": 0.1,
                            "tree_method": "hist",
                            "objective": "reg:squarederror",
                        }
                    },
                }
            ],
            "ensemble_method": "single",
            "train": {"n_rounds": 10, "early_stopping_rounds": 5},
        }
    )
    result = run_experiment(exp, artifact_dir=multitarget_dataset_dir / "artifacts")
    assert result.error is None
    assert "corr_sharpe" in result.metrics


def test_schema_rejects_primary_in_auxiliary_targets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="auxiliary_targets must not include"):
        ExperimentV1.model_validate(
            {
                "version": "1",
                "data": {
                    "data_dir": str(tmp_path),
                    "target_col": "target",
                    "auxiliary_targets": ["target", "target_aux"],
                },
                "models": [{"type": "XGBoost", "params": {}}],
            }
        )
