from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphapulse.experiments.data import load_mmc_validation_frame
from alphapulse.hpo.objective import _merge_validation_mmc_metrics
from alphapulse.models.xgboost_model import XGBoostModel
from alphapulse.pipeline.pipeline import Pipeline
from alphapulse.preprocessors.scaling import StandardScalerPreprocessor


def _write_mmc_dataset(data_dir: Path) -> list[str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 400
    eras = np.repeat([f"{i:04d}" for i in range(1133, 1143)], n // 10)
    ids = [f"n{i:016x}" for i in range(n)]
    val_df = pd.DataFrame(
        {
            "era": eras,
            "target": rng.standard_normal(n),
            "f_a": rng.standard_normal(n),
            "f_b": rng.standard_normal(n),
        },
        index=ids,
    )
    val_df.index.name = "id"
    val_df.to_parquet(data_dir / "validation.parquet")

    meta_df = pd.DataFrame(
        {
            "era": eras,
            "data_type": "validation",
            "numerai_meta_model": rng.uniform(0.0, 1.0, n),
        },
        index=ids,
    )
    meta_df.index.name = "id"
    meta_df.to_parquet(data_dir / "meta_model.parquet")
    return ["f_a", "f_b"]


def test_load_mmc_validation_frame_returns_aligned_meta(tmp_path: Path) -> None:
    feature_cols = _write_mmc_dataset(tmp_path)
    frame = load_mmc_validation_frame(
        tmp_path,
        feature_cols=feature_cols,
        target_col="target",
        train_subsample=0.5,
        seed=1,
    )
    assert frame is not None
    X_val, y_val, era_val, meta_preds = frame
    assert len(X_val) == len(y_val) == len(era_val) == len(meta_preds)
    assert np.isfinite(meta_preds).all()


def test_merge_validation_mmc_metrics_populates_mmc(tmp_path: Path) -> None:
    feature_cols = _write_mmc_dataset(tmp_path)
    frame = load_mmc_validation_frame(
        tmp_path,
        feature_cols=feature_cols,
        target_col="target",
        train_subsample=1.0,
        seed=1,
    )
    assert frame is not None
    X_val, y_val, _, _ = frame
    pipe = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        model=XGBoostModel(
            params={
                "max_depth": 3,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "objective": "reg:squarederror",
            }
        ),
    )
    pipe.fit(X_val.iloc[:300], y_val.iloc[:300], n_rounds=8)
    merged = _merge_validation_mmc_metrics(
        {"corr_sharpe": 1.0},
        pipeline=pipe,
        data_dir=tmp_path,
        feature_cols=feature_cols,
        target_col="target",
        train_subsample=1.0,
        seed=1,
    )[0]
    assert np.isfinite(merged["mmc"])
    assert np.isfinite(merged["mmc_sharpe"])
    assert np.isfinite(merged["payout_score"])
    assert merged["holdout_corr_sharpe"] == 1.0
    assert np.isfinite(merged["val_corr_sharpe"])
