import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_worker_style_cleanup_releases_refs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n = 500
    df = pd.DataFrame(
        {
            "era": np.repeat(["era_001", "era_002"], n // 2),
            "target": np.random.randn(n),
            "f_a": np.random.randn(n),
            "f_b": np.random.randn(n),
        }
    )
    df.to_parquet(data_dir / "train.parquet", index=False)
    features = {"feature_sets": {"medium": ["f_a", "f_b"]}, "targets": ["target"]}
    (data_dir / "features.json").write_text(json.dumps(features), encoding="utf-8")

    from alphapulse.experiments.data import load_train_only_frame

    rss_before = _rss_mb()
    for seed in range(2):
        X_train, y_train, _ = load_train_only_frame(
            data_dir,
            train_subsample=0.5,
            target_col="target",
            seed=seed,
            feature_columns=["f_a", "f_b"],
            need_era=True,
        )
        assert len(X_train) > 0
        del X_train, y_train
        gc.collect()
    rss_after = _rss_mb()
    assert rss_after <= rss_before + 80


def _rss_mb() -> float:
    try:
        import psutil  # type: ignore[import-untyped]

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except ImportError:
        return 0.0
