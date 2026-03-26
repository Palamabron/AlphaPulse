import json
from pathlib import Path

import numpy as np
import pandas as pd

from alphapulse.data import NumeraiDataLoader


def test_numerai_data_loader_loads_train_split(tmp_path: Path) -> None:
    rng = np.random.RandomState(0)
    n = 120
    eras = np.repeat(["era_0001", "era_0002", "era_0003"], n // 3)
    df = pd.DataFrame(
        {
            "feature_a": rng.randn(n).astype(np.float32),
            "feature_b": rng.randn(n).astype(np.float32),
            "era": eras,
            "target": rng.randn(n).astype(np.float32),
            "id": [f"id_{i}" for i in range(n)],
        }
    )
    df.to_parquet(tmp_path / "train.parquet", index=False)

    (tmp_path / "validation.parquet").write_bytes(
        (tmp_path / "train.parquet").read_bytes()
    )

    features_json = {
        "feature_sets": {"small": ["feature_a"], "all": ["feature_a", "feature_b"]}
    }
    (tmp_path / "features.json").write_text(json.dumps(features_json))

    loader = NumeraiDataLoader(tmp_path, feature_set="all")
    ds = loader.load_split("train")

    assert ds.n_rows == n
    assert ds.n_features == 2
    assert ds.feature_columns == ["feature_a", "feature_b"]
