from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphapulse.models.packboost_backend import packboost_cuda_available
from alphapulse.models.packboost_encoding import (
    bin_features_for_packboost,
    default_nfeatsets,
    encode_era_ids,
)
from alphapulse.models.packboost_model import PackboostModel


def test_bin_features_for_packboost_clips_integers() -> None:
    frame = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 4.0, 3.0],
            "b": [0, 1, 2, 3, 4],
        }
    )
    binned = bin_features_for_packboost(frame)
    assert binned.dtype == np.int8
    assert binned.max() <= 4
    assert binned.min() >= 0


def test_bin_features_for_packboost_quantizes_scaled_values() -> None:
    frame = pd.DataFrame({"a": [-2.5, -1.0, 0.0, 1.0, 2.5]})
    binned = bin_features_for_packboost(frame)
    assert binned.dtype == np.int8
    assert sorted(binned[:, 0].tolist()) == [0, 1, 2, 3, 4]


def test_default_nfeatsets_scales_with_feature_count() -> None:
    assert default_nfeatsets(20, requested=32) == 2
    assert default_nfeatsets(200, requested=32) == 25


def test_encode_era_ids_preserves_chronological_order() -> None:
    era = pd.Series(["era_0002", "era_0001", "era_0002", "era_0001"])
    encoded = encode_era_ids(era)
    assert encoded.tolist() == [1, 0, 1, 0]


def test_packboost_model_rejects_non_cuda_device() -> None:
    rng = np.random.default_rng(0)
    n = 40
    cols = [f"f_{i}" for i in range(6)]
    x = pd.DataFrame(rng.integers(0, 5, size=(n, len(cols))), columns=cols)
    x["era"] = np.repeat(["era_0001", "era_0002"], n // 2)
    y = pd.Series(rng.standard_normal(n))

    model = PackboostModel(device="cpu", n_rounds_base=2, n_rounds_boost=2)
    with pytest.raises(ValueError, match="only supports device='cuda'"):
        model.train(x, y)


@pytest.mark.skipif(not packboost_cuda_available(), reason="PackBoost CUDA unavailable")
def test_packboost_model_trains_on_cuda() -> None:
    rng = np.random.default_rng(1)
    n = 200
    cols = [f"f_{i}" for i in range(24)]
    x = pd.DataFrame(rng.integers(0, 5, size=(n, len(cols))), columns=cols)
    x["era"] = np.repeat([f"era_{i:04d}" for i in range(10)], n // 10)
    y = pd.Series(rng.standard_normal(n), dtype=np.float32)

    model = PackboostModel(
        device="cuda",
        n_rounds_base=5,
        n_rounds_boost=3,
        n_worst_eras=2,
        nfolds=4,
        max_depth=4,
        nfeatsets=2,
    )
    metrics = model.train(x, y)
    preds = model.predict(x)
    assert "n_boost_eras" in metrics
    assert len(preds) == n
    assert np.isfinite(preds).all()
