import importlib.util

import numpy as np
import pandas as pd
import pytest

from alphapulse.preprocessors.autoencoder import AutoencoderPreprocessor
from alphapulse.preprocessors.compression import (
    PCAPreprocessor,
    TruncatedSVDPreprocessor,
)
from alphapulse.preprocessors.noise import GaussianNoiseInjector

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_gaussian_noise_inference_safe_after_fit() -> None:
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(20, 3), columns=["a", "b", "c"])
    y = pd.Series(rng.randn(20))
    noise = GaussianNoiseInjector(sigma=0.5, seed=42)
    noise.fit(X, y)
    out = noise.transform(X.copy())
    pd.testing.assert_frame_equal(out, X)


def test_gaussian_noise_active_during_train() -> None:
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(20, 3), columns=["a", "b", "c"])
    y = pd.Series(rng.randn(20))
    noise = GaussianNoiseInjector(sigma=0.5, seed=42)
    noise.fit(X, y)
    noise.train()
    out = noise.transform(X.copy())
    assert not out.equals(X)


def test_gaussian_noise_numeric_only() -> None:
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {
            "a": rng.randn(10),
            "b": rng.randn(10),
            "era": ["e1"] * 5 + ["e2"] * 5,
        }
    )
    y = pd.Series(rng.randn(10))
    noise = GaussianNoiseInjector(sigma=0.5, seed=42)
    noise.fit(X, y)
    noise.train()
    out = noise.transform(X.copy())
    assert out["era"].tolist() == X["era"].tolist()
    assert not np.allclose(out[["a", "b"]].values, X[["a", "b"]].values)


def test_pca_with_era_column() -> None:
    rng = np.random.RandomState(1)
    X = pd.DataFrame(
        {
            "f1": rng.randn(15),
            "f2": rng.randn(15),
            "f3": rng.randn(15),
            "era": ["e1"] * 5 + ["e2"] * 5 + ["e3"] * 5,
        }
    )
    pca = PCAPreprocessor(n_components=2, random_state=0)
    pca.fit(X)
    out = pca.transform(X)
    assert out.shape == (15, 2)
    assert list(out.columns) == ["pca_0", "pca_1"]


def test_pca_deterministic_with_seed() -> None:
    rng = np.random.RandomState(3)
    X = pd.DataFrame(rng.randn(30, 6), columns=[f"f{i}" for i in range(6)])
    out_a = PCAPreprocessor(n_components=3, random_state=0).fit_transform(X)
    out_b = PCAPreprocessor(n_components=3, random_state=0).fit_transform(X)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_truncated_svd_shape_and_columns() -> None:
    rng = np.random.RandomState(2)
    X = pd.DataFrame(rng.randn(20, 8), columns=[f"f{i}" for i in range(8)])
    svd = TruncatedSVDPreprocessor(n_components=3, random_state=0)
    out = svd.fit_transform(X)
    assert out.shape == (20, 3)
    assert list(out.columns) == ["svd_0", "svd_1", "svd_2"]


def test_truncated_svd_deterministic_with_seed() -> None:
    rng = np.random.RandomState(4)
    X = pd.DataFrame(rng.randn(25, 5), columns=[f"f{i}" for i in range(5)])
    out_a = TruncatedSVDPreprocessor(n_components=2, random_state=1).fit_transform(X)
    out_b = TruncatedSVDPreprocessor(n_components=2, random_state=1).fit_transform(X)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_autoencoder_registered_in_registry() -> None:
    from alphapulse.hpo.builder import build_preprocessors

    steps = build_preprocessors([{"type": "Autoencoder", "params": {"latent_dim": 2}}])
    assert isinstance(steps[0], AutoencoderPreprocessor)
    assert steps[0].latent_dim == 2


def test_autoencoder_invalid_latent_dim_raises() -> None:
    with pytest.raises(ValueError, match="latent_dim"):
        AutoencoderPreprocessor(latent_dim=0)


@pytest.mark.skipif(HAS_TORCH, reason="torch installed — error path unreachable")
def test_autoencoder_requires_torch() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    with pytest.raises(ImportError, match=r"alphapulse\[deep\]"):
        AutoencoderPreprocessor(latent_dim=1).fit(X)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed — skip")
def test_autoencoder_fit_transform_shape() -> None:
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(40, 8), columns=[f"f{i}" for i in range(8)])
    ae = AutoencoderPreprocessor(latent_dim=3, epochs=2, batch_size=16, seed=0)
    out = ae.fit_transform(X)
    assert out.shape == (40, 3)
    assert list(out.columns) == ["ae_0", "ae_1", "ae_2"]
    assert np.isfinite(out.values).all()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed — skip")
def test_autoencoder_deterministic_with_seed() -> None:
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.randn(30, 6), columns=[f"f{i}" for i in range(6)])
    out_a = AutoencoderPreprocessor(latent_dim=2, epochs=2, seed=5).fit_transform(X)
    out_b = AutoencoderPreprocessor(latent_dim=2, epochs=2, seed=5).fit_transform(X)
    pd.testing.assert_frame_equal(out_a, out_b)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed — skip")
def test_autoencoder_nan_safe() -> None:
    rng = np.random.RandomState(2)
    X = pd.DataFrame(rng.randn(30, 6), columns=[f"f{i}" for i in range(6)])
    X.iloc[0, 0] = np.nan
    X.iloc[5, 3] = np.nan
    ae = AutoencoderPreprocessor(latent_dim=2, epochs=2, seed=0)
    out = ae.fit_transform(X)
    assert np.isfinite(out.values).all()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed — skip")
def test_autoencoder_ignores_non_numeric_columns() -> None:
    rng = np.random.RandomState(3)
    X = pd.DataFrame(rng.randn(20, 5), columns=[f"f{i}" for i in range(5)])
    X["era"] = ["e1"] * 10 + ["e2"] * 10
    ae = AutoencoderPreprocessor(latent_dim=2, epochs=2, seed=0)
    out = ae.fit_transform(X)
    assert out.shape == (20, 2)


def test_era_stable_selector_reduces_features() -> None:
    pytest.importorskip("lightgbm")
    from alphapulse.preprocessors.era_stable import EraStableFeatureSelector

    rng = np.random.RandomState(42)
    n = 120
    X = pd.DataFrame(rng.randn(n, 10), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(rng.randn(n))
    eras = pd.Series([f"e{i:02d}" for i in range(12) for _ in range(10)])

    selector = EraStableFeatureSelector(keep_fraction=0.5, n_estimators=10, min_eras=2)
    out = selector.fit_transform(X, y, eras=eras)
    assert out.shape[0] == n
    assert out.shape[1] == 5


def test_era_stable_selector_keep_fraction_respected() -> None:
    pytest.importorskip("lightgbm")
    from alphapulse.preprocessors.era_stable import EraStableFeatureSelector

    rng = np.random.RandomState(0)
    n = 80
    n_feats = 8
    X = pd.DataFrame(
        rng.randn(n, n_feats), columns=[f"feat_{i}" for i in range(n_feats)]
    )
    y = pd.Series(rng.randn(n))
    eras = pd.Series([f"e{i}" for i in range(4) for _ in range(20)])

    for frac in [0.25, 0.5, 0.75]:
        selector = EraStableFeatureSelector(
            keep_fraction=frac, n_estimators=5, min_eras=2
        )
        out = selector.fit_transform(X, y, eras=eras)
        expected = max(1, int(n_feats * frac))
        assert out.shape[1] == expected, (
            f"keep_fraction={frac}: expected {expected} cols, got {out.shape[1]}"
        )


def test_era_stable_selector_transform_only_uses_fitted_cols() -> None:
    pytest.importorskip("lightgbm")
    from alphapulse.preprocessors.era_stable import EraStableFeatureSelector

    rng = np.random.RandomState(1)
    n = 60
    X = pd.DataFrame(rng.randn(n, 6), columns=list("abcdef"))
    y = pd.Series(rng.randn(n))
    eras = pd.Series([f"e{i}" for i in range(3) for _ in range(20)])

    selector = EraStableFeatureSelector(keep_fraction=0.5, n_estimators=5, min_eras=2)
    selector.fit(X, y, eras=eras)
    fitted_cols = set(selector.selected_columns_)

    out = selector.transform(X)
    assert set(out.columns) == fitted_cols


def test_era_stable_selector_fallback_no_eras() -> None:
    pytest.importorskip("lightgbm")
    from alphapulse.preprocessors.era_stable import EraStableFeatureSelector

    rng = np.random.RandomState(7)
    n = 40
    X = pd.DataFrame(rng.randn(n, 6), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(rng.randn(n))

    selector = EraStableFeatureSelector(keep_fraction=0.5, n_estimators=5)
    out = selector.fit_transform(X, y, eras=None)
    assert out.shape == (n, 3)


def test_era_stable_selector_requires_y() -> None:
    pytest.importorskip("lightgbm")
    from alphapulse.preprocessors.era_stable import EraStableFeatureSelector

    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    selector = EraStableFeatureSelector()
    with pytest.raises(ValueError, match="requires y"):
        selector.fit(X, y=None)
