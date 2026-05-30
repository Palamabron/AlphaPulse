import numpy as np
import pandas as pd
import pytest

from alphapulse.preprocessors.compression import PCAPreprocessor
from alphapulse.preprocessors.noise import GaussianNoiseInjector


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


# ── EraStableFeatureSelector ──────────────────────────────────────────────────


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
    X = pd.DataFrame(rng.randn(n, n_feats), columns=[f"feat_{i}" for i in range(n_feats)])
    y = pd.Series(rng.randn(n))
    eras = pd.Series([f"e{i}" for i in range(4) for _ in range(20)])

    for frac in [0.25, 0.5, 0.75]:
        selector = EraStableFeatureSelector(keep_fraction=frac, n_estimators=5, min_eras=2)
        out = selector.fit_transform(X, y, eras=eras)
        expected = max(1, int(n_feats * frac))
        assert out.shape[1] == expected, f"keep_fraction={frac}: expected {expected} cols, got {out.shape[1]}"


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
