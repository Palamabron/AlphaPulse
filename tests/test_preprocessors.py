import numpy as np
import pandas as pd

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
