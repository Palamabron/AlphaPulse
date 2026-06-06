import numpy as np
import pandas as pd
import pytest

from alphapulse.evaluation.metrics import (
    era_sharpe_of_mmc,
    mmc_score,
    payout_score,
    per_era_mmc,
)

N_ERAS = 4
ROWS_PER_ERA = 50
N = N_ERAS * ROWS_PER_ERA
ERA_LABELS = [f"era_{i:04d}" for i in range(N_ERAS)]


def _eras() -> pd.Series:
    return pd.Series(np.repeat(ERA_LABELS, ROWS_PER_ERA))


def _target(seed: int = 0) -> pd.Series:
    return pd.Series(np.random.RandomState(seed).randn(N))


def test_mmc_score_returns_float() -> None:
    rng = np.random.RandomState(1)
    y_true = _target()
    y_pred = rng.randn(N)
    meta = rng.randn(N)
    result = mmc_score(y_true, y_pred, meta, _eras())
    assert isinstance(result, float)


def test_mmc_score_identical_to_meta_model_is_nan() -> None:
    """When predictions = meta model exactly, the residual after neutralization is
    the zero vector — Pearson correlation is undefined, so mmc_score returns NaN."""
    rng = np.random.RandomState(2)
    y_true = _target()
    meta = rng.randn(N)
    result = mmc_score(y_true, meta, meta, _eras())
    assert np.isnan(result)


def test_mmc_score_positive_for_informative_predictor() -> None:
    """Predictor correlated with target and orthogonal to meta has positive MMC."""
    rng = np.random.RandomState(3)
    y_arr = rng.randn(N)
    y_true = pd.Series(y_arr)
    noise = rng.randn(N) * 0.3
    y_pred = y_arr + noise
    meta = rng.randn(N)
    result = mmc_score(y_true, y_pred, meta, _eras())
    assert result > 0.0


def test_mmc_score_anti_correlated_predictor_is_negative() -> None:
    """A predictor perfectly anti-correlated with the target yields negative MMC."""
    rng = np.random.RandomState(4)
    y_arr = rng.randn(N)
    y_true = pd.Series(y_arr)
    y_pred = -y_arr
    meta = rng.randn(N)
    result = mmc_score(y_true, y_pred, meta, _eras())
    assert result < 0.0


def test_per_era_mmc_length_matches_n_eras() -> None:
    rng = np.random.RandomState(5)
    y_true = _target()
    y_pred = rng.randn(N)
    meta = rng.randn(N)
    per_era = per_era_mmc(y_true, y_pred, meta, _eras())
    assert len(per_era) == N_ERAS
    assert list(per_era.index) == sorted(ERA_LABELS)


def test_per_era_mmc_nan_for_single_row_era() -> None:
    """An era with only one row should produce NaN (can't compute correlation).
    e2 has 3 rows with predictor ranks non-collinear with meta ranks, so a
    finite MMC score is possible after neutralization.
    """
    # ranks for e2 pred: [3,1,2] (centered [1,-1,0]); meta: [1,2,3] (centered [-1,0,1])
    # These are not collinear so neutralization leaves a non-zero residual.
    y_true = pd.Series([0.5, 0.1, 0.9, 0.4])
    y_pred = np.array([0.4, 0.9, 0.3, 0.6])
    meta = np.array([0.3, 0.1, 0.5, 0.9])
    eras = pd.Series(["e1", "e2", "e2", "e2"])
    per_era = per_era_mmc(y_true, y_pred, meta, eras)
    assert np.isnan(per_era["e1"])
    assert np.isfinite(per_era["e2"])


def test_per_era_mmc_length_mismatch_raises() -> None:
    y_true = pd.Series([0.1, 0.2, 0.3])
    y_pred = np.array([0.1, 0.2])
    meta = np.array([0.5, 0.5, 0.5])
    eras = pd.Series(["e1", "e1", "e1"])
    with pytest.raises(ValueError, match="same length"):
        per_era_mmc(y_true, y_pred, meta, eras)


def test_mmc_score_length_mismatch_raises() -> None:
    y_true = pd.Series([0.1, 0.2, 0.3])
    y_pred = np.array([0.1, 0.2, 0.3])
    meta = np.array([0.5, 0.5])
    eras = pd.Series(["e1", "e1", "e1"])
    with pytest.raises(ValueError, match="same length"):
        mmc_score(y_true, y_pred, meta, eras)


def test_era_sharpe_of_mmc_constant_mmc_is_negative_infinity() -> None:
    """If per-era MMC is constant (zero std), sharpe should be -inf."""
    y_true = pd.Series(np.zeros(N))
    meta = np.ones(N)
    y_pred = meta.copy()
    result = era_sharpe_of_mmc(y_true, y_pred, meta, _eras())
    assert result == float("-inf")


def test_era_sharpe_of_mmc_returns_finite_for_variable_predictor() -> None:
    rng = np.random.RandomState(6)
    y_arr = rng.randn(N)
    y_true = pd.Series(y_arr)
    y_pred = y_arr + rng.randn(N) * 0.5
    meta = rng.randn(N)
    result = era_sharpe_of_mmc(y_true, y_pred, meta, _eras())
    assert np.isfinite(result)


def test_payout_score_formula() -> None:
    """payout = 0.75 * corr_sharpe + 2.25 * mmc_sharpe."""
    from alphapulse.evaluation.metrics import era_sharpe

    rng = np.random.RandomState(7)
    y_arr = rng.randn(N)
    y_true = pd.Series(y_arr)
    y_pred = y_arr + rng.randn(N) * 0.5
    meta = rng.randn(N)
    eras = _eras()

    cs = era_sharpe(y_true, y_pred, eras)
    ms = era_sharpe_of_mmc(y_true, y_pred, meta, eras)
    expected = 0.75 * (cs if np.isfinite(cs) else 0.0) + 2.25 * (
        ms if np.isfinite(ms) else 0.0
    )

    result = payout_score(y_true, y_pred, meta, eras)
    assert abs(result - expected) < 1e-10


def test_payout_score_custom_weights() -> None:
    rng = np.random.RandomState(8)
    y_arr = rng.randn(N)
    y_true = pd.Series(y_arr)
    y_pred = y_arr + rng.randn(N) * 0.3
    meta = rng.randn(N)
    eras = _eras()

    from alphapulse.evaluation.metrics import era_sharpe

    cs = era_sharpe(y_true, y_pred, eras)
    ms = era_sharpe_of_mmc(y_true, y_pred, meta, eras)
    expected = 1.0 * (cs if np.isfinite(cs) else 0.0) + 1.0 * (
        ms if np.isfinite(ms) else 0.0
    )

    result = payout_score(y_true, y_pred, meta, eras, corr_weight=1.0, mmc_weight=1.0)
    assert abs(result - expected) < 1e-10


def test_mmc_score_empty_eras_returns_nan() -> None:
    y_true = pd.Series([], dtype=float)
    y_pred = np.array([])
    meta = np.array([])
    eras = pd.Series([], dtype=str)
    result = mmc_score(y_true, y_pred, meta, eras)
    assert np.isnan(result)
