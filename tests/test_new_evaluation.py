import numpy as np
import pandas as pd
import pytest

from alphapulse.evaluation.ensemble_diagnostics import (
    compute_ensemble_diagnostics,
    correlation_matrix,
    effective_model_count,
)
from alphapulse.evaluation.submission import prepare_submission, validate_submission

N_ROWS = 120
N_ERAS = 6
ERA_SIZE = N_ROWS // N_ERAS


def _make_eras() -> pd.Series:
    return pd.Series([f"era{i:02d}" for i in range(N_ERAS) for _ in range(ERA_SIZE)])


def _make_preds(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(N_ROWS)


def test_correlation_matrix_shape() -> None:
    eras = _make_eras()
    oof = {"model_a": _make_preds(0), "model_b": _make_preds(1)}
    corr = correlation_matrix(oof, eras)
    assert corr.shape == (2, 2)
    assert list(corr.index) == ["model_a", "model_b"]
    assert list(corr.columns) == ["model_a", "model_b"]


def test_correlation_matrix_diagonal_is_one() -> None:
    eras = _make_eras()
    oof = {"m1": _make_preds(0), "m2": _make_preds(1)}
    corr = correlation_matrix(oof, eras)
    np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-6)


def test_effective_model_count_range() -> None:
    eras = _make_eras()
    oof = {"m1": _make_preds(0), "m2": _make_preds(99)}
    corr = correlation_matrix(oof, eras)
    w = np.array([0.5, 0.5])
    eff = effective_model_count(w, corr)
    # N_eff >= 1 always; can exceed n_models when predictions are negatively correlated
    assert eff >= 1.0


def test_effective_model_count_identical_models_is_one() -> None:
    eras = _make_eras()
    preds = _make_preds(0)
    oof = {"m1": preds, "m2": preds.copy()}
    corr = correlation_matrix(oof, eras)
    w = np.array([0.5, 0.5])
    eff = effective_model_count(w, corr)
    assert abs(eff - 1.0) < 0.05


def test_compute_ensemble_diagnostics_keys() -> None:
    eras = _make_eras()
    oof = {"alpha": _make_preds(7), "beta": _make_preds(13)}
    y = np.random.RandomState(42).randn(N_ROWS)
    result = compute_ensemble_diagnostics(oof, y, eras)
    assert set(result.keys()) == {
        "correlation_matrix",
        "effective_model_count",
        "mean_pairwise_correlation",
        "model_names",
    }
    assert result["model_names"] == ["alpha", "beta"]
    assert isinstance(result["effective_model_count"], float)
    assert isinstance(result["mean_pairwise_correlation"], float)


def test_compute_ensemble_diagnostics_with_weights() -> None:
    eras = _make_eras()
    oof = {"m1": _make_preds(1), "m2": _make_preds(2)}
    y = np.zeros(N_ROWS)
    result = compute_ensemble_diagnostics(oof, y, eras, weights=np.array([0.7, 0.3]))
    assert 1.0 <= result["effective_model_count"] <= 2.0 + 1e-6


def test_validate_submission_clean() -> None:
    df = pd.DataFrame({"prediction": np.linspace(0.0, 1.0, 50)})
    issues = validate_submission(df)
    assert issues == []


def test_validate_submission_missing_column() -> None:
    df = pd.DataFrame({"score": [0.5, 0.4]})
    issues = validate_submission(df)
    assert any("Missing" in msg for msg in issues)


def test_validate_submission_nan_predictions() -> None:
    df = pd.DataFrame({"prediction": [0.5, np.nan, 0.3]})
    issues = validate_submission(df)
    assert any("NaN" in msg for msg in issues)


def test_validate_submission_out_of_range() -> None:
    df = pd.DataFrame({"prediction": [0.5, 1.5, -0.1]})
    issues = validate_submission(df)
    assert any("ERROR" in msg and "[0, 1]" in msg for msg in issues)


def test_validate_submission_constant_predictions() -> None:
    df = pd.DataFrame({"prediction": [0.5] * 20})
    issues = validate_submission(df)
    assert any(
        "constant" in msg.lower() or "identical" in msg.lower() for msg in issues
    )


def test_validate_submission_id_alignment_missing() -> None:
    preds = pd.DataFrame({"id": ["a", "b"], "prediction": [0.5, 0.6]})
    live = pd.DataFrame({"id": ["a", "b", "c"]})
    issues = validate_submission(preds, live)
    assert any("missing" in msg.lower() for msg in issues)


def test_validate_submission_rejects_duplicate_ids() -> None:
    live = pd.DataFrame({"id": ["a", "b"], "feature": [0.0, 1.0]})
    predictions = pd.DataFrame({"id": ["a", "a"], "prediction": [0.2, 0.8]})

    issues = validate_submission(predictions, live)

    assert "ERROR: Duplicate prediction IDs found." in issues


def test_prepare_submission_shape_and_range() -> None:
    preds = np.random.randn(100)
    ids = [f"row_{i}" for i in range(100)]
    df = prepare_submission(preds, ids)
    assert list(df.columns) == ["id", "prediction"]
    assert len(df) == 100
    assert df["prediction"].min() >= 0.0
    assert df["prediction"].max() <= 1.0


def test_prepare_submission_rank_normalized() -> None:
    preds = np.array([3.0, 1.0, 2.0])
    ids = ["a", "b", "c"]
    df = prepare_submission(preds, ids)
    sorted_preds = df.set_index("id").loc[["b", "c", "a"], "prediction"].values
    assert sorted_preds[0] < sorted_preds[1] < sorted_preds[2]


def test_compute_feature_report_basic() -> None:
    pytest.importorskip("lightgbm")
    from alphapulse.evaluation.feature_report import compute_feature_report

    rng = np.random.RandomState(0)
    n = 60
    X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.randn(n))
    eras = pd.Series([f"e{i:02d}" for i in range(3) for _ in range(20)])

    result = compute_feature_report(X, y, eras, n_estimators=10, top_n=3)
    assert result["n_eras_used"] > 0
    assert result["n_features"] == 5
    assert len(result["top_by_mean"]) > 0
    assert "feature" in result["top_by_mean"][0]
    assert "mean_importance" in result["top_by_mean"][0]


def test_compute_feature_report_empty_eras() -> None:
    pytest.importorskip("lightgbm")
    from alphapulse.evaluation.feature_report import compute_feature_report

    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.randn(10, 3), columns=["a", "b", "c"])
    y = pd.Series(rng.randn(10))
    eras = pd.Series(["only_one"] * 10)

    result = compute_feature_report(
        X, y, eras, n_estimators=5, top_n=2, max_era_subsample=1
    )
    assert result["n_features"] == 3
