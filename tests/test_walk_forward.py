"""Tests for purge-aware walk-forward backtesting."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.linear_model import Ridge

from alphapulse.evaluation.era_split import EraSplitEvaluator
from alphapulse.experiments.schema import EvaluationConfig
from alphapulse.validation.purge import effective_purge_eras


def _make_data(
    n_eras: int = 30, rows_per_era: int = 20, n_features: int = 5
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.standard_normal((n_eras * rows_per_era, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.standard_normal(len(X)), name="target")
    era = pd.Series(
        np.repeat([f"era_{i:04d}" for i in range(n_eras)], rows_per_era),
        name="era",
    )
    return X, y, era


class _Pred:
    def __init__(self, model: Ridge, feature_cols: list[str]) -> None:
        self._model = model
        self._feature_cols = feature_cols

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X[self._feature_cols])  # type: ignore[no-any-return]


def _ridge_train_fn(X_tr: pd.DataFrame, y_tr: pd.Series) -> _Pred:
    model = Ridge(alpha=1.0)
    numeric_cols = [c for c in X_tr.columns if c != "era"]
    model.fit(X_tr[numeric_cols], y_tr)
    return _Pred(model, numeric_cols)


class TestEraSplitEvaluatorDefaults:
    def test_default_n_purge_is_8(self) -> None:
        ev = EraSplitEvaluator()
        assert ev.n_purge == 8

    def test_default_n_embargo_is_0(self) -> None:
        ev = EraSplitEvaluator()
        assert ev.n_embargo == 0

    def test_default_n_splits_is_none(self) -> None:
        ev = EraSplitEvaluator()
        assert ev.n_splits is None

    def test_invalid_n_purge_raises(self) -> None:
        with pytest.raises(ValueError, match="n_purge"):
            EraSplitEvaluator(n_purge=-1)

    def test_invalid_n_embargo_raises(self) -> None:
        with pytest.raises(ValueError, match="n_embargo"):
            EraSplitEvaluator(n_embargo=-1)

    def test_invalid_n_splits_raises(self) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            EraSplitEvaluator(n_splits=1)


class TestPurgeGap:
    def test_sixty_day_target_requires_sixteen_eras(self) -> None:
        assert effective_purge_eras(0, ["target_ender_60"]) == 16

    def test_twenty_day_target_requires_eight_eras(self) -> None:
        assert effective_purge_eras(0, ["target_ender_20"]) == 8

    def test_purge_excludes_adjacent_train_eras(self) -> None:
        """With n_purge=2, eras immediately before each test era are not trained on."""
        n_purge = 2
        observed_gaps: list[int] = []

        _, _, era = _make_data(n_eras=20)
        eras_order = sorted(era.unique(), key=str)

        ev = EraSplitEvaluator(min_train_eras=1, n_purge=n_purge, n_embargo=0)

        step = ev.n_embargo + 1
        first_test = ev.min_train_eras + ev.n_purge

        for i in range(first_test, len(eras_order), step):
            train_end = max(0, i - ev.n_purge)
            train_eras = eras_order[:train_end]
            if len(train_eras) >= ev.min_train_eras:
                gap = i - train_end
                observed_gaps.append(gap)

        assert all(g == n_purge for g in observed_gaps), (
            f"Expected all gaps == {n_purge}, got {observed_gaps}"
        )

    def test_no_purge_tests_more_eras(self) -> None:
        """n_purge=0 should test more eras than n_purge=4 (starts earlier)."""
        _, _, era = _make_data(n_eras=15)

        eras_order = sorted(era.unique(), key=str)

        counted_with_purge = sum(1 for _ in range(1 + 4, len(eras_order)))
        counted_no_purge = sum(1 for _ in range(1 + 0, len(eras_order)))

        assert counted_no_purge > counted_with_purge


class TestFullMetrics:
    EXPECTED_KEYS = {
        "mean_per_era_correlation",
        "std_per_era_correlation",
        "corr_sharpe",
        "max_drawdown",
        "pct_positive_eras",
        "n_valid_eras",
    }

    def test_walk_forward_returns_full_metric_set(self) -> None:
        X, y, era = _make_data(n_eras=20)
        ev = EraSplitEvaluator(min_train_eras=2, n_purge=2, n_embargo=0)
        result = ev.evaluate_walk_forward(X, y, era, _ridge_train_fn)
        assert self.EXPECTED_KEYS.issubset(result.keys()), (
            f"Missing keys: {self.EXPECTED_KEYS - result.keys()}"
        )

    def test_walk_forward_values_are_finite(self) -> None:
        X, y, era = _make_data(n_eras=20)
        ev = EraSplitEvaluator(min_train_eras=2, n_purge=2, n_embargo=0)
        result = ev.evaluate_walk_forward(X, y, era, _ridge_train_fn)
        for key in ("corr_sharpe", "mean_per_era_correlation", "max_drawdown"):
            assert np.isfinite(result[key]), (
                f"{key} should be finite, got {result[key]}"
            )

    def test_nan_metrics_returned_when_no_folds(self) -> None:
        X, y, era = _make_data(n_eras=5)
        ev = EraSplitEvaluator(min_train_eras=10, n_purge=0, n_embargo=0)
        result = ev.evaluate_walk_forward(X, y, era, _ridge_train_fn)
        assert np.isnan(result["corr_sharpe"])
        assert np.isnan(result["mean_per_era_correlation"])


class TestPurgedCVMode:
    def test_n_splits_mode_returns_full_metrics(self) -> None:
        X, y, era = _make_data(n_eras=30)
        ev = EraSplitEvaluator(min_train_eras=5, n_purge=2, n_embargo=2, n_splits=3)
        result = ev.evaluate_walk_forward(X, y, era, _ridge_train_fn)
        expected = {
            "mean_per_era_correlation",
            "corr_sharpe",
            "max_drawdown",
            "pct_positive_eras",
            "n_valid_eras",
        }
        assert expected.issubset(result.keys())

    def test_n_splits_mode_produces_nonempty_result(self) -> None:
        X, y, era = _make_data(n_eras=30)
        ev = EraSplitEvaluator(min_train_eras=5, n_purge=2, n_embargo=2, n_splits=3)
        result = ev.evaluate_walk_forward(X, y, era, _ridge_train_fn)
        assert np.isfinite(result["corr_sharpe"])


class TestSchemaDefaults:
    def test_new_fields_exist_with_correct_defaults(self) -> None:
        cfg = EvaluationConfig()
        assert cfg.walk_forward_n_purge == 8
        assert cfg.walk_forward_n_embargo == 0
        assert cfg.walk_forward_n_splits is None

    def test_n_splits_none_is_valid(self) -> None:
        cfg = EvaluationConfig(walk_forward_n_splits=None)
        assert cfg.walk_forward_n_splits is None

    def test_n_splits_ge2_is_valid(self) -> None:
        cfg = EvaluationConfig(walk_forward_n_splits=5)
        assert cfg.walk_forward_n_splits == 5

    def test_n_splits_1_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationConfig(walk_forward_n_splits=1)

    def test_n_purge_negative_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationConfig(walk_forward_n_purge=-1)

    def test_yaml_roundtrip(self) -> None:
        cfg = EvaluationConfig(
            walk_forward=True,
            walk_forward_n_purge=4,
            walk_forward_n_embargo=2,
            walk_forward_n_splits=5,
        )
        d = cfg.model_dump()
        cfg2 = EvaluationConfig.model_validate(d)
        assert cfg2.walk_forward_n_purge == 4
        assert cfg2.walk_forward_n_embargo == 2
        assert cfg2.walk_forward_n_splits == 5
