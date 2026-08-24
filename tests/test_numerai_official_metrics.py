"""Golden parity tests generated with numerai-tools==0.6.0 scoring.py.

Reference package: https://pypi.org/project/numerai-tools/0.6.0/
"""

import numpy as np
import pandas as pd
import pytest

from alphapulse.evaluation.metrics import (
    _score_series_summary,
    numerai_official_diagnostics,
    per_era_numerai_corr,
    per_era_numerai_mmc,
    per_era_weighted_corr_mmc,
    weighted_corr_mmc_sharpe,
)

TARGET = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.25, 0.75, 0.5])
PREDICTION = np.array([0.1, 0.8, 0.4, 0.7, 0.9, 0.2, 0.6, 0.3])
META_MODEL = np.array([0.3, 0.2, 0.8, 0.1, 0.9, 0.4, 0.7, 0.6])
TIED_PREDICTION = np.array([0.2, 0.2, 0.8, 0.8, 0.5, 0.5, 0.2, 0.8])
TIED_META_MODEL = np.array([0.1, 0.7, 0.7, 0.4, 0.9, 0.4, 0.1, 0.9])
ERA = pd.Series(["era_0001"] * len(TARGET))


def test_per_era_numerai_corr_matches_numerai_tools_0_6_0_normal_golden() -> None:
    result = per_era_numerai_corr(pd.Series(TARGET), PREDICTION, ERA)

    assert result["era_0001"] == pytest.approx(0.8378892299062128, abs=1e-12)


def test_per_era_numerai_mmc_matches_numerai_tools_0_6_0_normal_golden() -> None:
    result = per_era_numerai_mmc(pd.Series(TARGET), PREDICTION, META_MODEL, ERA)

    assert result["era_0001"] == pytest.approx(0.7314882694511073, abs=1e-12)


def test_per_era_numerai_corr_matches_numerai_tools_0_6_0_ties_golden() -> None:
    result = per_era_numerai_corr(pd.Series(TARGET), TIED_PREDICTION, ERA)

    assert result["era_0001"] == pytest.approx(0.3494859826646888, abs=1e-12)


def test_per_era_numerai_mmc_matches_numerai_tools_0_6_0_ties_golden() -> None:
    result = per_era_numerai_mmc(
        pd.Series(TARGET), TIED_PREDICTION, TIED_META_MODEL, ERA
    )

    assert result["era_0001"] == pytest.approx(0.14176596593392451, abs=1e-12)


def test_per_era_numerai_corr_matches_numerai_tools_0_6_0_constant_prediction() -> None:
    result = per_era_numerai_corr(pd.Series(TARGET), np.full(len(TARGET), 0.5), ERA)

    assert np.isnan(result["era_0001"])


def test_per_era_numerai_mmc_matches_numerai_tools_0_6_0_constant_prediction() -> None:
    result = per_era_numerai_mmc(
        pd.Series(TARGET), np.full(len(TARGET), 0.5), META_MODEL, ERA
    )

    assert result["era_0001"] == pytest.approx(0.0, abs=1e-12)


def test_per_era_numerai_mmc_matches_numerai_tools_0_6_0_constant_meta() -> None:
    result = per_era_numerai_mmc(
        pd.Series(TARGET), PREDICTION, np.full(len(TARGET), 0.5), ERA
    )

    assert np.isnan(result["era_0001"])


def test_per_era_numerai_corr_matches_numerai_tools_0_6_0_constant_target() -> None:
    result = per_era_numerai_corr(pd.Series(np.full(len(TARGET), 0.5)), PREDICTION, ERA)

    assert np.isnan(result["era_0001"])


def test_per_era_numerai_mmc_matches_numerai_tools_0_6_0_constant_target() -> None:
    result = per_era_numerai_mmc(
        pd.Series(np.full(len(TARGET), 0.5)), PREDICTION, META_MODEL, ERA
    )

    assert result["era_0001"] == pytest.approx(0.0, abs=1e-12)


def test_per_era_numerai_corr_matches_numerai_tools_0_6_0_nan_golden() -> None:
    prediction = PREDICTION.copy()
    prediction[0] = np.nan

    result = per_era_numerai_corr(pd.Series(TARGET), prediction, ERA)

    assert result["era_0001"] == pytest.approx(0.7045415961898207, abs=1e-12)


def test_per_era_numerai_mmc_matches_numerai_tools_0_6_0_nan_golden() -> None:
    prediction = PREDICTION.copy()
    prediction[0] = np.nan

    result = per_era_numerai_mmc(pd.Series(TARGET), prediction, META_MODEL, ERA)

    assert result["era_0001"] == pytest.approx(0.4768198319569445, abs=1e-12)


def test_official_metrics_reject_missing_rows_above_numerai_tools_filter_limit() -> (
    None
):
    prediction = PREDICTION.copy()
    prediction[:2] = np.nan

    with pytest.raises(ValueError, match="at least 80%"):
        per_era_numerai_corr(pd.Series(TARGET), prediction, ERA)


def test_weighted_corr_mmc_matches_frozen_components_and_weights() -> None:
    target = np.concatenate(
        [TARGET, np.array([1.0, 0.75, 0.5, 0.25, 0.0, 0.75, 0.25, 0.5])]
    )
    prediction = np.concatenate(
        [PREDICTION, np.array([0.6, 0.2, 0.9, 0.1, 0.4, 0.8, 0.3, 0.7])]
    )
    meta = np.concatenate(
        [META_MODEL, np.array([0.8, 0.1, 0.6, 0.2, 0.5, 0.9, 0.4, 0.3])]
    )
    eras = pd.Series(["era_0001"] * 8 + ["era_0002"] * 8)

    result = per_era_weighted_corr_mmc(pd.Series(target), prediction, meta, eras)

    expected = pd.Series(
        [2.2742655286946514, 0.39707884000495164],
        index=["era_0001", "era_0002"],
    )
    pd.testing.assert_series_equal(result, expected)


def test_weighted_corr_mmc_sharpe_matches_population_sharpe() -> None:
    target = np.concatenate(
        [TARGET, np.array([1.0, 0.75, 0.5, 0.25, 0.0, 0.75, 0.25, 0.5])]
    )
    prediction = np.concatenate(
        [PREDICTION, np.array([0.6, 0.2, 0.9, 0.1, 0.4, 0.8, 0.3, 0.7])]
    )
    meta = np.concatenate(
        [META_MODEL, np.array([0.8, 0.1, 0.6, 0.2, 0.5, 0.9, 0.4, 0.3])]
    )
    eras = pd.Series(["era_0001"] * 8 + ["era_0002"] * 8)

    result = weighted_corr_mmc_sharpe(pd.Series(target), prediction, meta, eras)

    assert result == pytest.approx(1.4230573787864622, abs=1e-12)


def test_indexed_inputs_are_aligned_by_row_id() -> None:
    index = pd.Index([f"row_{i}" for i in range(len(TARGET))])
    target = pd.Series(TARGET, index=index)
    prediction = pd.Series(PREDICTION, index=index).sample(frac=1.0, random_state=1)
    meta = pd.Series(META_MODEL, index=index).sample(frac=1.0, random_state=2)
    eras = pd.Series(["era_0001"] * len(index), index=index).sample(
        frac=1.0, random_state=3
    )

    corr = per_era_numerai_corr(target, prediction, eras)
    mmc = per_era_numerai_mmc(target, prediction, meta, eras)

    assert corr["era_0001"] == pytest.approx(0.8378892299062128, abs=1e-12)
    assert mmc["era_0001"] == pytest.approx(0.7314882694511073, abs=1e-12)


def test_official_metrics_reject_mismatched_or_duplicate_row_ids() -> None:
    index = pd.Index([f"row_{i}" for i in range(len(TARGET))])
    target = pd.Series(TARGET, index=index)
    eras = pd.Series(["era_0001"] * len(index), index=index)
    mismatched = pd.Series(PREDICTION, index=[*index[:-1], "other"])
    duplicated = pd.Series(PREDICTION, index=[*index[:-1], index[-2]])

    with pytest.raises(ValueError, match="row IDs must exactly match"):
        per_era_numerai_corr(target, mismatched, eras)
    with pytest.raises(ValueError, match="duplicate row IDs"):
        per_era_numerai_corr(target, duplicated, eras)


def test_official_metrics_reject_infinite_values() -> None:
    prediction = PREDICTION.copy()
    prediction[0] = np.inf

    with pytest.raises(ValueError, match="infinite"):
        per_era_numerai_corr(pd.Series(TARGET), prediction, ERA)


def test_weighted_diagnostic_propagates_invalid_components() -> None:
    constant_prediction = np.full(len(TARGET), 0.5)

    scores = per_era_weighted_corr_mmc(
        pd.Series(TARGET), constant_prediction, META_MODEL, ERA
    )
    sharpe = weighted_corr_mmc_sharpe(
        pd.Series(TARGET), constant_prediction, META_MODEL, ERA
    )

    assert np.isnan(scores["era_0001"])
    assert np.isnan(sharpe)


def test_official_diagnostics_expose_unambiguous_component_names() -> None:
    target = pd.Series(np.concatenate([TARGET, TARGET[::-1]]))
    prediction = np.concatenate([PREDICTION, PREDICTION[::-1]])
    meta = np.concatenate([META_MODEL, META_MODEL[::-1]])
    eras = pd.Series(["era_0001"] * 8 + ["era_0002"] * 8)

    result = numerai_official_diagnostics(
        target,
        prediction,
        eras,
        meta_model=meta,
    )

    assert {
        "numerai_corr_mean",
        "numerai_corr_sharpe",
        "numerai_mmc_mean",
        "numerai_mmc_sharpe",
        "weighted_corr_mmc_mean",
        "weighted_corr_mmc_sharpe",
    } <= result.keys()


@pytest.mark.parametrize(
    ("score", "expected_sign"),
    [(0.25, 1), (-0.25, -1), (0.0, 0)],
)
def test_official_summary_zero_variance_uses_population_sharpe_semantics(
    score: float,
    expected_sign: int,
) -> None:
    result = _score_series_summary(
        pd.Series([score, score]),
        prefix="score",
    )

    sharpe = result["score_sharpe"]
    if expected_sign == 0:
        assert np.isnan(sharpe)
    else:
        assert np.isinf(sharpe)
        assert np.sign(sharpe) == expected_sign
