import random

import numpy as np
import pandas as pd
import pytest

from alphapulse.features.catalog import TargetCatalog
from alphapulse.hpo.target_strategy import (
    TargetStrategy,
    sample_target_strategy,
    validate_target_strategy_early,
)


@pytest.fixture
def target_catalog() -> TargetCatalog:
    return TargetCatalog(
        targets=["target", "target_alpha_20", "target_cyrusd_60", "target_sparse"]
    )


def test_sample_target_strategy_modes(target_catalog: TargetCatalog) -> None:
    rng = random.Random(0)
    modes = {
        sample_target_strategy(rng, target_catalog, fast=True).target_mode
        for _ in range(50)
    }
    assert "single" in modes


def test_validate_rejects_sparse_auxiliary(target_catalog: TargetCatalog) -> None:
    n = 100
    targets_df = pd.DataFrame(
        {
            "target": np.random.randn(n),
            "target_sparse": [np.nan] * 80 + list(np.random.randn(20)),
        }
    )
    strategy = TargetStrategy(
        target_mode="multi_blend",
        primary_target="target",
        auxiliary_targets=["target_sparse"],
    )
    result = validate_target_strategy_early(
        targets_df,
        strategy,
        catalog=target_catalog,
        rng=random.Random(1),
    )
    assert result.ok
    assert result.strategy.target_mode == "single"
    assert result.strategy.auxiliary_targets == []


def test_validate_accepts_valid_auxiliary(target_catalog: TargetCatalog) -> None:
    n = 100
    targets_df = pd.DataFrame(
        {
            "target": np.random.randn(n),
            "target_alpha_20": np.random.randn(n),
        }
    )
    strategy = TargetStrategy(
        target_mode="multi_blend",
        primary_target="target",
        auxiliary_targets=["target_alpha_20"],
    )
    result = validate_target_strategy_early(
        targets_df,
        strategy,
        catalog=target_catalog,
        rng=random.Random(2),
    )
    assert result.ok
    assert result.strategy.target_mode == "multi_blend"


def test_validate_does_not_invoke_build(target_catalog: TargetCatalog) -> None:
    targets_df = pd.DataFrame(
        {
            "target": np.random.randn(50),
            "target_sparse": [np.nan] * 40 + list(np.random.randn(10)),
        }
    )
    strategy = TargetStrategy(
        target_mode="multi_blend",
        primary_target="target",
        auxiliary_targets=["target_sparse"],
    )
    result = validate_target_strategy_early(
        targets_df,
        strategy,
        catalog=target_catalog,
        rng=random.Random(3),
    )
    assert result.ok
    assert result.strategy.target_mode == "single"
