from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
from scripts.export_from_yaml import _internal_purge_eras
from scripts.hpo_pipeline import _diagnostic_era_sets

from alphapulse.experiments.schema import ExperimentV1


def test_yaml_export_uses_target_safe_internal_purge() -> None:
    experiment = SimpleNamespace(
        evaluation=SimpleNamespace(walk_forward_n_purge=8),
        data=SimpleNamespace(
            target_col="target_ender_60",
            auxiliary_targets=[],
        ),
    )

    assert _internal_purge_eras(cast(ExperimentV1, experiment)) == 16


def test_diagnostic_holdout_excludes_purge_gap() -> None:
    eras = [f"era_{index:04d}" for index in range(100)]
    era_train = pd.Series(np.repeat(eras, 2))

    train_set, holdout_set = _diagnostic_era_sets(era_train, purge_eras=16)

    assert train_set == set(eras[:64])
    assert holdout_set == set(eras[80:])
    assert set(eras[64:80]).isdisjoint(train_set | holdout_set)
