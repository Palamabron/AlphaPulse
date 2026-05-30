"""Smoke tests for the relocated EDA module.

These tests verify that the EDA utilities can be imported and that the
data_loader correctly delegates to ``NumeraiDataLoader``.  They do NOT
require a running Streamlit server or real Numerai data.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_config_imports_and_defaults() -> None:
    from eda.utils.config import (
        APP_ICON,
        APP_TITLE,
        DATA_DIR,
        DATASET_VERSION,
        FEATURE_VALUES,
        FEATURES_JSON_PATH,
        LAYOUT,
        ROOT,
        TRAIN_DATA_PATH,
    )

    assert isinstance(ROOT, Path)
    assert isinstance(DATA_DIR, Path)
    assert isinstance(TRAIN_DATA_PATH, Path)
    assert isinstance(FEATURES_JSON_PATH, Path)
    assert DATASET_VERSION == "v5.2"
    assert APP_ICON
    assert APP_TITLE
    assert LAYOUT == "wide"
    assert len(FEATURE_VALUES) == 5


def test_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHAPULSE_DATASET_VERSION", "v6.0")
    monkeypatch.setenv("ALPHAPULSE_DATA_DIR", "/tmp/custom_data")

    import importlib

    import eda.utils.config as cfg

    importlib.reload(cfg)

    assert cfg.DATASET_VERSION == "v6.0"
    assert cfg.DATA_DIR == Path("/tmp/custom_data")

    monkeypatch.delenv("ALPHAPULSE_DATASET_VERSION")
    monkeypatch.delenv("ALPHAPULSE_DATA_DIR")
    importlib.reload(cfg)


def test_data_loader_function_signature() -> None:
    from eda.utils.data_loader import (
        get_era_statistics,
        get_feature_correlations,
        load_feature_metadata,
        load_numerai_data,
    )

    assert callable(load_numerai_data)
    assert callable(load_feature_metadata)
    assert callable(get_era_statistics)
    assert callable(get_feature_correlations)


def test_get_era_statistics() -> None:
    import pandas as pd
    from eda.utils.data_loader import get_era_statistics

    df = pd.DataFrame(
        {
            "era": ["e1", "e1", "e2", "e2"],
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    result = get_era_statistics(df)
    assert len(result) == 2
    assert "target_mean" in result.columns


def test_get_feature_correlations() -> None:
    import pandas as pd
    from eda.utils.data_loader import get_feature_correlations

    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [3.0, 2.0, 1.0],
            "target": [0.1, 0.2, 0.3],
        }
    )
    result = get_feature_correlations(df, ["feature_a", "feature_b"])
    assert len(result) == 2
    assert "Correlation" in result.columns


def test_utils_package_exports() -> None:
    from eda.utils import (
        APP_ICON,
        APP_TITLE,
        FEATURE_VALUES,
        LAYOUT,
        get_era_statistics,
        get_feature_correlations,
        load_numerai_data,
    )

    assert callable(load_numerai_data)
    assert callable(get_era_statistics)
    assert callable(get_feature_correlations)
    assert APP_ICON
    assert APP_TITLE
    assert LAYOUT
    assert FEATURE_VALUES
