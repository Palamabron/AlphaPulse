from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphapulse.hpo.builder import TREE_MODEL_NAMES, build_models
from alphapulse.models.foundation_models import (
    TabICLModel,
    TabPFN3Model,
    TabPFNModel,
)

N_ROWS = 50
N_FEATURES = 4
N_WIDE_FEATURES = 12


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((N_ROWS, N_FEATURES)), columns=list("ABCD"))
    y = pd.Series(rng.standard_normal(N_ROWS))
    return X, y


@pytest.fixture
def wide_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        rng.standard_normal((N_ROWS, N_WIDE_FEATURES)),
        columns=[f"f{i}" for i in range(N_WIDE_FEATURES)],
    )
    y = pd.Series(rng.standard_normal(N_ROWS))
    return X, y


class FakeRegressor:
    def __init__(self) -> None:
        self.fit_X: pd.DataFrame | None = None
        self.fit_y: pd.Series | None = None
        self.predict_calls: list[pd.DataFrame] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.fit_X = X
        self.fit_y = y

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_calls.append(X)
        return np.zeros(len(X))


def _faked(model: TabPFNModel) -> tuple[TabPFNModel, FakeRegressor]:
    fake = FakeRegressor()
    model._make_regressor = lambda: fake  # type: ignore[method-assign]
    return model, fake


# ---------------------------------------------------------------------------
# FoundationModel scaling behavior (mocked regressor)
# ---------------------------------------------------------------------------


def test_train_subsamples_rows_to_cap(toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = toy_data
    model, fake = _faked(TabPFNModel(max_train_rows=20))
    model.train(X, y)
    assert fake.fit_X is not None and fake.fit_y is not None
    assert len(fake.fit_X) == 20
    assert len(fake.fit_y) == 20


def test_train_keeps_all_rows_under_cap(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    model, fake = _faked(TabPFNModel(max_train_rows=1000))
    model.train(X, y)
    assert fake.fit_X is not None
    assert len(fake.fit_X) == N_ROWS


def test_subsampling_is_deterministic(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    model_a, fake_a = _faked(TabPFNModel(max_train_rows=20, seed=7))
    model_b, fake_b = _faked(TabPFNModel(max_train_rows=20, seed=7))
    model_a.train(X, y)
    model_b.train(X, y)
    assert fake_a.fit_X is not None and fake_b.fit_X is not None
    pd.testing.assert_frame_equal(fake_a.fit_X, fake_b.fit_X)


def test_compression_applied_when_too_wide(
    wide_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = wide_data
    model, fake = _faked(TabPFNModel(max_features=4, compression="pca"))
    model.train(X, y)
    assert fake.fit_X is not None
    assert fake.fit_X.shape[1] == 4
    assert all(c.startswith("pca_") for c in fake.fit_X.columns)


def test_compression_applied_at_predict(
    wide_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = wide_data
    model, fake = _faked(TabPFNModel(max_features=4, compression="svd"))
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (N_ROWS,)
    assert fake.predict_calls[0].shape[1] == 4


def test_compression_skipped_when_narrow(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    model, fake = _faked(TabPFNModel(max_features=500))
    model.train(X, y)
    assert fake.fit_X is not None
    assert list(fake.fit_X.columns) == list(X.columns)


def test_compression_disabled_with_none(
    wide_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = wide_data
    model, fake = _faked(TabPFNModel(max_features=4, compression=None))
    model.train(X, y)
    assert fake.fit_X is not None
    assert fake.fit_X.shape[1] == N_WIDE_FEATURES


def test_compression_components_override(
    wide_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = wide_data
    model, fake = _faked(
        TabPFNModel(max_features=4, compression="pca", compression_components=2)
    )
    model.train(X, y)
    assert fake.fit_X is not None
    assert fake.fit_X.shape[1] == 2


def test_train_imputes_nans_before_compression(
    wide_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = wide_data
    X = X.copy()
    X.iloc[0, 0] = np.nan
    model, fake = _faked(TabPFNModel(max_features=4, compression="pca"))
    model.train(X, y)
    preds = model.predict(X)
    assert fake.fit_X is not None
    assert not fake.fit_X.isna().any().any()
    assert np.isfinite(preds).all()


def test_predict_is_chunked(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    model, fake = _faked(TabPFNModel(predict_chunk_rows=20))
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (N_ROWS,)
    assert len(fake.predict_calls) == 3


def test_invalid_compression_raises() -> None:
    with pytest.raises(ValueError, match="Unknown compression method"):
        TabPFNModel(compression="vae")


def test_invalid_max_train_rows_raises() -> None:
    with pytest.raises(ValueError, match="max_train_rows"):
        TabPFNModel(max_train_rows=0)


def test_save_load_roundtrip_keeps_compressor(
    wide_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    X, y = wide_data
    model, _ = _faked(TabPFNModel(max_features=4, compression="pca"))
    model.train(X, y)
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = TabPFNModel().load(path)
    preds = loaded.predict(X)
    assert preds.shape == (N_ROWS,)


def test_build_models_passes_scaling_params() -> None:
    spec = {
        "type": "TabPFN",
        "params": {"max_train_rows": 123, "compression": "svd"},
    }
    models = build_models([spec])
    model = models[0]
    assert isinstance(model, TabPFNModel)
    assert model.max_train_rows == 123
    assert model.compression == "svd"


# ---------------------------------------------------------------------------
# TabPFNModel
# ---------------------------------------------------------------------------


def test_tabpfn_model_init_defaults() -> None:
    model = TabPFNModel()
    assert model.name == "TabPFN"
    assert model.n_estimators == 8
    assert not model.is_trained


def test_tabpfn_model_train_predict(toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
    pytest.importorskip("tabpfn", reason="tabpfn not installed — skip")
    X, y = toy_data
    model = TabPFNModel(n_estimators=4, ignore_pretraining_limits=True)
    model.train(X, y)
    assert model.is_trained
    preds = model.predict(X)
    assert preds.shape == (N_ROWS,)
    assert np.isfinite(preds).all()


def test_tabpfn_model_no_numeric_raises() -> None:
    pytest.importorskip("tabpfn", reason="tabpfn not installed — skip")
    X = pd.DataFrame({"era": ["a", "b", "c"]})
    y = pd.Series([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="no numeric feature columns"):
        TabPFNModel().train(X, y)


def test_tabpfn_not_in_tree_model_names() -> None:
    assert "TabPFN" not in TREE_MODEL_NAMES


def test_tabpfn_registry_returns_tabpfn_model() -> None:
    models = build_models([{"type": "TabPFN", "params": {}}])
    assert isinstance(models[0], TabPFNModel)


# ---------------------------------------------------------------------------
# TabPFN3Model
# ---------------------------------------------------------------------------


def test_tabpfn3_model_init_defaults() -> None:
    model = TabPFN3Model()
    assert model.name == "TabPFN3"
    assert model.model_path == "auto"
    assert model.n_estimators == 8
    assert not model.is_trained


def test_tabpfn3_registry_returns_tabpfn3_model() -> None:
    models = build_models([{"type": "TabPFN3", "params": {}}])
    assert isinstance(models[0], TabPFN3Model)


def test_tabpfn3_not_in_tree_model_names() -> None:
    assert "TabPFN3" not in TREE_MODEL_NAMES


# ---------------------------------------------------------------------------
# TabICLModel
# ---------------------------------------------------------------------------


def test_tabicl_model_init_defaults() -> None:
    model = TabICLModel()
    assert model.name == "TabICL"
    assert model.n_estimators == 8
    assert not model.is_trained


def test_tabicl_model_train_predict(toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
    pytest.importorskip("tabicl", reason="tabicl not installed — skip")
    X, y = toy_data
    model = TabICLModel(n_estimators=4)
    model.train(X, y)
    assert model.is_trained
    preds = model.predict(X)
    assert preds.shape == (N_ROWS,)
    assert np.isfinite(preds).all()


def test_tabicl_model_no_numeric_raises() -> None:
    pytest.importorskip("tabicl", reason="tabicl not installed — skip")
    X = pd.DataFrame({"era": ["a", "b", "c"]})
    y = pd.Series([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="no numeric feature columns"):
        TabICLModel().train(X, y)


def test_tabicl_not_in_tree_model_names() -> None:
    assert "TabICL" not in TREE_MODEL_NAMES


def test_tabicl_registry_returns_tabicl_model() -> None:
    models = build_models([{"type": "TabICL", "params": {}}])
    assert isinstance(models[0], TabICLModel)
