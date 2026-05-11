import numpy as np
import pandas as pd
import pytest

from alphapulse.hpo.builder import TREE_MODEL_NAMES, build_models
from alphapulse.models.foundation_models import TabICLModel, TabPFNModel

N_ROWS = 50
N_FEATURES = 4


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((N_ROWS, N_FEATURES)), columns=list("ABCD"))
    y = pd.Series(rng.standard_normal(N_ROWS))
    return X, y


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
