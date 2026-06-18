"""Smoke-test PackBoost CUDA and all locally available foundation models."""

from __future__ import annotations

import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

N_ROWS = 120
N_FEATURES = 24
N_ERAS = 12


@dataclass
class SmokeResult:
    name: str
    ok: bool
    seconds: float
    detail: str = ""


def _toy_era_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    cols = [f"f_{i}" for i in range(N_FEATURES)]
    x = pd.DataFrame(rng.integers(0, 5, size=(N_ROWS, N_FEATURES)), columns=cols)
    x["era"] = np.repeat([f"era_{i:04d}" for i in range(N_ERAS)], N_ROWS // N_ERAS)
    y = pd.Series(rng.standard_normal(N_ROWS), dtype=np.float32)
    return x, y


def _toy_float_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(43)
    cols = [f"f_{i}" for i in range(N_FEATURES)]
    x = pd.DataFrame(rng.standard_normal((N_ROWS, N_FEATURES)), columns=cols)
    y = pd.Series(rng.standard_normal(N_ROWS), dtype=np.float32)
    return x, y


def _run(name: str, fn: Callable[[], str]) -> SmokeResult:
    t0 = time.perf_counter()
    try:
        detail = fn()
        return SmokeResult(
            name=name, ok=True, seconds=time.perf_counter() - t0, detail=detail
        )
    except Exception as exc:
        return SmokeResult(
            name=name,
            ok=False,
            seconds=time.perf_counter() - t0,
            detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def smoke_packboost() -> str:
    from alphapulse.models.packboost_model import PackboostModel

    x, y = _toy_era_data()
    model = PackboostModel(
        device="cuda",
        n_rounds_base=8,
        n_rounds_boost=4,
        n_worst_eras=2,
        nfolds=4,
        max_depth=4,
        nfeatsets=2,
    )
    metrics = model.train(x, y)
    preds = model.predict(x)
    if not np.isfinite(preds).all():
        raise RuntimeError("non-finite PackBoost predictions")
    return f"n_boost_eras={metrics.get('n_boost_eras', 0):.0f}, pred_mean={preds.mean():.6f}"


def smoke_tabpfn() -> str:
    from alphapulse.models.foundation_models import TabPFNModel

    x, y = _toy_float_data()
    model = TabPFNModel(
        max_train_rows=80,
        max_features=16,
        compression="pca",
        compression_components=8,
        n_estimators=2,
        ignore_pretraining_limits=True,
    )
    model.train(x, y)
    preds = model.predict(x)
    if not np.isfinite(preds).all():
        raise RuntimeError("non-finite TabPFN predictions")
    return f"pred_mean={preds.mean():.6f}"


def smoke_tabpfn3() -> str:
    from alphapulse.models.foundation_models import TabPFN3Model

    x, y = _toy_float_data()
    model = TabPFN3Model(
        max_train_rows=80,
        max_features=16,
        compression="pca",
        compression_components=8,
        n_estimators=2,
    )
    model.train(x, y)
    preds = model.predict(x)
    if not np.isfinite(preds).all():
        raise RuntimeError("non-finite TabPFN3 predictions")
    return f"pred_mean={preds.mean():.6f}"


def smoke_tabicl() -> str:
    from alphapulse.models.foundation_models import TabICLModel

    x, y = _toy_float_data()
    model = TabICLModel(
        max_train_rows=80,
        max_features=16,
        compression="pca",
        compression_components=8,
        n_estimators=2,
    )
    model.train(x, y)
    preds = model.predict(x)
    if not np.isfinite(preds).all():
        raise RuntimeError("non-finite TabICL predictions")
    return f"pred_mean={preds.mean():.6f}"


def main() -> int:
    from alphapulse.hpo.search_space import available_foundation_models
    from alphapulse.models.packboost_backend import packboost_cuda_available

    tests: list[tuple[str, Callable[[], str]]] = []
    if packboost_cuda_available():
        tests.append(("Packboost(CUDA)", smoke_packboost))
    else:
        print("SKIP Packboost: CUDA/PackBoost unavailable")

    for model_name in available_foundation_models():
        if model_name == "TabPFN":
            tests.append(("TabPFN", smoke_tabpfn))
        elif model_name == "TabPFN3":
            tests.append(("TabPFN3", smoke_tabpfn3))
        elif model_name == "TabICL":
            tests.append(("TabICL", smoke_tabicl))

    results: list[SmokeResult] = []
    print(f"Running {len(tests)} smoke test(s)...")
    for name, fn in tests:
        print(f"\n--- {name} ---")
        result = _run(name, fn)
        results.append(result)
        status = "PASS" if result.ok else "FAIL"
        print(
            f"{status} ({result.seconds:.1f}s) {result.detail.splitlines()[0] if result.detail else ''}"
        )
        if not result.ok:
            print(result.detail)

    passed = sum(1 for r in results if r.ok)
    print(f"\n=== Summary: {passed}/{len(results)} passed ===")
    for r in results:
        mark = "OK" if r.ok else "FAIL"
        print(f"  [{mark}] {r.name} ({r.seconds:.1f}s)")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
