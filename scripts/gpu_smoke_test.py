"""Smoke-test GPU training for XGBoost, LightGBM, and CatBoost via AlphaPulse."""

from __future__ import annotations

import multiprocessing
import sys
from typing import Any

import numpy as np
import pandas as pd

from alphapulse.hpo.builder import instantiate_model
from alphapulse.hpo.search_space import apply_gpu_model_params
from alphapulse.models.era_ensemble_model import EraEnsembleModel


def _synthetic_frame(n: int = 3000, p: int = 15) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(rng.standard_normal(n))
    return X, y


def _train_model(model_type: str, n_rounds: int = 40) -> dict[str, Any]:
    X, y = _synthetic_frame()
    params = apply_gpu_model_params(model_type, {"params": {}})
    model = instantiate_model(model_type, params, index=0, n_subs=1)
    if not isinstance(model, EraEnsembleModel):
        raise TypeError(f"Expected EraEnsembleModel for {model_type}")
    base = model.base_model_factory()
    metrics = base.train(X, y, n_rounds=n_rounds)
    preds = base.predict(X.iloc[:10])
    return {"model": model_type, "metrics": metrics, "pred_shape": preds.shape}


def _spawn_worker(model_type: str, result_queue: multiprocessing.Queue[dict]) -> None:
    try:
        result_queue.put({"ok": True, "result": _train_model(model_type)})
    except Exception as exc:
        result_queue.put({"ok": False, "model": model_type, "error": str(exc)})


def main() -> int:
    print("=== AlphaPulse GPU integration smoke test ===\n")
    failures: list[str] = []

    for model_type in ("XGBoost", "LightGBM", "CatBoost"):
        try:
            result = _train_model(model_type)
            print(f"[in-process] {model_type}: OK  pred_shape={result['pred_shape']}")
        except Exception as exc:
            failures.append(f"{model_type} in-process: {exc}")
            print(f"[in-process] {model_type}: FAIL  {exc}")

    ctx = multiprocessing.get_context("spawn")
    for model_type in ("XGBoost", "LightGBM", "CatBoost"):
        queue: multiprocessing.Queue[dict] = ctx.Queue()
        proc = ctx.Process(target=_spawn_worker, args=(model_type, queue))
        proc.start()
        proc.join(timeout=120)
        if proc.is_alive():
            proc.kill()
            failures.append(f"{model_type} spawn: timeout")
            print(f"[spawn] {model_type}: TIMEOUT")
            continue
        payload = queue.get()
        if payload.get("ok"):
            shape = payload["result"]["pred_shape"]
            print(f"[spawn] {model_type}: OK  pred_shape={shape}")
        else:
            failures.append(f"{model_type} spawn: {payload.get('error')}")
            print(f"[spawn] {model_type}: FAIL  {payload.get('error')}")

    if failures:
        print("\nFailures:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\nAll GPU smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
