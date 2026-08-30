import argparse
import importlib
import json
import queue
from pathlib import Path
from typing import Any

_TORCH_MODEL_TYPES = frozenset({"Packboost", "TabICL", "TabPFN", "TabPFN3"})


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _requires_torch(worker_kwargs: dict[str, Any]) -> bool:
    config = worker_kwargs["flat_config"]
    num_models = int(config.get("num_models", 1))
    model_types = {
        str(config.get(f"model_{index}_type", "XGBoost"))
        for index in range(1, num_models + 1)
    }
    return bool(model_types & _TORCH_MODEL_TYPES or config.get("use_packboost"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    worker_kwargs = json.loads(args.input.read_text(encoding="utf-8"))
    if _requires_torch(worker_kwargs):
        importlib.import_module("torch")

    from scripts.hpo_pipeline import _trial_worker

    result_queue: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
    _trial_worker(result_queue=result_queue, **worker_kwargs)
    payload = result_queue.get()

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
