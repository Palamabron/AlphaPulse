import importlib
import io
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cloudpickle


class PortablePredictBundle:
    """Load a prediction callable with its bundled AlphaPulse source code."""

    def __init__(self, payload: bytes, source_archive: bytes) -> None:
        self._payload = payload
        self._source_archive = source_archive
        self._predict_fn: Callable[..., Any] | None = None
        self._source_dir: tempfile.TemporaryDirectory[str] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._predict_fn is None:
            self._load_predict_fn()
        if self._predict_fn is None:
            raise RuntimeError("predict callable could not be loaded")
        return self._predict_fn(*args, **kwargs)

    def _load_predict_fn(self) -> None:
        self._source_dir = tempfile.TemporaryDirectory(
            prefix="alphapulse-predict-runtime-"
        )
        source_root = Path(self._source_dir.name)
        with zipfile.ZipFile(io.BytesIO(self._source_archive)) as archive:
            archive.extractall(source_root)
        sys.path.insert(0, str(source_root))
        for module_name in list(sys.modules):
            if module_name == "alphapulse" or module_name.startswith("alphapulse."):
                sys.modules.pop(module_name, None)
        importlib.invalidate_caches()
        self._predict_fn = cloudpickle.loads(self._payload)


def _source_archive() -> bytes:
    package_root = Path(__file__).resolve().parents[1]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_path in sorted(package_root.rglob("*.py")):
            relative_path = source_path.relative_to(package_root)
            archive.write(source_path, Path("alphapulse") / relative_path)
    return buffer.getvalue()


def dump_predict_fn(
    predict_fn: Callable[..., Any],
    path: Path,
) -> None:
    bundle = PortablePredictBundle(
        payload=cloudpickle.dumps(predict_fn),
        source_archive=_source_archive(),
    )
    module = sys.modules[__name__]
    cloudpickle.register_pickle_by_value(module)
    try:
        with open(path, "wb") as file:
            cloudpickle.dump(bundle, file)
    finally:
        cloudpickle.unregister_pickle_by_value(module)
