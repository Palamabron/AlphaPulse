import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SMOKE_SCRIPT = r"""
import json
import sys
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import alphapulse as preloaded_alphapulse


pkl_path = Path(sys.argv[1])
feature_columns = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
n_rows = int(sys.argv[3])
project_source_root = Path(sys.argv[4]).resolve()
preloaded_source = Path(preloaded_alphapulse.__file__).resolve()
sys.path[:] = [
    entry
    for entry in sys.path
    if not entry or Path(entry).resolve() != project_source_root
]
with open(pkl_path, "rb") as file:
    predict_fn = cloudpickle.load(file)
rng = np.random.default_rng(0)
all_columns = [*feature_columns, "__smoke_extra_1__", "__smoke_extra_2__"]
live_features = pd.DataFrame(
    rng.random((n_rows, len(all_columns))), columns=all_columns
)
live_benchmark_models = pd.DataFrame(
    {
        "v2_equivalent_return": rng.random(n_rows),
        "numerai_meta_model": rng.random(n_rows),
    }
)
result = predict_fn(live_features, live_benchmark_models)
import alphapulse

loaded_source = Path(alphapulse.__file__).resolve()
if loaded_source == preloaded_source:
    raise RuntimeError(
        f"export reused preloaded AlphaPulse instead of bundled source: {loaded_source}"
    )
if loaded_source.is_relative_to(project_source_root):
    raise RuntimeError(f"export loaded AlphaPulse from project source: {loaded_source}")
if not isinstance(result, pd.DataFrame):
    raise TypeError(f"expected DataFrame output, got {type(result).__name__}")
if list(result.columns) != ["prediction"]:
    raise ValueError(f"expected one prediction column, got {list(result.columns)}")
if len(result) != n_rows:
    raise ValueError(f"expected {n_rows} rows, got {len(result)}")
predictions = result["prediction"].to_numpy(dtype=np.float64)
if not np.isfinite(predictions).all():
    raise ValueError("predictions contain non-finite values")
if (predictions < 0.0).any() or (predictions > 1.0).any():
    raise ValueError(
        f"predictions are outside [0, 1]: "
        f"min={predictions.min():.4f}, max={predictions.max():.4f}"
    )
"""


def smoke_test_predict_fn(
    pkl_path: Path,
    feature_columns: list[str],
    n_rows: int = 10,
) -> None:
    """Validate an exported callable in a fresh isolated Python process."""
    if n_rows < 2:
        raise ValueError("n_rows must be >= 2")
    if not pkl_path.exists():
        raise FileNotFoundError(f"predict pickle not found: {pkl_path}")

    with tempfile.TemporaryDirectory(prefix="alphapulse-export-smoke-") as temp_dir:
        features_path = Path(temp_dir) / "features.json"
        features_path.write_text(json.dumps(feature_columns), encoding="utf-8")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-I",
                "-c",
                _SMOKE_SCRIPT,
                str(pkl_path.resolve()),
                str(features_path),
                str(n_rows),
                str(Path(__file__).resolve().parents[2]),
            ],
            cwd=temp_dir,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"smoke_test: predict validation failed: {details}")
