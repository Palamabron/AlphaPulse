# AlphaPulse
0.1.0 (internal release)

AlphaPulse is a config-driven framework for building, training, and deploying Numerai competition pipelines.

Numerai is a global data science tournament where you build ML models to predict stock-market signals; AlphaPulse streamlines dataset download, experiment definition, evaluation/backtesting, and automated HPO for that workflow.

-----

## Table of Contents

- [Installation & Setup](#installation--setup)
  - [Local Development](#local-development)
  - [Common Dev Commands](#common-dev-commands)
  - [Rebuild Environment](#rebuild-environment)
- [Numerai Competition Pipeline](#numerai-competition-pipeline)
  - [1. Download Dataset](#1-download-dataset)
  - [2. Run an Experiment (YAML-driven)](#2-run-an-experiment-yaml-driven)
  - [3. Run HPO (Automated Search)](#3-run-hpo-automated-search)
  - [4. Run AutoResearch (Agent-Driven Research Loop)](#4-run-autoresearch-agent-driven-research-loop)
  - [5. Quick Test & Smoke Test](#5-quick-test--smoke-test)
  - [6. Exporting for Submission](#6-exporting-for-submission)
- [Configuring Experiments (Experiment v1 YAML)](#configuring-experiments-experiment-v1-yaml)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

-----

## Installation & Setup

### Local Development

**Requirements:** Python 3.12+, Git, [uv](https://github.com/astral-sh/uv).

```bash
# Install dependencies (including dev extras)
uv sync --extra dev

# Install Git hooks
pre-commit install
pre-commit run --all-files
```

### Common Dev Commands

```bash
# Linting and Formatting
uv run ruff check src tests
uv run ruff format .

# Type Checking
uv run mypy src/alphapulse tests

# Tests
uv run pytest tests/ -v --tb=short

# Add/Remove Packages
uv add tenacity loguru
uv remove tenacity
```

### Rebuild Environment

When dependencies change in `pyproject.toml`:

```bash
# Fast sync
uv sync --extra dev

# Clean rebuild
rm -rf .venv
uv sync --extra dev
```

-----

## Numerai Competition Pipeline

AlphaPulse supports an end-to-end flow: preparing data, defining experiments via YAML, running HPO, and exporting a Numerai-ready `predict.pkl`.

### 1\. Download Dataset

The downloader expects Numerai API keys in environment variables: `NUMERAI_PUBLIC_API_KEY` and `NUMERAI_PRIVATE_API_KEY`.

This repo supports a local `.env` file (loaded by `python-dotenv`), or you can export variables in your shell:

```bash
export NUMERAI_PUBLIC_API_KEY="..."
export NUMERAI_PRIVATE_API_KEY="..."
```

```bash
uv run python scripts/download_dataset.py \
  --dataset-version v5.2 \
  --output-dir data
```

*Expected files in `data/v5.2/`: `train.parquet`, `validation.parquet`, and `features.json`.*

### 2\. Run an Experiment (YAML-driven)

Use this for manual iteration. Define your architecture in a YAML file and run:

```bash
uv run python scripts/run_experiment.py \
  --config experiments/example_v1.yaml \
  --artifact-dir artifacts/experiments
```

This script builds the pipeline, trains on the training set, and outputs backtest metrics for the validation split.

### 3\. Run HPO (Automated Search)

Use `scripts/hpo_pipeline.py` to search over preprocessing steps, models, and ensemble methods.

```bash
uv run python scripts/hpo_pipeline.py \
  --data-dir data/v5.2 \
  --train-subsample 0.125 \
  --num-trials 30 \
  --output-dir artifacts/hpo_x8 \
  --local
```

The best resulting configuration will be saved to `artifacts/hpo_x8/best_config.json`.

### 4\. Run AutoResearch (Agent-Driven Research Loop)

Use `scripts/autoresearch.py` to let a Claude agent drive the research process, iteratively proposing and testing pipeline improvements (adding models, tuning hyperparameters, changing ensembles, etc.).

```bash
uv run python scripts/autoresearch.py \
  --data-dir data/v5.2 \
  --train-subsample 0.125 \
  --trials 50 \
  --hours 2 \
  --output-dir artifacts/autoresearch \
  --agent-model claude-sonnet-4-6
```

The agent will:
- Analyze trial results and decide what to try next
- Propose mutations: add/remove models, tune hyperparameters, change ensemble methods, add preprocessors, set neutralization
- Track progress and reasoning in `research_state.json`
- Output the best configuration to `best_config.json` and a summary to `trials_summary.csv`

Optional: Start from an existing config with `--seed-config path/to/config.json`, or resume with `--resume`.

### 5\. Quick Test & Smoke Test

Use the "test pipeline" for a lightweight run that trains on a small subsample, backtests, and exports a pickle in one go.

```bash
uv run python scripts/run_test_pipeline.py \
  --data-dir data/v5.2 \
  --train-subsample 0.05 \
  --output-dir artifacts/test_run
```

### 6\. Exporting for Submission

To generate the `predict.pkl` required for Numerai:

**From an HPO result:**

```bash
uv run python scripts/export_numerai_pickle.py \
  --data-dir data/v5.2 \
  --best-config-path artifacts/hpo_x8/best_config.json \
  --train-subsample 0.125 \
  --output-dir artifacts/competition_pickle
```

**From a YAML Experiment:**
Prefer using `scripts/run_test_pipeline.py` or the specific export scripts currently in the [Roadmap](#roadmap).

-----

## Configuring Experiments (Experiment v1 YAML)

The YAML format is defined by `src/alphapulse/experiments/schema.py`.

### Minimal Example

```yaml
version: "1"
data:
  data_dir: data/v5.2
  train_subsample: 0.05
  target_col: target
  seed: 42

features:
  columns: null # null defaults to all features
  groups: {}

preprocessing: []

models:
  - type: XGBoost
    params:
      max_depth: 3
      learning_rate: 0.05
      tree_method: hist
      objective: reg:squarederror

ensemble_method: single
ensemble_params: {}

train:
  n_rounds: 40
  early_stopping_rounds: 5

evaluation:
  primary_metric: mean_per_era_correlation
```

### Advanced Features

  * **Feature Groups:** Define `features.groups` as a mapping of `group_name -> [columns]`. You can then assign specific models to specific groups using `models[].input_group: group_name`.
  * **Available Preprocessors:** `StandardScaler`, `RobustScaler`, `PCA`, `TruncatedSVD`, `GaussianNoise`, `VarianceSelector`, `LGBMImportanceSelector`, `Packboost`, and `GroupedPreprocessor`.
  * **Available Models:**
    - **Gradient Boosting:** `XGBoost`, `LightGBM`, `CatBoost`, `Packboost`
    - **Tree Ensembles:** `RandomForest`, `ExtraTrees`
    - **Linear:** `Ridge`
    - **Foundation Models:** `TabPFN`, `TabICL` (tabular foundation models)
    - **Meta Models:** `EraEnsemble` (era-specific ensemble), `SyntheticDataAugmenter` (diffusion-based augmentation)
  * **Ensembling:** `single`, `weighted`, or `stacking` (Meta-learners: `ridge` or `xgboost`).

-----

## Data Loading (Python API)

For programmatic access to Numerai data outside of the CLI scripts, use `NumeraiDataLoader`:

```python
from alphapulse import NumeraiDataLoader

loader = NumeraiDataLoader("data/v5.2", feature_set="medium")
train = loader.load_split("train", subsample=0.1)

train.X       # feature DataFrame
train.y       # target Series
train.era     # era Series (or None)
train.n_rows, train.n_features
```

-----

## Directory Structure

```text
├── artifacts/       # Model outputs, pickles, and HPO logs
├── data/            # Downloaded Numerai parquet files
├── experiments/     # YAML configuration files
├── scripts/         # Executable workflow scripts
│   ├── download_dataset.py
│   ├── run_experiment.py
│   ├── hpo_pipeline.py
│   ├── run_test_pipeline.py
│   ├── export_numerai_pickle.py
│   └── autoresearch.py
├── eda/             # Standalone Streamlit EDA dashboard
│   ├── app.py         # Main entry point (streamlit run eda/app.py)
│   ├── pages/         # Multi-page analysis modules
│   └── utils/         # Config & data loading (uses NumeraiDataLoader)
├── src/alphapulse/  # Core framework source code
│   ├── autoresearch/  # Agent-driven research loop
│   ├── logging_/      # Leaderboard and W&B helpers
│   └── ...            # pipeline, models, hpo, experiments, etc.
└── tests/           # Unit tests
```

-----

## Contributing

PRs are welcome. Please keep changes focused and ensure the pre-commit hooks pass:

```bash
pre-commit install
pre-commit run --all-files
```

Commit messages: prefer conventional commits (e.g. `feat: ...`, `fix: ...`, `docs: ...`).

-----

## Recent Additions (v0.1.0)

✅ **AutoResearch Loop:** Claude agent-driven research with automated mutation strategies
✅ **New Foundation Models:** TabPFN and TabICL support for tabular data
✅ **Enhanced Model Suite:** EraEnsemble, SyntheticDataAugmenter, and expanded sklearn models
✅ **Advanced Pipeline Features:** Multi-target, multi-head, and feature neutralization support
✅ **Robust Data Handling:** Automatic NaN/inf filtering in pipeline fit and predict
✅ **Ensemble Optimizer:** Automated ensemble weight optimization

## Roadmap

1.  **Payout Optimization:** Integrate Numerai payout-style scoring directly into HPO and experiment selection.
2.  **Config Tooling:** Add a helper to automatically convert `features.json` sets into YAML `features.groups`.
3.  **Unified Export:** Create a dedicated "export from YAML" script to bridge the gap between manual experiments and production pickles.
4.  **Validation:** Improve error messages for feature routing mismatches (e.g., missing columns in a group).
5.  **Metrics:** Add dedicated tests for MMC (Meta-Model Contribution) and alignment with Numerai benchmark signals.
6.  **Weights & Biases Integration:** Full logging and visualization support for experiments and AutoResearch.

-----

## License

`alphapulse` is distributed under the terms of the **MIT license**.
