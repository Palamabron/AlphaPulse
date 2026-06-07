# AlphaPulse v0.5.0

AlphaPulse is a config-driven framework for building, training, and deploying ML pipelines for the [Numerai](https://numer.ai) stock-market prediction tournament. It covers the full workflow: dataset download, experiment definition, backtesting, hyperparameter optimization (HPO), and automated weekly submission.

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
  - [7. Live Inference](#7-live-inference-production-predictions)
  - [8. Submit Predictions](#8-submit-predictions-to-numerai)
- [Configuring Experiments (Experiment v1 YAML)](#configuring-experiments-experiment-v1-yaml)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

-----

## About Numerai

Numerai is a crowd-sourced hedge fund and the world's largest stock-market ML tournament. Data scientists submit predictions that Numerai combines into a meta-model for real hedge fund trading.

**Key concepts:**
| Term | Meaning |
|------|---------|
| **Era** | One week of data; rows within an era are correlated stocks |
| **Target** | 20-day forward stock-specific alpha, neutral to market/sector |
| **CORR** | Spearman correlation of your predictions to the target per era |
| **MMC** | Meta Model Contribution — how much your model improves beyond others |
| **Sharpe** | Mean per-era correlation ÷ its standard deviation |
| **NMR** | Numerai's token; stake on your model to earn rewards or get burned |

-----

## Installation & Setup

### Local Development

**Requirements:** Python 3.12+, Git, [uv](https://github.com/astral-sh/uv).

```bash
# Core dependencies (required)
uv sync --extra dev

# Optional: Add specific feature sets
uv sync --extra hpo              # HPO with Ray Tune
uv sync --extra deep             # Deep learning models (pytorch_tabular)
uv sync --extra foundation       # TabPFN, TabICL foundation models
uv sync --extra eda              # EDA dashboard dependencies

# Install all extras at once
uv sync --all-extras

# Install Git hooks
pre-commit install
pre-commit run --all-files
```

**Dependency Groups:**
- `dev` — Development tools: ruff, mypy, pytest, pre-commit
- `hpo` — Hyperparameter optimization: ray[tune], optuna (advanced)
- `deep` — Deep learning: pytorch_tabular, torch
- `foundation` — Foundation models: tabpfn, tabicl
- `eda` — Dashboard: streamlit, plotly, scikit-learn

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

### Make Targets (Quality Tooling)

AlphaPulse provides Make targets for common development tasks:

```bash
# Code Quality
make fmt          # Format code with ruff (imports + format)
make lint         # Lint code with ruff (check + format --check)
make types        # Run mypy type checking
make test         # Run pytest with coverage
make check        # Run all checks: lint + types + test + deadcode
make deadcode     # Find unused code with vulture

# EDA Module
make eda-lint     # Lint EDA dashboard code (standalone check)

# Notebooks
make nb-format    # Format notebooks with nbqa
make nb-lint      # Lint notebooks with nbqa

# Git Hooks
make precommit    # Run pre-commit hooks on all files
```

**Note:** EDA module (`eda/`) is excluded from `make check`, `make types`, and `make test`. Use `make eda-lint` for EDA quality checks.

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

> **No W&B account?** Prefix the command with `WANDB_MODE=disabled` to skip W&B logging. You can also add `WANDB_MODE=disabled` to your `.env` file to make it permanent.

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
Prefer using `scripts/run_test_pipeline.py` or `scripts/export_from_yaml.py` for YAML-driven exports.

### 7. Live Inference — Production Predictions

Run your trained model on live tournament data to generate predictions:

```bash
uv run python scripts/live_inference.py \
  --model-path artifacts/competition_pickle/predict.pkl \
  --data-dir data/v5.2 \
  --output-path artifacts/live/predictions.csv
```

**Parameters:**
- `--model-path`: Path to the exported `predict.pkl` (from step 6)
- `--data-dir`: Directory containing `live.parquet` (download fresh data before each round)
- `--output-path`: Where to save the submission CSV
- `--benchmark-col`: Benchmark column name (default: `v2_equivalent_return`)
- `--validate`: Run submission format validation (default: `true`)

**Output:** CSV file with `id` and `prediction` columns ready for submission.

### 8. Submit Predictions to Numerai

Upload your predictions to the tournament:

```bash
uv run python scripts/submit_predictions.py \
  --predictions-path artifacts/live/predictions.csv \
  --model-name my_model_name
```

**Prerequisites:**
- Set `NUMERAI_PUBLIC_API_KEY` and `NUMERAI_PRIVATE_API_KEY` in environment or `.env`
- Install numerapi: `pip install numerapi` (or add to pyproject.toml)

**Parameters:**
- `--predictions-path`: Path to the CSV from step 7
- `--model-name`: Your model name as it appears in the Numerai dashboard
- `--tournament`: Tournament identifier (default: `numerai`)
- `--validate`: Run format validation before upload (default: `true`)

The script will:
1. Validate submission format
2. Look up your model ID
3. Upload predictions for the current round
4. Return submission ID for tracking

### Complete Production Workflow

End-to-end flow for weekly tournament submissions:

```bash
# 1. Download latest data (do this each week before the deadline)
uv run python scripts/download_dataset.py \
  --dataset-version v5.2 \
  --output-dir data

# 2. Run live inference with your trained model
uv run python scripts/live_inference.py \
  --model-path artifacts/competition_pickle/predict.pkl \
  --data-dir data/v5.2 \
  --output-path artifacts/live/predictions.csv

# 3. Submit to Numerai
uv run python scripts/submit_predictions.py \
  --predictions-path artifacts/live/predictions.csv \
  --model-name my_model_name
```

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
    - **Foundation Models:**
      - `TabPFN` — TabPFN v2 in-context learning (n_estimators=8)
      - `TabPFN3` — TabPFN v3 local OSS (model_path="auto", n_estimators=8)
      - `TabPFN3Reasoning` — TabPFN v3 API with reasoning (thinking_mode=true, thinking_effort="medium")
      - `TabICL` — TabICL v2 in-context learning (n_estimators=8, kv_cache=false)
    - **Deep Learning:** `TabularDL` — pytorch_tabular wrapper (architecture: "ft_transformer"|"mlp")
    - **Meta Models:**
      - `EraEnsemble` — V3X-style era partitioning with Ridge meta-learner (n_subs=10)
      - `SyntheticDataAugmenter` — Diffusion-based data augmentation from elite rows (top_fraction=0.10, n_synthetic=500)
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

## EDA Dashboard

AlphaPulse includes a standalone Streamlit dashboard for exploratory data analysis of Numerai datasets. The dashboard provides interactive visualizations and statistical analysis with English/Polish bilingual support.

### Installation

Install EDA dependencies:

```bash
uv sync --extra eda
```

### Running the Dashboard

```bash
streamlit run eda/app.py
```

The dashboard will start at http://localhost:8501 by default.

### Configuration

Configure via environment variables:

```bash
export ALPHAPULSE_DATA_DIR=data/v5.2           # Path to data directory
export ALPHAPULSE_DATASET_VERSION=v5.2         # Dataset version string
```

Or create a `.env` file in the project root.

### Available Analysis Pages

The dashboard provides 8 specialized analysis modules:

1. **Target Analysis** — Target variable distribution, era-wise statistics, stability metrics, ridgeplots
2. **Feature Analysis** — Single/multi-feature exploration, correlation tracking, temporal behavior, batch analysis
3. **Correlations** — Feature-target correlations, inter-feature correlation matrices, network graphs
4. **Era Analysis** — Temporal patterns, era statistics, rolling metrics, feature stability over time
5. **Feature Distributions** — Distribution statistics, chi-square uniformity tests, entropy calculations
6. **Feature Importance** — Pearson correlation and LightGBM importance rankings with visualization
7. **Clustering** — Hierarchical clustering, dendrograms, redundancy detection, correlation heatmaps
8. **HPO Analysis** — Hyperparameter optimization results visualization (loads `all_trials.json` from artifacts)

### Features

- **Multi-target support** — Select any target column from the dataset
- **Feature set selection** — Choose between small (~20%), medium (~50%), or all (100%) features
- **Bilingual interface** — Toggle between English and Polish
- **Data validation** — Sanity checks for discrete vs continuous features
- **Interactive visualizations** — Plotly-based charts with zoom, pan, and export
- **Data export** — Download analysis results as CSV
- **Caching** — Efficient data loading and computation caching for large datasets

### Dashboard Structure

```
eda/
├── app.py                  # Main entry point
├── pages/                  # Analysis modules
│   ├── target_analysis.py
│   ├── feature_analysis.py
│   ├── correlations.py
│   ├── era_analysis.py
│   ├── feature_distributions.py
│   ├── feature_importance.py
│   ├── clustering.py
│   └── hpo_analysis.py
└── utils/                  # Utilities
    ├── config.py           # Path resolution and constants
    ├── data_loader.py      # NumeraiDataLoader wrapper
    ├── translations.py     # YAML-based translation system
    ├── i18n.py             # Translation helper (session state access)
    └── common.py           # Shared analysis utilities
```

### Development

The EDA module is self-contained and excluded from the main quality checks:

```bash
# Lint EDA code
make eda-lint

# The EDA module is excluded from:
# - mypy type checking
# - pytest coverage
# - Main make check pipeline
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
│   ├── export_from_yaml.py
│   ├── live_inference.py
│   ├── submit_predictions.py
│   ├── make_feature_groups.py
│   └── autoresearch.py
├── eda/             # Standalone Streamlit EDA dashboard
│   ├── app.py         # Main entry point (streamlit run eda/app.py)
│   ├── pages/         # Multi-page analysis modules
│   └── utils/         # Config & data loading (uses NumeraiDataLoader)
├── src/alphapulse/  # Core framework source code
│   ├── autoresearch/  # Agent-driven research loop
│   ├── evaluation/    # Backtesting, metrics, diagnostics, export validation
│   ├── experiments/   # YAML schema, runner, data loading
│   ├── hpo/           # HPO objective, search space, builder, registry
│   ├── logging_/      # Leaderboard and W&B helpers
│   ├── models/        # All model implementations + factory
│   ├── pipeline/      # Pipeline, ensemble, neutralizer, stacker
│   ├── preprocessors/ # All preprocessor implementations + factory
│   ├── utils/         # Seed utility (set_global_seed)
│   └── validation/    # Purged era cross-validation
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

## Roadmap

See [CHANGELOG.md](CHANGELOG.md) for completed releases.

**Completed — v0.5.0 (Production Hardening):**
- **HPO fault tolerance:** Each local trial runs in an isolated subprocess; crashes mark the trial failed and the sweep continues. A SQLite-backed `TrialDB` persists trial state. `--resume` skips already-completed trials.
- **Provenance artifact:** On every export, a hermetically sealed bundle is written: resolved config, `uv export` dependency snapshot, and git commit hash.
- **Canonical artifact naming:** Exported models follow `<TIMESTAMP>_<ARCH>_<TARGET>_<CONFIG_HASH>.pkl` with a `latest_predict.pkl` symlink.
- **Masked loss for auxiliary targets:** `MultiTargetPipeline` drops NaN rows per-target; targets with fewer than 10 valid rows are skipped entirely.
- **Feature neutralization in eval loop:** `Backtester` and `EraSplitEvaluator` accept an optional `FeatureNeutralizer`; predictions are neutralized before metric computation.
- **W&B experiment runner integration:** `scripts/run_experiment.py` logs configs, per-era metrics, and artifact paths to W&B via `--wandb-project`.

-----

## License

`alphapulse` is distributed under the terms of the **MIT license**.
