# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaPulse is a config-driven framework for building, training, and deploying ML pipelines for the Numerai stock-market prediction tournament. It provides a full workflow from dataset download through experiment definition, backtesting, hyperparameter optimization (HPO), and export of Numerai-compatible prediction functions.

**Tech Stack**: Python 3.12+, pandas, xgboost/lightgbm/catboost, optuna, cloudpickle, pydantic (for YAML schema validation), tyro (CLI), uv (package management)

## Development Commands

### Setup
```bash
# Install dependencies (including dev extras)
uv sync --extra dev

# Install pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Quality Checks
```bash
# Lint & format
uv run ruff check src tests
uv run ruff format .

# Type checking
uv run mypy src/alphapulse tests

# Run all tests
uv run pytest tests/ -v --tb=short

# Run a single test file
uv run pytest tests/test_foo.py -v

# Run a specific test
uv run pytest tests/test_foo.py::test_bar -v
```

### Numerai Workflow Scripts

**Download dataset:**
```bash
# Requires NUMERAI_PUBLIC_API_KEY and NUMERAI_PRIVATE_API_KEY in environment or .env
uv run python scripts/download_dataset.py --dataset-version v5.2 --output-dir data
```

**Run a YAML-defined experiment:**
```bash
uv run python scripts/run_experiment.py \
  --config experiments/example_v1.yaml \
  --artifact-dir artifacts/experiments
```

**Run HPO (hyperparameter search):**
```bash
uv run python scripts/hpo_pipeline.py \
  --data-dir data/v5.2 \
  --train-subsample 0.125 \
  --num-trials 30 \
  --output-dir artifacts/hpo_x8 \
  --local
```

**Smoke test pipeline:**
```bash
uv run python scripts/run_test_pipeline.py \
  --data-dir data/v5.2 \
  --train-subsample 0.05 \
  --output-dir artifacts/test_run
```

**Export Numerai submission pickle:**
```bash
uv run python scripts/export_numerai_pickle.py \
  --data-dir data/v5.2 \
  --best-config-path artifacts/hpo_x8/best_config.json \
  --train-subsample 0.125 \
  --output-dir artifacts/competition_pickle
```

**Run AutoResearch (agent-driven research loop):**
```bash
uv run python scripts/autoresearch.py \
  --data-dir data/v5.2 \
  --train-subsample 0.125 \
  --trials 50 \
  --hours 2 \
  --output-dir artifacts/autoresearch \
  --agent-model claude-sonnet-4-6
```

## EDA Dashboard

The EDA dashboard is a standalone Streamlit app that lives in the top-level `eda/` directory (not inside `src/alphapulse/`). It imports `NumeraiDataLoader` from the core library for data loading, but is otherwise self-contained with its own config and page modules.

**Install EDA dependencies:**
```bash
uv sync --extra eda
```

**Run the dashboard:**
```bash
streamlit run eda/app.py
```

**Environment overrides:**
- `ALPHAPULSE_DATA_DIR` — path to the data directory (default: `data/v5.2`)
- `ALPHAPULSE_DATASET_VERSION` — dataset version string (default: `v5.2`)

**Structure:**
- `eda/app.py` — main entry point
- `eda/pages/` — multi-page analysis modules (clustering, correlations, era analysis, target analysis, feature analysis, feature distributions, feature importance)
- `eda/utils/config.py` — path resolution with env var overrides
- `eda/utils/data_loader.py` — wraps `NumeraiDataLoader`

The EDA app is excluded from mypy, ruff, test coverage, and the main `make check` pipeline. Use `make eda-lint` for standalone linting.

## Architecture

### Core Abstractions

**Pipeline** (`src/alphapulse/pipeline/pipeline.py`)
- Central orchestrator: chains preprocessors, trains models, and combines predictions via ensemble strategies
- `fit(X, y, X_val, y_val, **model_train_kwargs)` → trains preprocessors and all models
- `predict(X)` → preprocesses features and returns ensemble predictions
- **Robust data handling:** automatically filters NaN/inf values before and after preprocessing, imputes invalid predictions with median
- `to_numerai_predict(benchmark_col)` → wraps the pipeline in a Numerai-compatible callable for submission
- `save_pipeline(path)` / `load_pipeline(path)` → cloudpickle serialization

**MultiTargetPipeline** (`src/alphapulse/pipeline/multi_target.py`)
- Trains separate pipelines for multiple target columns
- `fit()` and `predict()` for multi-target scenarios

**MultiHeadPipeline** (`src/alphapulse/pipeline/multihead.py`)
- Trains multiple specialized "heads" with different configurations
- Each head can use different models, preprocessing, or feature subsets
- `HeadSpec` defines per-head configuration

**FeatureNeutralizer** (`src/alphapulse/pipeline/neutralizer.py`)
- Neutralizes predictions against specified feature columns
- Removes systematic biases or exposures

**Stacker** (`src/alphapulse/pipeline/stacker.py`)
- Stacking ensemble with out-of-fold predictions
- Meta-learner: Ridge or XGBoost

**EnsembleOptimizer** (`src/alphapulse/pipeline/ensemble_optimizer.py`)
- Optimizes ensemble weights using validation set
- Maximizes chosen metric (e.g., Sharpe, correlation)

**BaseModel** (`src/alphapulse/models/base.py`)
- Abstract interface: `train()`, `predict()`, `save()`, `load()`
- **Gradient Boosting Models:** `XGBoostModel`, `LightGBMModel`, `CatBoostModel`, `PackboostModel`
- **Tree Ensembles:** `RandomForestModel`, `ExtraTreesModel`
- **Linear Models:** `RidgeModel`
- **Foundation Models:** `TabPFNModel`, `TabICLModel` (tabular deep learning, zero/few-shot)
- **Meta Models:** `EraEnsembleModel` (trains separate models per era), `SyntheticDataAugmenter` (diffusion-based data augmentation)
- `ModelFactory.create(model_type, params)` → instantiates models from config
- `suggest_augmentation(config)` → helper to add data augmentation to pipeline

**BasePreprocessor** (`src/alphapulse/preprocessors/base.py`)
- Abstract interface: `fit()`, `transform()`
- **Scaling:** `StandardScalerPreprocessor`, `RobustScalerPreprocessor`
- **Dimensionality Reduction:** `PCAPreprocessor`, `TruncatedSVDPreprocessor`
- **Feature Selection:** `VarianceFeatureSelector`, `LGBMImportanceSelector`
- **Noise Injection:** `GaussianNoiseInjector`
- **Special:** `PackboostPreprocessor`, `GroupedPreprocessor` (applies preprocessor to feature groups)
- `PreprocessorFactory` (`src/alphapulse/preprocessors/factory.py`) instantiates preprocessors from config dicts

**EnsembleStrategy** (`src/alphapulse/pipeline/ensemble.py`)
- Supports: `single` (pass-through), `weighted` (weighted average), `stacking` (meta-learner: ridge or xgboost)
- `fit(n_models, get_val_predictions, y_val)` → for stacking, trains meta-model on validation OOF predictions
- `combine(predictions)` → returns final ensemble prediction

**PurgedEraCV** (`src/alphapulse/validation/purged_cv.py`)
- Era-aware cross-validation with purging and embargo to prevent look-ahead bias in temporal data
- `split(X, y, groups)` → yields train/test index arrays, respecting era boundaries and purge/embargo parameters
- `split_eras(era_series)` → yields (train_eras, test_eras) string lists
- Critical for backtesting Numerai models, as rows within an era are correlated

**NumeraiDataLoader** (`src/alphapulse/data.py`)
- Loads parquet files from a versioned data directory (e.g. `data/v5.2/`)
- `load_split(split_name, subsample=None)` → returns `NumeraiDataset` with `.X`, `.y`, `.era` properties
- Resolves feature sets from `features.json` (e.g. `"small"`, `"medium"`, `"all"`)

### Experiment System

**YAML Configuration** (`src/alphapulse/experiments/schema.py`)
- `ExperimentV1`: pydantic model defining the experiment schema (version "1")
- Key sections: `data`, `features`, `preprocessing`, `models`, `ensemble_method`, `train`, `evaluation`
- `features.groups`: define named feature subsets; assign models to groups via `models[].input_group`
- `to_pipeline_config()` → converts to dict for `build_pipeline()`

**Experiment Runner** (`src/alphapulse/experiments/runner.py`)
- Loads YAML, validates against schema, constructs pipeline, trains, backtests, and saves artifacts

### HPO System

**Objective Function** (`src/alphapulse/hpo/objective.py`)
- `run_trial(config, X_train, y_train, X_val, y_val, era_val, feature_cols, seed)` → builds pipeline, trains, backtests, returns metrics dict
- `ray_trainable(config, **kwargs)` → wrapper for Ray Tune integration

**Search Space** (`src/alphapulse/hpo/search_space.py`)
- `sample_random_config()` → generates random flat config dict
- `resolve_flat_config(flat_config)` → converts flat HPO dict to nested pipeline config

**Builder** (`src/alphapulse/hpo/builder.py`)
- `build_pipeline(config, feature_columns, feature_groups)` → constructs `Pipeline` from config dict
- `build_pipeline_or_multi(...)` → handles both single and multi-target pipelines

**Registry** (`src/alphapulse/hpo/registry.py`)
- Centralized registry for models, preprocessors, and ensemble methods
- Provides search space definitions for HPO

### AutoResearch System

**Research Loop** (`src/alphapulse/autoresearch/loop.py`)
- `run_autoresearch(X_train, y_train, X_val, y_val, era_val, feature_cols, **kwargs)` → main entry point
- Runs trials within time/count budget with Claude agent deciding what to try next
- Outputs: `best_config.json`, `research_state.json`, `trials_summary.csv`

**Agent** (`src/alphapulse/autoresearch/agent.py`)
- Claude-powered research agent that analyzes trial history and proposes next experiments
- Decides between mutation strategies based on recent performance
- Maintains reasoning trail in research state

**Mutations** (`src/alphapulse/autoresearch/mutations.py`)
- Mutation strategies: `add_model`, `remove_model`, `tune_model_params`, `change_ensemble`, `add_preprocessor`, `remove_preprocessor`, `set_neutralization`
- Each mutation modifies the pipeline config in a specific way
- Agent chooses which mutation to apply and with what parameters

**State** (`src/alphapulse/autoresearch/state.py`)
- `ResearchState`: tracks trial history, agent reasoning, and best result
- `TrialRecord`: stores config, metrics, error, model types, and agent action for each trial
- Supports resume from disk

### Evaluation

**Backtester** (`src/alphapulse/evaluation/backtester.py`)
- `evaluate(X, y, era)` → computes era-based metrics (per-era correlation, sharpe, mean correlation, etc.)
- Returns dict with keys like `sharpe`, `mean_per_era_correlation`, `correlation`, `max_drawdown`

**Metrics** (`src/alphapulse/evaluation/metrics.py`)
- `per_era_correlation(y_true, y_pred, era)` → correlation within each era
- `per_era_spearman(y_true, y_pred, era)` → Spearman correlation per era
- `era_sharpe(per_era_corr)` → sharpe ratio of per-era correlations
- `era_correlation_metrics(y_true, y_pred, era)` → full suite of era-based metrics
- `calculate_metrics(y_true, y_pred, era)` → comprehensive metrics dict
- `rank_normalize(preds)` → maps predictions to [0, 1] uniform distribution (required for Numerai submissions)

**Era Splitting** (`src/alphapulse/evaluation/era_split.py`)
- Utilities for splitting data by era for validation

## Code Conventions

These are defined in `.cursor/coding-guidelines.md` and apply to all new code:

- Write clean, simple, readable code — small functions/classes that do one thing
- Type hints required for all functions
- No docstrings or comments unless logic is genuinely non-obvious
- Follow SOLID, KISS, DRY, YAGNI principles
- Prefer composition over inheritance
- Fail fast: validate inputs early, raise clear errors
- Return early to avoid deep nesting
- No magic numbers — use named constants

## Feature Routing

When working with multi-model pipelines:
- `features.groups` in YAML maps group names to column lists
- `models[].input_group` assigns a model to a specific feature group
- If `input_group` is `null`, the model receives all features or the global `features.columns` list
- The pipeline validates at runtime that all referenced groups exist and contain valid columns

## What is Numerai?

Numerai is a crowd-sourced hedge fund and the world's largest stock market ML tournament. Data scientists worldwide submit predictions that Numerai combines into a **Stake-Weighted Meta Model** used for real hedge fund trading.

**How it works:**
- Numerai provides free, obfuscated (encrypted) tabular financial data — each row is a stock at a point in time
- Participants train models and submit live predictions (Tuesday–Saturday, new round each week)
- Numerai aggregates all submissions into the Meta Model for actual trading
- Scoring takes ~1 month because the target measures 20 business days of future returns

**Dataset structure (v5.2):**
- `era`: Friday of each week (the prediction date); rows within an era are correlated stocks
- `features`: Quantitative stock attributes (fundamentals like P/E, technicals like RSI, market data, analyst ratings) — obfuscated so no financial domain knowledge is needed
- Feature sets: `"small"`, `"medium"`, `"all"` — defined in `features.json`
- `target`: Stock-specific alpha (returns neutral to market/country/sector beta), 20-day forward-looking
- Auxiliary targets: Different neutralizations and time horizons (20- or 60-day); may contain NaNs

**Scoring metrics:**
- **CORR (Correlation)**: Spearman correlation of predictions to the target within each era; primary metric
- **MMC (Meta Model Contribution)**: How much your predictions improve the Meta Model beyond what others already provide; rewards originality
- **Sharpe**: Mean per-era correlation divided by its standard deviation; measures consistency

**Staking (NMR token):**
- Stake NMR (Numerai's cryptocurrency) on your model to earn rewards or get burned
- Positive performance → earn more NMR; negative performance → NMR is permanently burned (destroyed)
- Staking improves your weight in the Meta Model; unstaking takes ~1 month

## Numerai-Specific Details

- **Era**: A time period (e.g. week) grouping correlated rows. Cross-validation must respect era boundaries.
- **Purging**: Dropping eras between train/test splits to eliminate look-ahead bias.
- **Embargo**: Dropping eras after the test set to simulate realistic submission timing.
- **Rank normalization**: Numerai requires predictions in [0, 1] uniform distribution. Use `rank_normalize()` before submission.
- **Submission format**: `predict.pkl` containing a callable `predict(live_features, live_benchmark_models) -> DataFrame` with a single `prediction` column.

## Testing Philosophy

From `.cursor/coding-guidelines.md`:
- Test behavior, not implementation
- One assertion per logical concept
- Arrange-Act-Assert structure
- No test should depend on another test

## Common Patterns

**Adding a new preprocessor:**
1. Create class in `src/alphapulse/preprocessors/` inheriting `BasePreprocessor`
2. Implement `fit(X, y)` and `transform(X)` → `pd.DataFrame`
3. Register in `PreprocessorFactory.create()`
4. Add to `__init__.py` exports

**Adding a new model:**
1. Create class in `src/alphapulse/models/` inheriting `BaseModel`
2. Implement `train()`, `predict()`, `save()`, `load()`
3. Register in `ModelFactory.create()`
4. Add to `__init__.py` exports

**Adding a new ensemble method:**
1. Extend `EnsembleStrategy.fit()` and `combine()` in `src/alphapulse/pipeline/ensemble.py`
2. Update the `Literal` type in `src/alphapulse/experiments/schema.py` for `ensemble_method`

## Key Recent Changes

**Pipeline robustness (v0.1.0):**
- Pipeline now automatically filters NaN and inf values before and after preprocessing in both `fit()` and `predict()`
- Invalid rows during prediction are imputed with median of valid predictions
- Prevents training errors from bad data while maintaining output shape

**AutoResearch system (v0.1.0):**
- New Claude agent-driven research loop that autonomously explores pipeline configurations
- Agent analyzes trial history and proposes strategic mutations
- Supports time and trial count budgets
- Full state persistence and resume capability

**Expanded model library (v0.1.0):**
- Foundation models: TabPFN and TabICL for zero/few-shot tabular learning
- Meta models: EraEnsemble for era-specific modeling, SyntheticDataAugmenter for diffusion-based data augmentation
- All sklearn tree models: RandomForest, ExtraTrees
- Ridge regression for linear baselines

**Advanced pipeline features (v0.1.0):**
- Multi-target and multi-head pipeline support
- Feature neutralization for removing systematic biases
- Ensemble weight optimization
- Stacking with OOF predictions
