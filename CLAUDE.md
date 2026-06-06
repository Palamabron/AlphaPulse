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

# Install HPO dependencies (Ray Tune)
uv sync --extra hpo

# Install deep learning models (pytorch_tabular)
uv sync --extra deep

# Install foundation models (TabPFN, TabICL)
uv sync --extra foundation

# Install EDA dependencies
uv sync --extra eda

# Install pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Quality Checks via Make
```bash
make fmt        # Format code (ruff imports + format)
make lint       # Lint code (ruff check + format --check)
make types      # Run mypy type checking
make test       # Run pytest with coverage
make check      # Run lint + types + test + deadcode
make deadcode   # Find dead code with vulture
make eda-lint   # Lint EDA dashboard code
make nb-format  # Format notebooks with nbqa
make nb-lint    # Lint notebooks with nbqa
```

### Or run tools directly
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

The EDA dashboard is a standalone Streamlit app that lives in the top-level `eda/` directory (not inside `src/alphapulse/`). It imports `NumeraiDataLoader` from the core library for data loading, but is otherwise self-contained with its own config and page modules. The UI supports English and Polish.

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
- `eda/app.py` — main entry point (dataset overview, sanity checks, navigation)
- `eda/pages/` — multi-page analysis modules: `clustering`, `correlations`, `era_analysis`, `target_analysis`, `feature_analysis`, `feature_distributions`, `feature_importance`, `hpo_analysis`
- `eda/locales/` — YAML translation files (en.yaml, pl.yaml)
- `eda/utils/translations.py` — translation system (YAML-based)
- `eda/utils/config.py` — path resolution with env var overrides
- `eda/utils/data_loader.py` — wraps `NumeraiDataLoader`
- `eda/utils/common.py` — shared utilities (correlations, era stats, download buttons, chart theming)

The EDA app is excluded from mypy, ruff, test coverage, and the main `make check` pipeline. Use `make eda-lint` for standalone linting.

**Translation System:**

The EDA dashboard uses YAML-based translations for bilingual support (English/Polish). All UI text should be externalized to translation files.

```python
from eda.utils import get_translations

# Get translations for current language (from session state)
t = get_translations()

# Simple usage
st.title(t["app.title"])
st.header(t["dataset.header"])

# With variable interpolation
st.success(t.format("dataset.data_loaded", target="target_cyrus_v4_20"))
st.metric(t["overview.rows"], f"{len(train):,}")

# Safe get with default
text = t.get("optional.key", default="Fallback text")
```

Translation files are organized by category with dot-notation keys:
- `common.*` — shared UI elements (download_csv, loading, error, etc.)
- `app.*` — main app page
- `dataset.*` — dataset configuration
- `overview.*` — dataset overview section
- `target_info.*` — target information
- `hpo.*` — HPO analysis page
- `errors.*` — error messages

When adding new UI text:
1. Add key to both `eda/locales/en.yaml` and `eda/locales/pl.yaml`
2. Use dot notation for organization (e.g., `"my_page.section.element"`)
3. Use `{variable}` syntax for interpolation in YAML
4. Access via `t["my_page.section.element"]` or `t.format("my_page.section.element", variable="value")`

## Architecture

### Core Abstractions

**Pipeline** (`src/alphapulse/pipeline/pipeline.py`)
- Central orchestrator: chains preprocessors, trains models, and combines predictions via ensemble strategies
- `fit(X, y, X_val, y_val, **model_train_kwargs)` → trains preprocessors and all models
- `predict(X, eras=None)` → preprocesses features and returns ensemble predictions
- **Robust data handling:** automatically filters NaN/inf values before and after preprocessing, imputes invalid predictions with median; helpers in `pipeline/row_utils.py`
- `to_numerai_predict(benchmark_col)` → wraps the pipeline in a Numerai-compatible callable for submission
- `save_pipeline(path)` / `load_pipeline(path)` → cloudpickle serialization

**MultiTargetPipeline** (`src/alphapulse/pipeline/multi_target.py`)
- Trains separate pipelines for multiple target columns
- `fit()` and `predict()` for multi-target scenarios

**MultiHeadPipeline** (`src/alphapulse/pipeline/multihead.py`)
- Trains multiple specialized "heads" with different configurations
- Each head can use different models, preprocessing, or feature subsets
- `HeadSpec` defines per-head configuration: `model`, `input_columns`, `input_group`, `local_preprocessors`, `feature_groups`

**FeatureNeutralizer** (`src/alphapulse/pipeline/neutralizer.py`)
- Neutralizes predictions against specified feature columns
- `neutralize(predictions, features, eras=None)` → removes feature exposure
- `optimize_proportion(predictions, features, y_true, eras)` → finds best neutralization proportion

**Stacker** (`src/alphapulse/pipeline/stacker.py`)
- Stacking ensemble with out-of-fold predictions
- Meta-learner: Ridge or XGBoost
- `collect_oof(trial_params_list)` → gathers OOF predictions across CV folds
- `score_individual(oof_matrix, y_oof, eras_oof)` → ranks individual models

**EnsembleOptimizer** (`src/alphapulse/pipeline/ensemble_optimizer.py`)
- Scipy-based ensemble weight optimization using validation set (`method="SLSQP"`)
- `fit(oof_matrix, y_oof, eras_oof)` → optimize weights
- `predict(pred_matrix)` → apply weights

**BaseModel** (`src/alphapulse/models/base.py`)
- Abstract interface: `train()`, `predict()`, `save()`, `load()`
- **Gradient Boosting Models:** `XGBoostModel`, `LightGBMModel`, `CatBoostModel`, `PackboostModel`
- **Tree Ensembles:** `RandomForestModel`, `ExtraTreesModel`
- **Linear Models:** `RidgeModel`
- **Foundation Models:**
  - `TabPFNModel(n_estimators=8)` — TabPFN v2 in-context learning
  - `TabPFN3Model(model_path="auto", n_estimators=8)` — TabPFN v3 local OSS
  - `TabPFN3ReasoningModel(thinking_mode=True, thinking_effort="medium")` — TabPFN v3 API with reasoning
  - `TabICLModel(n_estimators=8, kv_cache=False)` — TabICL v2 in-context learning
- **Deep Learning:** `TabularDLModel(architecture="ft_transformer"|"mlp", dl_params, trainer_params)` — pytorch_tabular wrapper
- **Meta Models:**
  - `EraEnsembleModel(base_model_factory, n_subs=10)` — V3X-style era partitioning with Ridge meta-learner
  - `SyntheticDataAugmenter(top_fraction=0.10, n_synthetic=500, backend="auto")` — diffusion-based data augmentation from elite rows
- `ModelFactory.create(model_type, params)` → instantiates models from config
- `ModelFactory.suggest(trial)` → Optuna-based HPO model sampling (wraps tree models in `EraEnsembleModel`)

**BasePreprocessor** (`src/alphapulse/preprocessors/base.py`)
- Abstract interface: `fit(X, y)`, `transform(X)`, `fit_transform(X, y)`
- `TrainEvalPreprocessor` protocol: `train()` / `eval()` switching for noise injection
- **Scaling:** `StandardScalerPreprocessor`, `RobustScalerPreprocessor`
- **Dimensionality Reduction:** `PCAPreprocessor(n_components)`, `TruncatedSVDPreprocessor(n_components=10)`
- **Feature Selection:** `VarianceFeatureSelector(keep_fraction, threshold, mode)`, `LGBMImportanceSelector(keep_fraction=0.75, n_estimators=100)`, `EraStableFeatureSelector(keep_fraction, n_estimators, stability_weight)` — ranks features by blended mean importance / cross-era stability score
- **Noise Injection:** `GaussianNoiseInjector(sigma=0.01, seed=42)` — active only during `train()` mode
- **Special:** `PackboostPreprocessor` — adds Packboost predictions as an extra feature; `GroupedPreprocessor(groups, group_preprocessors)` — applies different preprocessing chains to named feature groups
- `PreprocessorFactory(n_features, prefix)` → Optuna-based HPO factory with `suggest(trial)` and `suggest_fixed(...)` methods

**EnsembleStrategy** (`src/alphapulse/pipeline/ensemble.py`)
- Supports: `single` (pass-through), `weighted` (weighted average), `stacking` (meta-learner: ridge or xgboost)
- `fit(n_models, get_val_predictions, y_val)` → for stacking, trains meta-model on validation OOF predictions
- `combine(predictions)` → returns final ensemble prediction

**PurgedEraCV** (`src/alphapulse/validation/purged_cv.py`)
- Era-aware cross-validation with purging and embargo to prevent look-ahead bias in temporal data
- `PurgedEraCV(n_splits=5, n_purge=4, n_embargo=4, max_train_eras=None, min_train_eras=10)`
- `split(X, y, groups)` → yields train/test index arrays, respecting era boundaries
- `split_eras(era_series)` → yields (train_eras, test_eras) string lists
- `summary(era_series)` → overview of split sizes
- Critical for backtesting Numerai models, as rows within an era are correlated

**NumeraiDataLoader** (`src/alphapulse/data.py`)
- `NumeraiDataLoader(data_dir, feature_set=None, target_col="target")`
- Loads parquet files from a versioned data directory (e.g. `data/v5.2/`)
- `load_split(split_name, subsample=None, seed=42)` → returns `NumeraiDataset`
- `NumeraiDataset` properties: `.X` (features DataFrame), `.y` (target Series), `.era` (era Series), `.n_rows`, `.n_features`
- Resolves feature sets from `features.json` (e.g. `"small"`, `"medium"`, `"all"`)

### Experiment System

**YAML Configuration** (`src/alphapulse/experiments/schema.py`)
- `ExperimentV1`: pydantic model defining the experiment schema (version "1")
- Key sections: `data`, `features`, `preprocessing`, `models`, `ensemble_method`, `train`, `evaluation`
- `features.groups`: define named feature subsets; assign models to groups via `models[].input_group`
- `ModelSpec(type, params={}, input_columns=None, input_group=None, preprocessors=[], n_subs=10)`
- `EvaluationConfig(primary_metric="corr_sharpe", era_holdout_last_n=None, walk_forward=False)`
- `to_pipeline_config()` → converts to dict for `build_pipeline()`

**Experiment Runner** (`src/alphapulse/experiments/runner.py`)
- `RunResult` dataclass — `metrics`, `config_hash`, `duration_sec`, `paths`, `error`, `pipeline_config`
- Loads YAML, validates against schema, constructs pipeline, trains, backtests, and saves artifacts

### HPO System

**Objective Function** (`src/alphapulse/hpo/objective.py`)
- `TrialResult` dataclass — `sharpe`, `metrics`, `config`, `model_types`, `elapsed_seconds`, `error`
- `run_trial(config, X_train, y_train, X_val, y_val, era_val, feature_cols, seed)` → builds pipeline, trains, backtests, returns `TrialResult`
- `ray_trainable(config, **kwargs)` → wrapper for Ray Tune integration

**Search Space** (`src/alphapulse/hpo/search_space.py`)
- `sample_random_config(seed=None, phase="phase_b")` → generates random flat config dict
- `get_full_param_space()` → Ray Tune search space definition
- `resolve_flat_config(flat_config)` → converts flat HPO dict to nested pipeline config
- `get_train_kwargs_from_flat(flat)` → extracts `n_rounds`, `early_stopping_rounds` from flat config

**Builder** (`src/alphapulse/hpo/builder.py`)
- `build_pipeline(config, feature_columns)` → constructs `Pipeline` from config dict
- `build_multi_head_pipeline(config, feature_columns, feature_groups)` → constructs `MultiHeadPipeline`
- `build_pipeline_or_multi(...)` → auto-selects pipeline type based on config
- `build_preprocessors(config)`, `build_models(config)` → sub-builders

**Registry** (`src/alphapulse/hpo/registry.py`)
- `MODEL_REGISTRY` — dict of `(class, default_params)` for all models
- `PREPROCESSOR_REGISTRY` — dict of `(class, default_params)` for all preprocessors

### AutoResearch System

**Research Loop** (`src/alphapulse/autoresearch/loop.py`)
- `run_autoresearch(X_train, y_train, X_val, y_val, era_val, feature_cols, max_hours, max_trials, output_dir, seed_config, seed, agent_model, resume)` → main entry point
- Runs trials within time/count budget with Claude agent deciding what to try next
- Outputs: `best_config.json`, `research_state.json`, `trials_summary.csv`, `leaderboard.json`

**Agent** (`src/alphapulse/autoresearch/agent.py`)
- `decide_next_action(state, model="claude-sonnet-4-6")` → calls Claude to produce a `MutationDecision`
- `MutationDecision` dataclass — `action_name`, `action_kwargs`, `reasoning`
- Maintains strategy guidance and scoring context in its system prompt

**Mutations** (`src/alphapulse/autoresearch/mutations.py`)
- `add_model(config, model_type, params)`
- `remove_model(config, model_index)`
- `tune_model_params(config, model_index, param_updates)`
- `change_ensemble(config, method, params)`
- `add_preprocessor(config, preprocessor_type, params, position)`
- `remove_preprocessor(config, position)`
- `set_neutralization(config, proportion)`

**State** (`src/alphapulse/autoresearch/state.py`)
- `TrialRecord` dataclass — `trial_number`, `sharpe`, `metrics`, `config`, `model_types`, `elapsed_seconds`, `action_taken`, `agent_reasoning`, `error`
- `ResearchState` — tracks `trials`, `best_trial`, `current_config`, `start_time`; supports resume from disk

### Evaluation

**Backtester** (`src/alphapulse/evaluation/backtester.py`)
- `Backtester(predictor, feature_columns=None)` — evaluates any object with `predict(X)`
- `evaluate(X, y, era, meta_model_preds=None)` → dict with `sharpe`, `mean_per_era_correlation`, `max_drawdown`, `pct_positive_eras`, `n_valid_eras`

**Metrics** (`src/alphapulse/evaluation/metrics.py`)
- `rank_normalize(preds)` → maps predictions to [0, 1] uniform distribution (required for Numerai submissions)
- `per_era_correlation(y_true, y_pred, eras, method="spearman")` → Series of per-era correlations
- `per_era_spearman(y_true, y_pred, eras)` → convenience wrapper
- `era_sharpe(y_true, y_pred, eras)` → Sharpe ratio of per-era correlations
- `mmc_score(y_true, y_pred, meta_model, eras)` → Meta Model Contribution
- `era_correlation_metrics(y_true, y_pred, eras)` → dict: `mean_era_corr`, `std_era_corr`, `corr_sharpe`, `max_drawdown`, `pct_positive_eras`, `n_valid_eras`
- `calculate_metrics(y_true, y_pred, eras)` → canonical backtest metrics dict

**Era Splitting** (`src/alphapulse/evaluation/era_split.py`)
- Utilities for splitting data by era for validation

**Export Validation** (`src/alphapulse/evaluation/export_validation.py`)
- `smoke_test_predict_fn(pkl_path, feature_columns, n_rows=10)` → loads `predict.pkl` in a clean subprocess, runs a forward pass on a synthetic frame with edge-case columns, raises on any failure

**Submission Validation** (`src/alphapulse/evaluation/submission.py`)
- `validate_submission(predictions_df, live_df, id_col, pred_col)` → returns list of validation error strings; empty list means submission is well-formed

**Ensemble Diagnostics** (`src/alphapulse/evaluation/ensemble_diagnostics.py`)
- `correlation_matrix(oof_predictions, eras)` → per-era inter-model Spearman correlation matrix; high correlation means ensemble gains little over a single model

**Feature Report** (`src/alphapulse/evaluation/feature_report.py`)
- `compute_feature_report(X, y, eras)` → per-era feature importance diagnostics; tracks mean importance and cross-era stability

**Global Seed Utility** (`src/alphapulse/utils/seed.py`)
- `set_global_seed(seed)` → locks Python `random`, NumPy, PyTorch (if available), and sets `PYTHONHASHSEED` so child processes (e.g. Ray workers) inherit a fixed hash seed

## Code Conventions

These are defined in `.cursor/coding-guidelines.md` and apply to all new code:

- Write clean, simple, readable code — small functions/classes that do one thing
- Type hints required for all functions
- **No comments** — Delete unnecessary comments. If information is crucial, use Google-style docstrings instead
- **Docstrings (when needed):** Use Google format for complex functions only. Most functions should be self-documenting
- Follow SOLID, KISS, DRY, YAGNI principles
- Prefer composition over inheritance
- Fail fast: validate inputs early, raise clear errors
- Return early to avoid deep nesting
- No magic numbers — use named constants

**Comment Policy:**
- Inline comments are code smell — refactor to make code self-explanatory
- Section comments (e.g., `# ── Data loading ──`) should be removed — use functions/classes for organization
- Only exception: complex algorithms where the "why" isn't obvious from the code itself
- If you need to explain what code does, the code needs rewriting
- If you need to explain why (business logic, non-obvious requirements), use a docstring

**Google-Style Docstring Example:**
```python
def complex_function(data: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    """Process data and return metrics above threshold.

    Args:
        data: Input DataFrame with 'value' and 'category' columns.
        threshold: Minimum value to include in results. Defaults to 0.5.

    Returns:
        Dict mapping category names to their aggregated values.

    Raises:
        ValueError: If data is empty or missing required columns.
    """
```

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
3. Register in `PreprocessorFactory.create()` and `PREPROCESSOR_REGISTRY`
4. Add to `__init__.py` exports

**Adding a new model:**
1. Create class in `src/alphapulse/models/` inheriting `BaseModel`
2. Implement `train()`, `predict()`, `save()`, `load()`
3. Register in `ModelFactory.create()` and `MODEL_REGISTRY`
4. Add to `__init__.py` exports

**Adding a new ensemble method:**
1. Extend `EnsembleStrategy.fit()` and `combine()` in `src/alphapulse/pipeline/ensemble.py`
2. Update the `Literal` type in `src/alphapulse/experiments/schema.py` for `ensemble_method`
