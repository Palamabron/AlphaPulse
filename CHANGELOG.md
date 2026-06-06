# Changelog

All notable changes to AlphaPulse are documented here.

---

## [0.5.0] — Production Hardening

- **HPO fault tolerance:** Each local trial runs in an isolated subprocess; a crash marks that trial failed and the sweep continues. A SQLite-backed `TrialDB` (`src/alphapulse/hpo/trial_db.py`) persists trial state across crashes. `--resume` flag skips already-completed trial numbers. `--trial-timeout` caps each subprocess.
- **Provenance artifact:** On every `export_numerai_pickle.py` run, a `*_provenance.json` bundle is written alongside the model: resolved config, `uv export` dependency snapshot, and git commit hash — enabling hermetic environment verification on resume or deploy.
- **Canonical artifact naming:** Exported models follow `<TIMESTAMP>_<ARCH>_<TARGET>_<CONFIG_HASH>` naming (e.g. `20260606T120000_XGBoost_target_a1b2c3d4_predict.pkl`) with a `latest_predict.pkl` symlink for convenience.
- **Masked loss for auxiliary targets:** `MultiTargetPipeline` now drops NaN rows per-target before training each model; targets with fewer than 10 valid rows are skipped entirely — prevents crashes with NaN-sparse auxiliary targets in both tree and DL models.
- **Feature neutralization in eval loop:** `Backtester` and `EraSplitEvaluator` accept an optional `FeatureNeutralizer`; when set, predictions are neutralized against feature columns before metric computation, rewarding genuinely novel alpha over crowded factor exposure.
- **W&B experiment runner integration:** `scripts/run_experiment.py` gains a `--wandb-project` flag; configs, per-era metrics, duration, config hash, and artifact paths are logged to W&B on every run. AutoResearch W&B wiring was already complete.

## [0.4.0] — Pre-Training Critical Path

- **Global seed threading:** Centralized `set_global_seed()` utility invoked at script genesis — locks Python `random`, `numpy`, `torch`, and cross-validation subsampling to guarantee identical walk-forward splits across independent executions.
- **Nested early stopping:** Each walk-forward training fold carves an inner temporal validation set (respecting purge/embargo) exclusively for loss monitoring; the outer fold remains untouched for metric reporting.
- **Per-era rank normalization before metrics:** `rank_normalize()` enforced strictly per era (not globally) before any correlation or Sharpe computation, matching Numerai's exact scoring pipeline.
- **Feature schema contract:** Exact ordered feature list serialised into every artifact; validated at load time — fail fast on missing columns, silently drop unexpected ones.
- **OOM protection / lazy data loading:** Memory-mapped or streaming access for the full dataset; native `DMatrix`/`Dataset` formats for tree models to avoid 3× memory spikes during histogram construction.
- **Export artifact smoke test:** After serialising `predict.pkl`, a clean subprocess loads it and runs a forward pass on a synthetic frame with edge-case columns — artifact marked deployment-ready only on success.
- **Column taxonomy:** Every dataset column tagged as `feature | target | auxiliary_target | metadata | benchmark` in config; benchmark columns (e.g. `v2_equivalent_return`) bypass model training but reach the evaluation module intact.

## [0.3.0] — Validation & Metrics

- **Feature routing validation:** Clear error messages for `input_group` mismatches validated at YAML parse time; `HeadSpec` distinguishes undefined group vs missing columns.
- **MMC metric tests:** Full test coverage for `mmc_score`, `per_era_mmc`, `era_sharpe_of_mmc`, and `payout_score`.

## [0.2.0] — Walk-Forward Backtesting

- **Purge-aware walk-forward backtesting:** `EraSplitEvaluator` with `n_purge`, `n_embargo`, `n_splits`.
- **Consistent HPO scoring:** All trials evaluated via 3-fold walk-forward CV — no fixed holdout split.
- **Full metric parity:** Walk-forward returns `max_drawdown`, `pct_positive_eras`, `n_valid_eras`.
- **Ensemble diagnostics and submission validation utilities.**
- **Per-era feature importance report.**

## [0.1.0] — Initial Release

- **AutoResearch Loop** with Claude agent.
- **Foundation Models:** TabPFN, TabICL.
- **Enhanced Model Suite:** EraEnsemble, SyntheticDataAugmenter.
- **Multi-target and Multi-head Pipelines.**
- **Robust NaN/inf Filtering** in Pipeline.
- **Ensemble Weight Optimization.**
- **Comprehensive EDA Dashboard** with HPO Analysis.
- **Config Tooling:** `scripts/make_feature_groups.py` converts `features.json` sets into YAML `features.groups`.
- **Unified Export:** `scripts/export_from_yaml.py` for YAML-driven production pickles.
