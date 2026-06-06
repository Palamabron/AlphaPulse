# Changelog

All notable changes to AlphaPulse are documented here.

---

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
