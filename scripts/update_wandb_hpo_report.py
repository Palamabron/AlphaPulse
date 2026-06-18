"""Update the published AlphaPulse HPO W&B report in place."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import wandb_workspaces.reports.v2 as wr

REPORT_URL = (
    "https://wandb.ai/dsc-pjatk-warsaw/alphapulse-hpo-20260616-232741/reports/"
    "AlphaPulse-HPO-Summary--VmlldzoxNzI1NDg1Ng=="
)
PROJECT = "alphapulse-hpo-20260616-232741"
TRIALS_DB = Path("artifacts/hpo_9h_v11/trials.db")

METRICS_SECTION_TITLE = "Metric definitions (holdout vs validation)"
METRICS_SECTION_BODY = """\
AlphaPulse HPO logs **two evaluation splits**. Do not compare columns across splits.

| W&B / leaderboard name | Split | Meaning |
|---|---|---|
| `holdout/HoldoutSharpe` | train holdout | corr_sharpe on the last train eras |
| `holdout/HoldoutMeanCorr` | train holdout | mean per-era Spearman on holdout |
| `validation/ValidationSharpe` | validation | corr_sharpe on `validation.parquet` |
| `validation/ValidationMmcSharpe` | validation | MMC Sharpe with `meta_model.parquet` |
| `validation/ValidationMeanCorr` | validation | mean per-era Spearman on validation |
| `validation/PayoutScore` | validation | **HPO objective** |

**Payout formula (Numerai-style):**

`PayoutScore = 0.75 * ValidationSharpe + 2.25 * ValidationMmcSharpe`

Leaderboard ranks by `PayoutScore` (validation). `HoldoutSharpe` is a separate generalization check on train.

**Diagnostics charts** are prefixed by split: `diagnostics/holdout/...` vs `diagnostics/validation/...`.
"""


def _trial_stats(db_path: Path) -> tuple[int, int, int, dict[str, float | int | str]]:
    conn = sqlite3.connect(db_path)
    counts = dict(
        conn.execute("SELECT status, COUNT(*) FROM trials GROUP BY status").fetchall()
    )
    completed = int(counts.get("completed", 0))
    failed = int(counts.get("failed", 0))
    total = completed + failed + int(counts.get("running", 0))

    best_payout: tuple | None = None
    best_holdout: tuple | None = None
    for tn, metrics_raw, fc_raw in conn.execute(
        "SELECT trial_number, metrics, flat_config FROM trials WHERE status='completed'"
    ):
        metrics = json.loads(metrics_raw or "{}")
        fc = json.loads(fc_raw or "{}")
        payout = metrics.get("payout_score")
        holdout = metrics.get("holdout_corr_sharpe", metrics.get("corr_sharpe"))
        val_sh = metrics.get("val_corr_sharpe")
        mmc = metrics.get("mmc_sharpe")
        n = int(fc.get("num_models", 1))
        models = "+".join(str(fc.get(f"model_{i}_type", "?")) for i in range(1, n + 1))
        row = (int(tn), holdout, val_sh, payout, mmc, models)
        if payout is not None and (best_payout is None or payout > best_payout[3]):
            best_payout = row
        if holdout is not None and (best_holdout is None or holdout > best_holdout[1]):
            best_holdout = row

    if best_payout is None or best_holdout is None:
        raise RuntimeError("No completed trials with metrics in trials.db")

    return (
        total,
        completed,
        failed,
        {
            "best_payout_trial": best_payout[0],
            "best_payout": float(best_payout[3]),
            "best_payout_val_sh": float(best_payout[2] or 0),
            "best_payout_holdout_sh": float(best_payout[1] or 0),
            "best_payout_mmc": float(best_payout[4] or 0),
            "best_payout_models": best_payout[5],
            "best_holdout_trial": best_holdout[0],
            "best_holdout_sh": float(best_holdout[1] or 0),
            "best_holdout_val_sh": float(best_holdout[2] or 0),
            "best_holdout_models": best_holdout[5],
        },
    )


def _snapshot_markdown(total: int, completed: int, failed: int) -> str:
    return (
        f"Generated from project `dsc-pjatk-warsaw/{PROJECT}`.\n\n"
        f"This report summarizes the hyperparameter and ensemble search: "
        f"**{total} trials started**, **{completed} completed**, **{failed} failed** "
        f"at last snapshot."
    )


def _executive_summary_fixed(stats: dict[str, float | int | str], failed: int) -> str:
    return (
        f"**Main finding:** `trial_{int(stats['best_holdout_trial']):03d}` "
        f"(`{stats['best_holdout_models']}`) has the strongest holdout signal: "
        f"`holdout/HoldoutSharpe = {float(stats['best_holdout_sh']):.3f}` and "
        f"`validation/ValidationSharpe = {float(stats['best_holdout_val_sh']):.3f}`.\n\n"
        f"**Payout leader:** `trial_{int(stats['best_payout_trial']):03d}` "
        f"(`{stats['best_payout_models']}`) reaches "
        f"`validation/PayoutScore = {float(stats['best_payout']):.3f}` "
        f"with `validation/ValidationMmcSharpe = {float(stats['best_payout_mmc']):.3f}`, "
        f"but `holdout/HoldoutSharpe = {float(stats['best_payout_holdout_sh']):.3f}`.\n\n"
        f"**Operational caveat:** prioritize stability — failed trials often reflect "
        f"TabPFN timeouts or GPU OOM; check `failed` count in the snapshot above."
    )


def _top_runs_table(stats: dict[str, float | int | str]) -> str:
    hp_t = int(stats["best_holdout_trial"])
    pp_t = int(stats["best_payout_trial"])
    return f"""\
| Rank lens | Best run | Model setup | Key metrics | Interpretation |
|---|---:|---|---|---|
| ValidationSharpe | `trial_{hp_t:03d}` | `{stats["best_holdout_models"]}` | ValidationSharpe `{float(stats["best_holdout_val_sh"]):.3f}`; HoldoutSharpe `{float(stats["best_holdout_sh"]):.3f}` | Best holdout generalization among completed runs. |
| HoldoutSharpe | `trial_{hp_t:03d}` | `{stats["best_holdout_models"]}` | HoldoutSharpe `{float(stats["best_holdout_sh"]):.3f}` | Same run — strongest train-era holdout Sharpe. |
| PayoutScore | `trial_{pp_t:03d}` | `{stats["best_payout_models"]}` | PayoutScore `{float(stats["best_payout"]):.3f}`; ValidationSharpe `{float(stats["best_payout_val_sh"]):.3f}`; HoldoutSharpe `{float(stats["best_payout_holdout_sh"]):.3f}` | Best validation payout objective; weak holdout Sharpe — tradeoff, not a single clear winner. |
"""


def _has_metrics_section(blocks: list) -> bool:
    return any(
        getattr(b, "text", None) == METRICS_SECTION_TITLE
        for b in blocks
        if type(b).__name__ == "H2"
    )


def main() -> None:
    total, completed, failed, stats = _trial_stats(TRIALS_DB)
    report = wr.Report.from_url(REPORT_URL)

    report.blocks[1].text = _snapshot_markdown(total, completed, failed)
    report.blocks[3].text = _executive_summary_fixed(stats, failed)
    report.blocks[5].text = (
        "The project is an HPO and ensemble search over tabular/predictive models. "
        "Families include `TabPFN`, `TabICL`, `LightGBM`, `XGBoost`, `CatBoost`, and "
        "`Packboost`, with ensemble modes `single`, `weighted`, and `stacking`.\n\n"
        "Logged metrics use explicit holdout vs validation names (see next section). "
        "The HPO objective is `validation/PayoutScore`."
    )
    report.blocks[9].text = _top_runs_table(stats)
    report.blocks[12].text = (
        "Diagnostics are split by evaluation lane:\n"
        "- `diagnostics/holdout/...` — per-era correlation, drawdown, feature exposure, "
        "SHAP/importance on train holdout\n"
        "- `diagnostics/validation/...` — payout/MMC scalars and validation per-era charts\n\n"
        "Artifacts include `run_table` and `wandb-history` collections per trial."
    )

    if not _has_metrics_section(report.blocks):
        report.blocks.insert(6, wr.H2(text=METRICS_SECTION_TITLE))
        report.blocks.insert(7, wr.MarkdownBlock(text=METRICS_SECTION_BODY))

    report.save(draft=False)

    verify = wr.Report.from_url(REPORT_URL)
    texts = []
    for block in verify.blocks:
        if hasattr(block, "text") and block.text:
            texts.append(block.text)
    joined = "\n".join(texts)
    for needle in (
        METRICS_SECTION_TITLE,
        "PayoutScore = 0.75 * ValidationSharpe + 2.25 * ValidationMmcSharpe",
        "holdout/HoldoutSharpe",
        f"trial_{int(stats['best_payout_trial']):03d}",
    ):
        if needle not in joined:
            raise RuntimeError(f"Verification failed: missing {needle!r}")
    print("Report updated and verified.")
    print(REPORT_URL)


if __name__ == "__main__":
    main()
