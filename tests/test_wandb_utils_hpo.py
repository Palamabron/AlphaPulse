from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from alphapulse.hpo.objective import TrialResult
from alphapulse.logging_.wandb_utils import log_hpo_convergence, log_hpo_trial_metrics


def test_log_hpo_trial_metrics_logs_feature_routing_fields() -> None:
    mock_wandb = MagicMock()
    result = TrialResult(
        trial_number=1,
        sharpe=1.0,
        metrics={
            "corr_sharpe": 1.0,
            "holdout_corr_sharpe": 1.0,
            "val_corr_sharpe": 0.4,
            "payout_score": 0.9,
            "mmc_sharpe": 0.2,
        },
        model_type="XGBoost",
        elapsed_seconds=12.0,
        params={
            "active_groups": ["small", "strength"],
            "routed_feature_count": 512,
        },
        corr_sharpe=1.0,
        mmc_sharpe=0.2,
        payout_score=0.9,
    )
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        log_hpo_trial_metrics(result, objective=0.9, model_types="XGBoost")

    logged = mock_wandb.log.call_args.args[0]
    assert logged["active_groups"] == "small+strength"
    assert logged["active_groups_count"] == 2
    assert logged["routed_feature_count"] == 512
    assert logged["holdout/HoldoutSharpe"] == 1.0
    assert logged["validation/ValidationSharpe"] == 0.4
    assert logged["validation/ValidationMmcSharpe"] == 0.2
    assert logged["validation/LegacyPayoutProxy"] == 0.9


def _convergence_result(
    trial_number: int,
    *,
    objective: str,
    value: float,
    corr_sharpe: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        trial_number=trial_number,
        error=None,
        metrics={objective: value},
        corr_sharpe=corr_sharpe,
        payout_score=None,
        mmc_sharpe=None,
    )


def test_convergence_uses_exact_official_objective() -> None:
    mock_wandb = MagicMock()
    results = [
        _convergence_result(
            1,
            objective="numerai_corr_sharpe",
            value=0.1,
            corr_sharpe=0.9,
        ),
        _convergence_result(
            2,
            objective="numerai_corr_sharpe",
            value=0.3,
            corr_sharpe=0.1,
        ),
    ]

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        log_hpo_convergence(
            results,
            project="project",
            group="group",
            objective="numerai_corr_sharpe",
        )

    logged = [call.args[0] for call in mock_wandb.log.call_args_list]
    assert [row["best_numerai_corr_sharpe_so_far"] for row in logged] == [0.1, 0.3]


def test_convergence_minimizes_max_drawdown() -> None:
    mock_wandb = MagicMock()
    results = [
        _convergence_result(1, objective="max_drawdown", value=0.4, corr_sharpe=0.1),
        _convergence_result(2, objective="max_drawdown", value=0.5, corr_sharpe=0.2),
        _convergence_result(3, objective="max_drawdown", value=0.2, corr_sharpe=0.3),
    ]

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        log_hpo_convergence(
            results,
            project="project",
            group="group",
            objective="max_drawdown",
        )

    logged = [call.args[0] for call in mock_wandb.log.call_args_list]
    assert [row["best_max_drawdown_so_far"] for row in logged] == [0.4, 0.4, 0.2]
