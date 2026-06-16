from unittest.mock import MagicMock, patch

from alphapulse.hpo.objective import TrialResult
from alphapulse.logging_.wandb_utils import log_hpo_trial_metrics


def test_log_hpo_trial_metrics_logs_feature_routing_fields() -> None:
    mock_wandb = MagicMock()
    result = TrialResult(
        trial_number=1,
        sharpe=1.0,
        metrics={"corr_sharpe": 1.0, "payout_score": 0.9},
        model_type="XGBoost",
        elapsed_seconds=12.0,
        params={
            "active_groups": ["small", "strength"],
            "routed_feature_count": 512,
        },
        corr_sharpe=1.0,
        payout_score=0.9,
    )
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        log_hpo_trial_metrics(result, objective=0.9, model_types="XGBoost")

    logged = mock_wandb.log.call_args.args[0]
    assert logged["active_groups"] == "small+strength"
    assert logged["active_groups_count"] == 2
    assert logged["routed_feature_count"] == 512
