from typing import Any

from .schema import ExperimentV1


def is_multi_target_experiment(exp: ExperimentV1) -> bool:
    return bool(exp.data.auxiliary_targets)


def experiment_target_flat(exp: ExperimentV1) -> dict[str, Any]:
    aux = [
        t
        for t in dict.fromkeys(exp.data.auxiliary_targets or [])
        if t != exp.data.target_col
    ]
    if not aux:
        return {
            "target_mode": "single",
            "primary_target": exp.data.target_col,
            "auxiliary_targets": [],
            "target_blend_method": exp.data.target_blend_method,
        }
    n_subs = exp.models[0].n_subs if exp.models else 10
    return {
        "target_mode": "multi_blend",
        "primary_target": exp.data.target_col,
        "auxiliary_targets": aux,
        "target_blend_method": exp.data.target_blend_method,
        "n_subs": n_subs,
    }


def needs_internal_val_for_experiment(exp: ExperimentV1) -> bool:
    from ..pipeline.ensemble import needs_internal_val_for_ensemble

    pipeline_cfg = exp.to_pipeline_config()
    return needs_internal_val_for_ensemble(pipeline_cfg)
