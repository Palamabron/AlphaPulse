from __future__ import annotations

import os

from alphapulse.utils.gpu_cleanup import (
    _collect_descendants,
    _is_hpo_trial_worker,
    cleanup_after_trial_subprocess,
    release_cuda_memory,
)


def test_release_cuda_memory_does_not_raise() -> None:
    release_cuda_memory()


def test_collect_descendants_includes_parent() -> None:
    descendants = _collect_descendants(os.getpid())
    assert os.getpid() in descendants


def test_is_hpo_trial_worker_false_for_parent() -> None:
    assert not _is_hpo_trial_worker(os.getpid(), os.getpid())


def test_cleanup_after_trial_subprocess_returns_shape() -> None:
    result = cleanup_after_trial_subprocess(None)
    assert "killed_pids" in result
    assert "remaining_gpu_pids" in result
