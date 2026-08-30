from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from alphapulse.utils import gpu_cleanup
from alphapulse.utils.gpu_cleanup import (
    _collect_descendants,
    _is_hpo_trial_worker,
    cleanup_after_trial_subprocess,
    release_cuda_memory,
    snapshot_process_tree,
)


def test_release_cuda_memory_does_not_raise() -> None:
    release_cuda_memory()


def test_collect_descendants_includes_parent() -> None:
    descendants = _collect_descendants(os.getpid())
    assert os.getpid() in descendants


def test_collect_descendants_includes_child_process() -> None:
    child = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    try:
        descendants = _collect_descendants(os.getpid())
        assert child.pid in descendants
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_is_hpo_trial_worker_false_for_parent() -> None:
    assert not _is_hpo_trial_worker(os.getpid(), os.getpid())


def test_cleanup_after_trial_subprocess_returns_shape() -> None:
    result = cleanup_after_trial_subprocess(None)
    assert "killed_pids" in result
    assert "remaining_gpu_pids" in result


def test_preflight_never_kills_unscoped_gpu_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[int] = []

    def record_kill(pid: int) -> bool:
        killed.append(pid)
        return True

    monkeypatch.setattr(gpu_cleanup, "nvidia_compute_pids", lambda: [777])
    monkeypatch.setattr(
        gpu_cleanup,
        "_read_cmdline",
        lambda pid: "python multiprocessing.spawn alphapulse",
    )
    monkeypatch.setattr(gpu_cleanup, "_read_ppid", lambda pid: 555)
    monkeypatch.setattr(
        gpu_cleanup,
        "_signal_kill",
        record_kill,
    )
    monkeypatch.setattr(gpu_cleanup, "release_cuda_memory", lambda: None)

    result = gpu_cleanup.cleanup_stale_gpu_processes(parent_pid=100)

    assert killed == []
    assert result["remaining_gpu_pids"] == [777]


def test_snapshot_cleanup_kills_reparented_worker_child(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    worker_code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "time.sleep(0.5)"
    )
    worker = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", worker_code, str(child_pid_path)]
    )
    snapshot: dict[int, str] = {}
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and worker.poll() is None:
            snapshot.update(snapshot_process_tree(worker.pid))
            if child_pid_path.exists():
                child_pid = int(child_pid_path.read_text())
            time.sleep(0.05)
        worker.wait(timeout=5)
        assert child_pid is not None
        assert child_pid in snapshot

        cleanup_after_trial_subprocess(
            worker.pid,
            kill_worker_tree=True,
            worker_snapshot=snapshot,
        )

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and gpu_cleanup._read_cmdline(child_pid):
            time.sleep(0.05)
        assert gpu_cleanup._read_cmdline(child_pid) == ""
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)
        if child_pid is not None:
            gpu_cleanup._signal_kill(child_pid)
