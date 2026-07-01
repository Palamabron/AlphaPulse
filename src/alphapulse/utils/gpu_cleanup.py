"""GPU memory cleanup helpers for HPO trial subprocesses."""

from __future__ import annotations

import gc
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

_TRIAL_WORKER_MARKERS = (
    "scripts/hpo_pipeline.py",
    "multiprocessing.spawn",
    "alphapulse",
)
_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


def release_cuda_memory() -> None:
    """Release CUDA caches in the current process."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001, S110
        pass
    gc.collect()


def _read_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
    except OSError:
        return ""


def _read_ppid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def _collect_descendants(root_pid: int) -> list[int]:
    by_ppid: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        ppid = _read_ppid(pid)
        if ppid is not None:
            by_ppid.setdefault(ppid, []).append(pid)

    ordered: list[int] = []
    stack = [root_pid]
    while stack:
        current = stack.pop()
        ordered.append(current)
        stack.extend(by_ppid.get(current, []))
    return ordered


def _signal_kill(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, _KILL_SIGNAL)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def nvidia_compute_pids() -> list[int]:
    """Return PIDs with active CUDA contexts according to nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return []
    try:
        proc = subprocess.run(  # noqa: S603
            [
                nvidia_smi,
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    pids: list[int] = []
    for line in proc.stdout.strip().splitlines():
        token = line.strip().split(",")[0].strip()
        if not token or not token.isdigit():
            continue
        pids.append(int(token))
    return pids


def _is_hpo_trial_worker(pid: int, parent_pid: int) -> bool:
    if pid == parent_pid:
        return False
    cmd = _read_cmdline(pid)
    if not cmd:
        return False
    if any(marker in cmd for marker in _TRIAL_WORKER_MARKERS):
        return True
    ppid = _read_ppid(pid)
    return ppid == parent_pid


def kill_process_tree(root_pid: int, *, except_pid: int | None = None) -> list[int]:
    """SIGKILL a process and all descendants."""
    killed: list[int] = []
    for pid in reversed(_collect_descendants(root_pid)):
        if except_pid is not None and pid == except_pid:
            continue
        if _signal_kill(pid):
            killed.append(pid)
    return killed


def cleanup_stale_gpu_processes(
    *,
    parent_pid: int | None = None,
    worker_pid: int | None = None,
    kill_worker_tree: bool = False,
) -> dict[str, object]:
    """Kill orphaned HPO trial workers still holding GPU memory."""
    parent = parent_pid or os.getpid()
    killed: list[int] = []

    if worker_pid is not None and kill_worker_tree:
        killed.extend(kill_process_tree(worker_pid, except_pid=parent))

    for pid in nvidia_compute_pids():
        if pid == parent:
            continue
        if worker_pid is not None and pid in _collect_descendants(worker_pid):
            if _signal_kill(pid):
                killed.append(pid)
            continue
        if _is_hpo_trial_worker(pid, parent):
            if _signal_kill(pid):
                killed.append(pid)

    release_cuda_memory()
    if killed:
        time.sleep(0.5)

    remaining = [p for p in nvidia_compute_pids() if p != parent]
    return {"killed_pids": sorted(set(killed)), "remaining_gpu_pids": remaining}


def cleanup_after_trial_subprocess(
    worker_pid: int | None,
    *,
    parent_pid: int | None = None,
    kill_worker_tree: bool = False,
) -> dict[str, object]:
    """Run post-trial GPU cleanup in the HPO parent process."""
    return cleanup_stale_gpu_processes(
        parent_pid=parent_pid,
        worker_pid=worker_pid,
        kill_worker_tree=kill_worker_tree,
    )
