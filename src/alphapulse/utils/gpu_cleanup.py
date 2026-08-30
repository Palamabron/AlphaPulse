"""GPU memory cleanup helpers for HPO trial subprocesses."""

from __future__ import annotations

import csv
import gc
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

_TRIAL_WORKER_MARKERS = (
    "scripts/hpo_pipeline.py",
    "multiprocessing.spawn",
    "alphapulse",
)
_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _windows_process_rows() -> list[dict[str, str]]:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        return []
    command = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | "
        "ConvertTo-Csv -NoTypeInformation"
    )
    try:
        proc = subprocess.run(  # noqa: S603
            [
                powershell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    return [
        {str(key): value or "" for key, value in row.items() if key is not None}
        for row in csv.DictReader(proc.stdout.splitlines())
    ]


def _windows_process_row(pid: int) -> dict[str, str] | None:
    expected_pid = str(pid)
    return next(
        (
            row
            for row in _windows_process_rows()
            if row.get("ProcessId") == expected_pid
        ),
        None,
    )


def _psutil_process(pid: int) -> Any:
    try:
        import psutil  # type: ignore[import-untyped]

        return psutil.Process(pid)
    except Exception:  # noqa: BLE001
        return None


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
    if os.name == "nt":
        process = _psutil_process(pid)
        if process is not None:
            try:
                return " ".join(process.cmdline())
            except Exception:  # noqa: BLE001, S110
                pass
        row = _windows_process_row(pid)
        return row.get("CommandLine", "") if row is not None else ""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
    except OSError:
        return ""


def _read_ppid(pid: int) -> int | None:
    if os.name == "nt":
        process = _psutil_process(pid)
        if process is not None:
            try:
                return int(process.ppid())
            except Exception:  # noqa: BLE001, S110
                pass
        row = _windows_process_row(pid)
        if row is None:
            return None
        try:
            return int(row["ParentProcessId"])
        except (KeyError, ValueError):
            return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def _collect_descendants(root_pid: int) -> list[int]:
    by_ppid: dict[int, list[int]] = {}
    if os.name == "nt":
        process = _psutil_process(root_pid)
        if process is not None:
            try:
                return [
                    root_pid,
                    *(child.pid for child in process.children(recursive=True)),
                ]
            except Exception:  # noqa: BLE001, S110
                pass
        rows = _windows_process_rows()
        for row in rows:
            try:
                process_id = int(row["ProcessId"])
                parent_id = int(row["ParentProcessId"])
            except (KeyError, ValueError):
                continue
            by_ppid.setdefault(parent_id, []).append(process_id)
        visible_pids = {row.get("ProcessId") for row in rows}
        if not rows or str(root_pid) not in visible_pids:
            process = _psutil_process(root_pid)
            if process is None:
                return [root_pid]
            try:
                return [
                    root_pid,
                    *(child.pid for child in process.children(recursive=True)),
                ]
            except Exception:  # noqa: BLE001
                return [root_pid]
    else:
        try:
            proc_entries = list(Path("/proc").iterdir())
        except OSError:
            proc_entries = []
        for entry in proc_entries:
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
    except OSError:
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
    if pid not in set(_collect_descendants(parent_pid)):
        return False
    cmd = _read_cmdline(pid)
    if not cmd:
        return False
    return any(marker in cmd for marker in _TRIAL_WORKER_MARKERS)


def snapshot_process_tree(root_pid: int) -> dict[int, str]:
    """Capture stable process identities before a worker can be reparented."""
    snapshot: dict[int, str] = {}
    for pid in _collect_descendants(root_pid):
        cmdline = _read_cmdline(pid)
        if cmdline:
            snapshot[pid] = cmdline
    return snapshot


def _kill_process_snapshot(
    snapshot: dict[int, str],
    *,
    except_pid: int | None = None,
) -> list[int]:
    killed: list[int] = []
    for pid, expected_cmdline in reversed(snapshot.items()):
        if pid == except_pid:
            continue
        if _read_cmdline(pid) != expected_cmdline:
            continue
        if _signal_kill(pid):
            killed.append(pid)
    return killed


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
    worker_snapshot: dict[int, str] | None = None,
) -> dict[str, object]:
    """Kill orphaned HPO trial workers still holding GPU memory."""
    parent = parent_pid or os.getpid()
    killed: list[int] = []

    tracked_processes = dict(worker_snapshot or {})
    if worker_pid is not None:
        tracked_processes.update(snapshot_process_tree(worker_pid))
    worker_descendants = set(tracked_processes)
    if worker_pid is not None and kill_worker_tree:
        if tracked_processes:
            killed.extend(_kill_process_snapshot(tracked_processes, except_pid=parent))
        else:
            killed.extend(kill_process_tree(worker_pid, except_pid=parent))
    elif worker_pid is not None:
        for pid in nvidia_compute_pids():
            expected_cmdline = tracked_processes.get(pid)
            if (
                pid != parent
                and pid in worker_descendants
                and expected_cmdline is not None
                and _read_cmdline(pid) == expected_cmdline
                and _signal_kill(pid)
            ):
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
    worker_snapshot: dict[int, str] | None = None,
) -> dict[str, object]:
    """Run post-trial GPU cleanup in the HPO parent process."""
    return cleanup_stale_gpu_processes(
        parent_pid=parent_pid,
        worker_pid=worker_pid,
        kill_worker_tree=kill_worker_tree,
        worker_snapshot=worker_snapshot,
    )
