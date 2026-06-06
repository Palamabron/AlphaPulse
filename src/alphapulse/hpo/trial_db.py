import json
import sqlite3
from pathlib import Path
from typing import Any

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trials (
    trial_number INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    flat_config TEXT NOT NULL,
    metrics TEXT,
    error TEXT,
    elapsed_seconds REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


class TrialDB:
    """SQLite-backed persistent store for HPO trial state.

    Enables crash-safe HPO sweeps: each trial is recorded before it starts
    (status='running') and updated to 'completed' or 'failed' afterward.
    On resume, already-completed trial numbers are skipped.

    Args:
        path: Path to the SQLite database file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def insert_trial(self, trial_number: int, flat_config: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO trials (trial_number, status, flat_config) "
            "VALUES (?, 'running', ?)",
            (trial_number, json.dumps(flat_config)),
        )
        self._conn.commit()

    def update_trial(
        self,
        trial_number: int,
        *,
        status: str,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE trials SET status=?, metrics=?, error=?, elapsed_seconds=? "
            "WHERE trial_number=?",
            (
                status,
                json.dumps(metrics) if metrics is not None else None,
                error,
                elapsed_seconds,
                trial_number,
            ),
        )
        self._conn.commit()

    def completed_trials(self) -> set[int]:
        """Return the set of trial numbers that already completed successfully."""
        cur = self._conn.execute(
            "SELECT trial_number FROM trials WHERE status='completed'"
        )
        return {row[0] for row in cur.fetchall()}

    def load_all_trials(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT trial_number, status, flat_config, metrics, error, "
            "elapsed_seconds, created_at FROM trials ORDER BY trial_number"
        )
        rows = []
        for row in cur.fetchall():
            rows.append(
                {
                    "trial_number": row[0],
                    "status": row[1],
                    "flat_config": json.loads(row[2]),
                    "metrics": json.loads(row[3]) if row[3] else None,
                    "error": row[4],
                    "elapsed_seconds": row[5],
                    "created_at": row[6],
                }
            )
        return rows

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TrialDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
