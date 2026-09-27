"""SQLite-backed persistence: job records, checkpoints and results.

Checkpoint protocol used by the job manager:

* stage ``validated``  - input passed validation, not yet solved
* stage ``solved``     - result computed and durably stored, job not yet
                         marked completed

On recovery, a ``solved`` checkpoint lets the manager finish the job without
re-solving; anything earlier resumes from the solve step. Completed jobs are
never re-solved and their results are never dropped.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    model_json  TEXT NOT NULL,
    model_hash  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    error_json  TEXT,
    result_json TEXT,
    checkpoint  TEXT
);
"""


class JobStore:
    """Durable store shared by the job manager; survives process restarts."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def initialize(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        assert self._conn is not None, "store not initialized"
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    # -- writes -----------------------------------------------------------

    def create_job(
        self,
        job_id: str,
        status: str,
        model_json: str,
        model_hash: str,
        created_at: str,
    ) -> None:
        self._execute(
            "INSERT INTO jobs (job_id, status, model_json, model_hash,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (job_id, status, model_json, model_hash, created_at, created_at),
        )

    def update_status(
        self,
        job_id: str,
        status: str,
        updated_at: str,
        *,
        error_json: str | None = None,
        result_json: str | None = None,
        clear_checkpoint: bool = False,
    ) -> None:
        sets = ["status = ?", "updated_at = ?"]
        params: list = [status, updated_at]
        if error_json is not None:
            sets.append("error_json = ?")
            params.append(error_json)
        if result_json is not None:
            sets.append("result_json = ?")
            params.append(result_json)
        if clear_checkpoint:
            sets.append("checkpoint = NULL")
        params.append(job_id)
        self._execute(
            f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?",
            tuple(params),
        )

    def save_checkpoint(
        self,
        job_id: str,
        stage: str,
        updated_at: str,
        *,
        result_json: str | None = None,
    ) -> None:
        checkpoint = json.dumps({"stage": stage})
        if result_json is not None:
            self._execute(
                "UPDATE jobs SET checkpoint = ?, result_json = ?,"
                " updated_at = ? WHERE job_id = ?",
                (checkpoint, result_json, updated_at, job_id),
            )
        else:
            self._execute(
                "UPDATE jobs SET checkpoint = ?, updated_at = ?"
                " WHERE job_id = ?",
                (checkpoint, updated_at, job_id),
            )

    def transition(
        self,
        job_id: str,
        from_statuses: tuple[str, ...],
        to_status: str,
        updated_at: str,
        *,
        error_json: str | None = None,
        result_json: str | None = None,
        clear_result: bool = False,
        clear_checkpoint: bool = False,
    ) -> bool:
        """Atomic compare-and-set on job status.

        Returns True only if the job was in one of ``from_statuses`` and the
        transition was applied. Concurrent terminal writes therefore cannot
        flip an already-terminal job.
        """
        sets = ["status = ?", "updated_at = ?"]
        params: list = [to_status, updated_at]
        if error_json is not None:
            sets.append("error_json = ?")
            params.append(error_json)
        if result_json is not None:
            sets.append("result_json = ?")
            params.append(result_json)
        if clear_result:
            sets.append("result_json = NULL")
        if clear_checkpoint:
            sets.append("checkpoint = NULL")
        placeholders = ", ".join("?" for _ in from_statuses)
        params.extend(from_statuses)
        params.append(job_id)
        cur = self._execute(
            f"UPDATE jobs SET {', '.join(sets)}"
            f" WHERE status IN ({placeholders}) AND job_id = ?",
            tuple(params),
        )
        return cur.rowcount > 0

    # -- reads ------------------------------------------------------------

    def get_job(self, job_id: str) -> dict | None:
        assert self._conn is not None, "store not initialized"
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_jobs(self) -> list[dict]:
        assert self._conn is not None, "store not initialized"
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at, job_id"
            ).fetchall()
        return [dict(r) for r in rows]

    def unfinished_jobs(self) -> list[dict]:
        """Jobs that were pending or running when the process stopped."""
        assert self._conn is not None, "store not initialized"
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status IN ('pending', 'running')"
                " ORDER BY created_at, job_id"
            ).fetchall()
        return [dict(r) for r in rows]
