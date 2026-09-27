"""Concurrent job manager: scheduling, cancellation, timeout, recovery.

Isolation & determinism guarantees:

* each job runs in its own worker thread with its own cancel event; a
  failure / cancel / timeout in one job never touches another job's state;
* results are written to the durable store only via the checkpoint
  protocol (``validated`` -> ``solved`` -> ``completed``); a job that is
  cancelled or fails has any partial result erased, so nothing partial can
  be mistaken for a valid result;
* every result is bound to ``model_hash`` (SHA-256 of the canonical model
  JSON); identical models re-solve to bit-identical results;
* in-memory bookkeeping is limited to non-terminal jobs, so memory stays
  bounded regardless of how many jobs were completed historically.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .errors import (
    FEAError,
    JobCancelledError,
    JobNotFoundError,
    JobStateError,
    JobTimeoutError,
    LimitExceededError,
)
from .limits import ResourceLimits
from .models import JobInfo, JobStatus, ModelInput, SolveResult
from .solver import solve_model
from .storage import JobStore
from .validation import validate_model


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_model_json(model: ModelInput) -> str:
    """Canonical serialization: stable across dict ordering and whitespace."""
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def model_hash_of(model: ModelInput) -> str:
    return hashlib.sha256(canonical_model_json(model).encode()).hexdigest()


class _ActiveJob:
    """In-memory handle for a non-terminal job (bounded by design)."""

    __slots__ = ("cancel_event", "deadline")

    def __init__(self, deadline: float) -> None:
        self.cancel_event = threading.Event()
        self.deadline = deadline


_NON_TERMINAL = (JobStatus.PENDING.value, JobStatus.RUNNING.value)


class _Superseded(Exception):
    """The job was finalized concurrently; the worker must stand down."""


class JobManager:
    def __init__(
        self,
        store: JobStore,
        *,
        limits: ResourceLimits | None = None,
    ) -> None:
        self.store = store
        self.limits = limits or ResourceLimits()
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._active: dict[str, _ActiveJob] = {}
        self._started = False
        #: Testability seam: invoked inside the worker right before solving.
        self._pre_solve_hook = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._executor = ThreadPoolExecutor(
                max_workers=self.limits.max_concurrent_jobs,
                thread_name_prefix="fea-job",
            )
            self._started = True
        self._recover_unfinished()

    def shutdown(self, *, wait: bool = True, timeout: float = 10.0) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
            self._started = False
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=False)

    def _recover_unfinished(self) -> None:
        """Resume jobs left non-terminal by a previous process.

        A ``solved`` checkpoint means the result is already durable: finish
        the job without re-solving. Anything earlier is re-run from the
        solve stage. Completed jobs are never touched.
        """
        for row in self.store.unfinished_jobs():
            job_id = row["job_id"]
            checkpoint = (
                json.loads(row["checkpoint"]) if row["checkpoint"] else None
            )
            if (
                checkpoint
                and checkpoint.get("stage") == "solved"
                and row["result_json"]
            ):
                self.store.update_status(
                    job_id,
                    JobStatus.COMPLETED.value,
                    _utcnow(),
                    clear_checkpoint=True,
                )
                continue
            model = ModelInput.model_validate_json(row["model_json"])
            self._active[job_id] = _ActiveJob(
                time.monotonic() + self.limits.job_timeout_seconds
            )
            self.store.update_status(
                job_id, JobStatus.PENDING.value, _utcnow()
            )
            self._submit_to_executor(job_id, model)

    # -- submission --------------------------------------------------------

    def submit(self, model: ModelInput) -> JobInfo:
        """Validate, register and enqueue a job. Rejects over-limit input."""
        self.limits.check_model_size(len(model.nodes), len(model.elements))
        validate_model(model)  # raises ValidationError before any record

        with self._lock:
            if not self._started or self._executor is None:
                raise JobStateError("job manager is not running")
            if len(self._active) >= (
                self.limits.max_concurrent_jobs + self.limits.max_queued_jobs
            ):
                raise LimitExceededError(
                    "too many active jobs; try again later",
                    details={"active": len(self._active)},
                )
            job_id = uuid.uuid4().hex
            now = _utcnow()
            self.store.create_job(
                job_id,
                JobStatus.PENDING.value,
                canonical_model_json(model),
                model_hash_of(model),
                now,
            )
            self._active[job_id] = _ActiveJob(
                time.monotonic() + self.limits.job_timeout_seconds
            )
            self._submit_to_executor(job_id, model)
        return self.get(job_id)

    def _submit_to_executor(self, job_id: str, model: ModelInput) -> None:
        assert self._executor is not None
        self._executor.submit(self._run_job, job_id, model)

    # -- worker ------------------------------------------------------------

    def _check_cancelled_or_timed_out(self, job_id: str) -> None:
        handle = self._active.get(job_id)
        if handle is None:
            raise JobCancelledError(f"job {job_id} was cancelled")
        if handle.cancel_event.is_set():
            raise JobCancelledError(f"job {job_id} was cancelled")
        if time.monotonic() > handle.deadline:
            raise JobTimeoutError(
                f"job {job_id} exceeded timeout of "
                f"{self.limits.job_timeout_seconds}s"
            )

    def _run_job(self, job_id: str, model: ModelInput) -> None:
        try:
            self._check_cancelled_or_timed_out(job_id)
            if not self.store.transition(
                job_id, _NON_TERMINAL, JobStatus.RUNNING.value, _utcnow()
            ):
                raise _Superseded  # cancelled before we even started

            validate_model(model)
            self.store.save_checkpoint(job_id, "validated", _utcnow())
            self._check_cancelled_or_timed_out(job_id)

            if self._pre_solve_hook is not None:
                self._pre_solve_hook(job_id)
            self._check_cancelled_or_timed_out(job_id)

            result = solve_model(model)
            result_json = result.model_dump_json()
            self.store.save_checkpoint(
                job_id, "solved", _utcnow(), result_json=result_json
            )
            self._check_cancelled_or_timed_out(job_id)

            # Conditional write: if a cancel/timeout landed concurrently,
            # the transition fails and the job keeps its terminal state.
            if not self.store.transition(
                job_id,
                _NON_TERMINAL,
                JobStatus.COMPLETED.value,
                _utcnow(),
                result_json=result_json,
                clear_checkpoint=True,
            ):
                raise _Superseded
        except _Superseded:
            pass  # terminal state already written by someone else
        except JobCancelledError:
            self._finalize_aborted(job_id, JobStatus.CANCELLED)
        except JobTimeoutError as exc:
            self._finalize_aborted(
                job_id, JobStatus.FAILED, error=exc.to_dict()
            )
        except FEAError as exc:
            self._finalize_aborted(
                job_id, JobStatus.FAILED, error=exc.to_dict()
            )
        except Exception as exc:  # defensive: never leak partial state
            err = FEAError(f"unexpected solver error: {exc}")
            self._finalize_aborted(
                job_id, JobStatus.FAILED, error=err.to_dict()
            )
        finally:
            with self._lock:
                self._active.pop(job_id, None)

    def _finalize_aborted(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: dict | None = None,
    ) -> None:
        """Terminal non-success state; erase any partial result atomically.

        Conditional on the job still being non-terminal, so a concurrent
        completion is never clobbered.
        """
        self.store.transition(
            job_id,
            _NON_TERMINAL,
            status.value,
            _utcnow(),
            error_json=json.dumps(error) if error else None,
            clear_result=True,
            clear_checkpoint=True,
        )

    # -- queries -----------------------------------------------------------

    def get(self, job_id: str) -> JobInfo:
        row = self.store.get_job(job_id)
        if row is None:
            raise JobNotFoundError(
                f"job {job_id} not found", details={"job_id": job_id}
            )
        return self._row_to_info(row)

    def list_jobs(self) -> list[JobInfo]:
        return [self._row_to_info(r) for r in self.store.list_jobs()]

    @staticmethod
    def _row_to_info(row: dict) -> JobInfo:
        status = JobStatus(row["status"])
        result = None
        # Only completed jobs may expose a result - never partial output.
        if status is JobStatus.COMPLETED and row["result_json"]:
            result = SolveResult.model_validate_json(row["result_json"])
        error = json.loads(row["error_json"]) if row["error_json"] else None
        return JobInfo(
            job_id=row["job_id"],
            status=status,
            model_hash=row["model_hash"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error=error,
            result=result,
        )

    def cancel(self, job_id: str) -> JobInfo:
        row = self.store.get_job(job_id)
        if row is None:
            raise JobNotFoundError(
                f"job {job_id} not found", details={"job_id": job_id}
            )
        with self._lock:
            handle = self._active.get(job_id)
            if handle is not None:
                handle.cancel_event.set()
        # Atomic transition: loses the race against a concurrent terminal
        # write (e.g. the job completed just now) -> report the real state.
        if not self.store.transition(
            job_id,
            _NON_TERMINAL,
            JobStatus.CANCELLED.value,
            _utcnow(),
            clear_result=True,
            clear_checkpoint=True,
        ):
            current = JobStatus(self.store.get_job(job_id)["status"])
            raise JobStateError(
                f"job {job_id} is already {current.value}",
                details={"job_id": job_id, "status": current.value},
            )
        return self.get(job_id)
