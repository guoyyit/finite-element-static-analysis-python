"""Job manager: submission, bounded concurrency, recovery and result caching.

Concurrency: a fixed-size ThreadPoolExecutor runs blocking numpy solves.
Each job owns its state and cancellation event; a failure, timeout or
cancellation of one job never touches another, and no partial results are
ever exposed (terminal non-completed jobs persist *only* an error record).

Recovery: on :meth:`start`, checkpoints are scanned.  Completed jobs keep
their results and are never re-solved; pending/running jobs are re-queued
from their persisted model payload and resume from the start (linear solves
are deterministic, so an interrupted job reproduces the identical answer).

Deterministic caching: completed results are keyed by the version-bound model
hash, so an identical model+parameters submitted again returns the same job
output without recomputation.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

from fem.checkpoint import CheckpointStore
from fem.config import ServiceConfig
from fem.errors import (
    FemError,
    JobCancelledError,
    JobInvalidStateError,
    JobLimitExceededError,
    JobNotFoundError,
    JobTimeoutError,
)
from fem.job import Job
from fem.models import (
    ModelInput,
    JobErrorInfo,
    JobInfo,
    JobState,
    SolveResults,
    TERMINAL_STATES,
)
from fem.solver import AssembledSystem, solve
from fem.validation import validate_model

logger = logging.getLogger("fem.manager")


class JobManager:
    def __init__(self, config: ServiceConfig | None = None,
                 store: CheckpointStore | None = None) -> None:
        self.config = config or ServiceConfig()
        self.store = store or CheckpointStore(self.config.storage_dir)
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.RLock()
        self._started = False
        # model_hash -> job_id holding the canonical completed result
        self._result_index: dict[str, str] = {}
        # Number of submitted-but-not-terminal jobs (bounded concurrency)
        self._active_slots = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Initialize storage, recover checkpoints and start workers."""
        with self._lock:
            if self._started:
                return
            self.store.initialize()
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.worker_threads,
                thread_name_prefix="fem-worker")
            self._started = True
            self._recover()

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> None:
        with self._lock:
            executor = self._executor
            self._executor = None
            self._started = False
        if executor is not None:
            # Cancel queued-but-not-started work; running solves stop between
            # elements via their cancel/timeout checks at the next check.
            for fut in list(self._futures.values()):
                fut.cancel()
            executor.shutdown(wait=wait, cancel_futures=True)

    # -- recovery ----------------------------------------------------------
    def _recover(self) -> None:
        snapshots = self.store.load_all()
        for job_id, info in snapshots.items():
            if info.state == JobState.COMPLETED and info.results is not None:
                job = Job(info)
                job.done_event.set()
                self._jobs[job_id] = job
                self._result_index.setdefault(info.model_hash, job_id)
                continue

            # Terminal non-completed jobs (failed / cancelled / timeout) are
            # durable records: keep their state and error, never re-run, and
            # never imply a result.
            if info.state in TERMINAL_STATES:
                self._jobs[job_id] = Job(info)
                continue

            # Only genuinely unfinished work (pending / running) resumes.
            raw = self.store.load_model_raw(job_id)
            if raw is None:
                # Without its model payload a job cannot be recomputed; retain
                # the snapshot as-is so no stale "result" is implied.
                self._jobs[job_id] = Job(info)
                continue
            try:
                model = ModelInput.model_validate_json(raw)
            except ValueError:
                self._jobs[job_id] = Job(info)
                continue
            # An interrupted RUNNING job may carry an assembly checkpoint;
            # keep it so assembly resumes where it stopped.  A missing/stale
            # blob resets progress to a clean assembly run.
            keep_assembly = (info.state == JobState.RUNNING
                             and info.assembled_through > 0)
            updates = {"state": JobState.PENDING, "error": None,
                       "results": None, "updated_at": time.time()}
            if not keep_assembly:
                updates["assembled_through"] = 0
                updates["assembly_file"] = None
            restart = info.model_copy(update=updates)
            job = Job(restart)
            self._jobs[job_id] = job
            self._active_slots += 1
            self.store.save(restart)
            self._enqueue(job, model)

    # -- submission --------------------------------------------------------
    def _prepare(self, model: ModelInput):
        """Validate a model and return (normalized, hash, per-job config).

        Pure: creates no job and writes nothing, so callers can validate a
        whole batch before taking any side effect.
        """
        cfg = self._job_config(model)
        normalized = validate_model(model, cfg)
        return normalized, normalized.canonical_hash(), cfg

    @staticmethod
    def _model_payload(model: ModelInput) -> bytes:
        return json.dumps(model.model_dump(mode="json"),
                          sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")

    def submit(self, model: ModelInput) -> JobInfo:
        self.start()
        normalized, model_hash, _cfg = self._prepare(model)
        with self._lock:
            existing_id = self._result_index.get(model_hash)
            if existing_id is not None:
                return self._jobs[existing_id].snapshot()
            return self._create_locked(model, model_hash)

    def submit_batch(self, models: list[ModelInput]) -> list[JobInfo]:
        """Atomic batch submission.

        Every model is fully validated before any job is created; if any one
        fails validation or the batch would exceed the active-job limit, the
        whole call raises and no batch job is enqueued.
        """
        self.start()
        prepared = [self._prepare(m) for m in models]  # may raise, no effects
        with self._lock:
            # Reuse completed identical models, coalesce in-batch duplicates,
            # and only count genuinely new (model_hash) entries against the
            # active-job limit.  Preserve request order in the returned list.
            todo: list[str] = []
            hash_to_model: dict[str, ModelInput] = {}
            for model, (_, model_hash, _cfg) in zip(models, prepared):
                if (model_hash not in self._result_index
                        and model_hash not in hash_to_model):
                    todo.append(model_hash)
                    hash_to_model[model_hash] = model
            if self._active_slots + len(todo) > self.config.max_jobs:
                raise JobLimitExceededError(
                    f"batch would create {len(todo)} jobs and exceed active "
                    f"limit {self.config.max_jobs}",
                    details={"limit": self.config.max_jobs,
                             "requested": len(todo),
                             "active": self._active_slots})
            created: dict[str, JobInfo] = {}
            for model_hash in todo:
                created[model_hash] = self._create_locked(
                    hash_to_model[model_hash], model_hash)
            results: list[JobInfo] = []
            for model, (_, model_hash, _cfg) in zip(models, prepared):
                existing_id = self._result_index.get(model_hash)
                if existing_id is not None:
                    results.append(self._jobs[existing_id].snapshot())
                else:
                    results.append(created[model_hash])
        return results

    def _create_locked(self, model: ModelInput, model_hash: str) -> JobInfo:
        if self._active_slots >= self.config.max_jobs:
            raise JobLimitExceededError(
                f"active job limit {self.config.max_jobs} reached",
                details={"limit": self.config.max_jobs})
        job_id = secrets.token_hex(8)
        while job_id in self._jobs:
            job_id = secrets.token_hex(8)
        now = time.time()
        info = JobInfo(job_id=job_id, state=JobState.PENDING,
                       model_hash=model_hash, created_at=now, updated_at=now)
        job = Job(info)
        self._jobs[job_id] = job
        self._active_slots += 1
        # Persist snapshot first, then model payload, then enqueue: after a
        # crash the on-disk model is always recoverable.
        self.store.save(info)
        self.store.save_model(job_id, self._model_payload(model))
        self._enqueue(job, model)
        return job.snapshot()

    def _enqueue(self, job: Job, model: ModelInput) -> None:
        assert self._executor is not None
        future = self._executor.submit(self._run, job, model)
        self._futures[job.job_id] = future
        future.add_done_callback(
            lambda f, jid=job.job_id: self._after_done(jid, f))

    def _job_config(self, model: ModelInput) -> ServiceConfig:
        opts = model.options
        if opts.equilibrium_tolerance is None and opts.timeout_seconds is None:
            return self.config
        return replace(
            self.config,
            equilibrium_tolerance=(opts.equilibrium_tolerance
                                   if opts.equilibrium_tolerance is not None
                                   else self.config.equilibrium_tolerance),
            job_timeout_seconds=(opts.timeout_seconds
                                 if opts.timeout_seconds is not None
                                 else self.config.job_timeout_seconds))

    # -- worker body -------------------------------------------------------
    def _run(self, job: Job, model: ModelInput) -> None:
        job_id = job.job_id
        try:
            cfg = self._job_config(model)
            normalized = validate_model(model, cfg)
            start = time.monotonic()

            # Resume from the most recent assembly checkpoint, if any.
            partial = None
            with job.lock:
                through = job.info.assembled_through
            if 0 < through < normalized.ne:
                matrix = self.store.load_assembly(job_id)
                if matrix is not None and matrix.shape == (normalized.ndof,) * 2:
                    partial = AssembledSystem(kglob=matrix,
                                              assembled_through=through)
                else:
                    # Stale/partial blob: start assembly over rather than risk
                    # a corrupted matrix.
                    partial = None
                    snap = job.update(assembled_through=0,
                                      assembly_file=None)
                    self.store.save(snap)

            def progress_cb(p: float) -> None:
                snap = job.update(progress=float(p))
                self.store.save(snap)

            def assembly_cb(system: AssembledSystem) -> None:
                if job.is_cancel_requested():
                    raise JobCancelledError("cancelled during assembly")
                if (cfg.job_timeout_seconds
                        and time.monotonic() - start > cfg.job_timeout_seconds):
                    raise JobTimeoutError(
                        f"solve exceeded {cfg.job_timeout_seconds:.3f}s budget",
                        details={"timeout_seconds":
                                 cfg.job_timeout_seconds})
                if 0 < system.assembled_through < normalized.ne:
                    # Persist the partial matrix first, then the metadata that
                    # points at it; a crash in between is detected on resume
                    # and assembly restarts safely.
                    self.store.save_assembly(job_id, system.kglob)
                    snap = job.update(
                        progress=0.6 * system.assembled_through / normalized.ne,
                        assembled_through=system.assembled_through,
                        assembly_file=f"{job_id}.K.npy")
                    self.store.save(snap)

            with job.lock:
                if job.is_cancel_requested():
                    raise JobCancelledError("cancelled before execution")
                snap = job.update(state=JobState.RUNNING)
                self.store.save(snap)

            results = solve(
                normalized, cfg,
                model_hash=normalized.canonical_hash(),
                progress_cb=progress_cb,
                cancel_event=job.cancel_event,
                assembly=partial,
                start_monotonic=start)

            # A cancel that landed while finishing must not publish results.
            if job.is_cancel_requested():
                raise JobCancelledError("cancelled before publish")

            snap = job.update(state=JobState.COMPLETED, progress=1.0,
                              results=results, error=None,
                              assembled_through=normalized.ne)
            self.store.save(snap)
            # Result is durable: release the large partial matrix now.
            self.store.delete_assembly(job_id)
            with self._lock:
                self._result_index.setdefault(snap.model_hash, job_id)
        except (JobCancelledError,) as exc:
            self._terminal(job, JobState.CANCELLED, exc)
        except JobTimeoutError as exc:
            self._terminal(job, JobState.TIMEOUT, exc)
        except FemError as exc:
            self._terminal(job, JobState.FAILED, exc)
        except Exception as exc:  # defensive: never kill a worker thread
            logger.exception("unexpected solver failure for job %s", job_id)
            self._terminal(job, JobState.FAILED,
                           FemError(str(exc) or "unexpected error"))

    def _terminal(self, job: Job, state: JobState, exc: FemError) -> None:
        err = JobErrorInfo(code=exc.code, message=exc.message,
                           details=exc.details)
        # No partial results are persisted: clear results field explicitly.
        snap = job.update(state=state, error=err, results=None)
        try:
            self.store.save(snap)
            # Terminal non-completed jobs hold no reusable assembly; drop the
            # potentially large matrix so finished runs leave no residue.
            self.store.delete_assembly(job.job_id)
        except FemError:
            logger.exception("failed to persist terminal state %s", job.job_id)

    def _after_done(self, job_id: str, future: Future) -> None:
        with self._lock:
            self._futures.pop(job_id, None)
            if self._active_slots > 0:
                self._active_slots -= 1
        if (not future.cancelled()) and (exc := future.exception()) is not None:
            logger.error("job %s worker raised: %r", job_id, exc)

    # -- queries / control -------------------------------------------------
    def _require(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"unknown job {job_id!r}",
                                   details={"job_id": job_id})
        return job

    def get(self, job_id: str) -> JobInfo:
        with self._lock:
            return self._require(job_id).snapshot()

    def list_jobs(self) -> list[JobInfo]:
        with self._lock:
            return [j.snapshot() for j in
                    sorted(self._jobs.values(), key=lambda x: x.info.created_at)]

    def cancel(self, job_id: str) -> JobInfo:
        with self._lock:
            job = self._require(job_id)
        if job.state in TERMINAL_STATES:
            raise JobInvalidStateError(
                f"job {job_id} already {job.state.value}",
                details={"state": job.state.value})
        if job.request_cancel():
            future = self._futures.get(job_id)
            if future is not None and future.cancel():
                # Dequeued before it ever ran: record the terminal state now.
                self._terminal(job, JobState.CANCELLED,
                               JobCancelledError("cancelled while queued"))
        return job.snapshot()

    def get_result(self, job_id: str) -> SolveResults:
        with self._lock:
            job = self._require(job_id)
            snap = job.snapshot()
        if snap.state != JobState.COMPLETED or snap.results is None:
            raise JobInvalidStateError(
                f"job {job_id} has no result (state={snap.state.value})",
                details={"state": snap.state.value,
                         "error": snap.error.model_dump() if snap.error else None})
        return snap.results

    def delete_job(self, job_id: str) -> None:
        with self._lock:
            job = self._require(job_id)
            if job.state not in TERMINAL_STATES:
                raise JobInvalidStateError(
                    f"cannot delete running job {job_id}; cancel it first",
                    details={"state": job.state.value})
            self._jobs.pop(job_id, None)
            if self._result_index.get(job.info.model_hash) == job_id:
                self._result_index.pop(job.info.model_hash, None)
        self.store.delete(job_id)
