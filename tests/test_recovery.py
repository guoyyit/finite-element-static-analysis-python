"""Restart recovery: unfinished jobs resume from checkpoints, completed
jobs stay queryable and are never re-solved."""

from __future__ import annotations

import json

import pytest

import fea.jobs as jobs_module
from fea.jobs import JobManager, canonical_model_json, model_hash_of
from fea.limits import ResourceLimits
from fea.models import JobStatus
from fea.solver import solve_model
from fea.storage import JobStore
from tests.conftest import make_model, wait_terminal


def _seed_job(store: JobStore, job_id: str, status: str, model,
              checkpoint_stage: str | None = None,
              result_json: str | None = None) -> None:
    store.create_job(
        job_id, status, canonical_model_json(model), model_hash_of(model),
        "2026-09-28T00:00:00+00:00",
    )
    if checkpoint_stage is not None:
        store.save_checkpoint(
            job_id, checkpoint_stage, "2026-09-28T00:00:01+00:00",
            result_json=result_json,
        )


def _restart(db_path) -> JobManager:
    """Open a fresh store+manager on the same DB (simulates a restart)."""
    store = JobStore(db_path)
    store.initialize()
    mgr = JobManager(store, limits=ResourceLimits(max_concurrent_jobs=2))
    mgr.start()
    return mgr


def test_solved_checkpoint_completes_without_resolve(tmp_path, model):
    db = tmp_path / "jobs.db"
    store = JobStore(db)
    store.initialize()
    # Crash happened after the solve checkpoint but before completion.
    result_json = solve_model(model).model_dump_json()
    _seed_job(store, "job-solved", "running", model,
              checkpoint_stage="solved", result_json=result_json)
    store.close()

    mgr = _restart(db)
    try:
        # If recovery re-solved, this would raise.
        def boom(_model):
            raise AssertionError("must not re-solve a solved checkpoint")

        jobs_module.solve_model, original = boom, jobs_module.solve_model
        try:
            info = mgr.get("job-solved")
        finally:
            jobs_module.solve_model = original
        assert info.status is JobStatus.COMPLETED
        assert info.result is not None
        assert info.result.model_dump_json() == result_json
    finally:
        mgr.shutdown()
        mgr.store.close()


def test_validated_checkpoint_resolves_on_recovery(tmp_path, model):
    db = tmp_path / "jobs.db"
    store = JobStore(db)
    store.initialize()
    _seed_job(store, "job-validated", "running", model,
              checkpoint_stage="validated")
    store.close()

    mgr = _restart(db)
    try:
        done = wait_terminal(mgr, "job-validated")
        assert done.status is JobStatus.COMPLETED
        assert done.result is not None
        # Same deterministic result as a fresh solve.
        assert done.result.model_dump_json() == \
            solve_model(model).model_dump_json()
    finally:
        mgr.shutdown()
        mgr.store.close()


def test_pending_job_resumes_on_recovery(tmp_path, model):
    db = tmp_path / "jobs.db"
    store = JobStore(db)
    store.initialize()
    _seed_job(store, "job-pending", "pending", model)
    store.close()

    mgr = _restart(db)
    try:
        done = wait_terminal(mgr, "job-pending")
        assert done.status is JobStatus.COMPLETED
    finally:
        mgr.shutdown()
        mgr.store.close()


def test_completed_job_preserved_across_restart(tmp_path, model):
    db = tmp_path / "jobs.db"
    store = JobStore(db)
    store.initialize()
    mgr1 = JobManager(store, limits=ResourceLimits())
    mgr1.start()
    done = wait_terminal(mgr1, mgr1.submit(model).job_id)
    result_before = done.result.model_dump_json()
    mgr1.shutdown()
    store.close()

    mgr2 = _restart(db)
    try:
        info = mgr2.get(done.job_id)
        assert info.status is JobStatus.COMPLETED
        assert info.result.model_dump_json() == result_before
        # Give the recovery scan a chance to misbehave; it must not.
        import time

        time.sleep(0.2)
        assert mgr2.get(done.job_id).status is JobStatus.COMPLETED
    finally:
        mgr2.shutdown()
        mgr2.store.close()


def test_failed_and_cancelled_jobs_untouched_by_recovery(tmp_path, model):
    db = tmp_path / "jobs.db"
    store = JobStore(db)
    store.initialize()
    _seed_job(store, "job-failed", "failed", model)
    _seed_job(store, "job-cancelled", "cancelled", model)
    store.close()

    mgr = _restart(db)
    try:
        import time

        time.sleep(0.2)
        assert mgr.get("job-failed").status is JobStatus.FAILED
        assert mgr.get("job-cancelled").status is JobStatus.CANCELLED
        assert mgr.get("job-failed").result is None
    finally:
        mgr.shutdown()
        mgr.store.close()


def test_result_deterministic_across_restart(tmp_path, model):
    """Same model solved before and after a restart: identical floats."""
    db = tmp_path / "jobs.db"
    store = JobStore(db)
    store.initialize()
    mgr1 = JobManager(store, limits=ResourceLimits())
    mgr1.start()
    j1 = wait_terminal(mgr1, mgr1.submit(model).job_id)
    mgr1.shutdown()
    store.close()

    mgr2 = _restart(db)
    try:
        j2 = wait_terminal(mgr2, mgr2.submit(model).job_id)
        assert j1.model_hash == j2.model_hash
        assert j1.result.model_dump_json() == j2.result.model_dump_json()
    finally:
        mgr2.shutdown()
        mgr2.store.close()
