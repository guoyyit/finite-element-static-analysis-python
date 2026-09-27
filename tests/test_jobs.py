"""Job manager: concurrency, isolation, cancel, timeout, limits, determinism."""

from __future__ import annotations

import threading
import time

import pytest

from fea.errors import (
    JobNotFoundError,
    JobStateError,
    LimitExceededError,
    ValidationError,
)
from fea.jobs import JobManager
from fea.limits import ResourceLimits
from fea.models import JobStatus
from tests.conftest import make_model, wait_terminal


def test_submit_and_complete(manager, model):
    info = manager.submit(model)
    assert info.status in (JobStatus.PENDING, JobStatus.RUNNING)
    done = wait_terminal(manager, info.job_id)
    assert done.status is JobStatus.COMPLETED
    assert done.result is not None
    assert done.result.max_equilibrium_residual <= 1e-6
    assert done.error is None


def test_result_bound_to_model_hash(manager, model):
    done = wait_terminal(manager, manager.submit(model).job_id)
    from fea.jobs import model_hash_of

    assert done.model_hash == model_hash_of(model)


def test_same_model_deterministic_across_jobs(manager, model):
    j1 = wait_terminal(manager, manager.submit(model).job_id)
    j2 = wait_terminal(manager, manager.submit(model).job_id)
    assert j1.model_hash == j2.model_hash
    assert j1.result.model_dump_json() == j2.result.model_dump_json()


def test_get_unknown_job_raises(manager):
    with pytest.raises(JobNotFoundError) as ei:
        manager.get("no-such-job")
    assert ei.value.code == "job_not_found"


def test_invalid_model_rejected_before_job_created(manager):
    bad = make_model(elements=[{"id": 1, "nodes": [1, 2, 3, 4],
                                "material_id": 99, "thickness": 0.1}])
    with pytest.raises(ValidationError):
        manager.submit(bad)
    assert manager.list_jobs() == []  # no record, no partial state


def test_oversized_model_rejected_before_start(store):
    limits = ResourceLimits(max_nodes=4, max_elements=100)
    mgr = JobManager(store, limits=limits)
    mgr.start()
    try:
        oversized = make_model(nodes=[
            {"id": i, "x": float(i), "y": 0.0} for i in range(1, 6)
        ] + [{"id": 6, "x": 0.0, "y": 1.0}])
        with pytest.raises(LimitExceededError) as ei:
            mgr.submit(oversized)
        assert ei.value.code == "limit_exceeded"
        assert mgr.list_jobs() == []
    finally:
        mgr.shutdown()


def test_concurrent_jobs_isolated(manager, model):
    # One job is doomed (singular: no BCs), others are healthy.
    doomed = make_model(displacement_bcs=[])
    jobs = [manager.submit(model) for _ in range(4)]
    bad = manager.submit(doomed)
    more = [manager.submit(model) for _ in range(4)]

    bad_done = wait_terminal(manager, bad.job_id)
    assert bad_done.status is JobStatus.FAILED
    assert bad_done.error["code"] == "singular_matrix"
    assert bad_done.result is None  # no partial output

    for j in jobs + more:
        done = wait_terminal(manager, j.job_id)
        assert done.status is JobStatus.COMPLETED
        assert done.result is not None


def test_failure_leaves_no_partial_result(manager):
    doomed = make_model(displacement_bcs=[])
    done = wait_terminal(manager, manager.submit(doomed).job_id)
    assert done.status is JobStatus.FAILED
    row = manager.store.get_job(done.job_id)
    assert row["result_json"] is None
    assert row["checkpoint"] is None


def test_cancel_pending_job(store, model):
    # Single worker + a blocked first job keeps the second job pending.
    limits = ResourceLimits(max_concurrent_jobs=1)
    mgr = JobManager(store, limits=limits)
    gate = threading.Event()
    mgr._pre_solve_hook = lambda _jid: gate.wait(5)
    mgr.start()
    try:
        j1 = mgr.submit(make_model())  # occupies the only worker
        j2 = mgr.submit(make_model())  # stays pending behind j1
        cancelled = mgr.cancel(j2.job_id)
        assert cancelled.status is JobStatus.CANCELLED
        assert cancelled.result is None
        gate.set()  # release j1
        # Cancelling must not disturb the other job.
        assert wait_terminal(mgr, j1.job_id).status is JobStatus.COMPLETED
        # Cancelled job stays cancelled (worker must not revive it).
        time.sleep(0.2)
        assert mgr.get(j2.job_id).status is JobStatus.CANCELLED
        assert mgr.get(j2.job_id).result is None
    finally:
        gate.set()
        mgr.shutdown()


def test_cancel_running_job(store, model):
    limits = ResourceLimits(max_concurrent_jobs=1)
    mgr = JobManager(store, limits=limits)
    gate = threading.Event()
    mgr._pre_solve_hook = lambda _jid: gate.wait(5)
    mgr.start()
    try:
        j1 = mgr.submit(make_model())
        # Wait until the job is actually running inside the hook.
        deadline = time.monotonic() + 5
        while mgr.get(j1.job_id).status is not JobStatus.RUNNING:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        cancelled = mgr.cancel(j1.job_id)
        assert cancelled.status is JobStatus.CANCELLED
        gate.set()
        time.sleep(0.2)  # worker wakes, must not overwrite the cancel
        info = mgr.get(j1.job_id)
        assert info.status is JobStatus.CANCELLED
        assert info.result is None
        row = mgr.store.get_job(j1.job_id)
        assert row["result_json"] is None
    finally:
        gate.set()
        mgr.shutdown()


def test_cancel_terminal_job_rejected(manager, model):
    done = wait_terminal(manager, manager.submit(model).job_id)
    with pytest.raises(JobStateError) as ei:
        manager.cancel(done.job_id)
    assert ei.value.code == "job_state_error"


def test_cancel_unknown_job(manager):
    with pytest.raises(JobNotFoundError):
        manager.cancel("nope")


def test_job_timeout(store, model):
    limits = ResourceLimits(job_timeout_seconds=1e-9)
    mgr = JobManager(store, limits=limits)
    mgr.start()
    try:
        info = mgr.submit(model)
        done = wait_terminal(mgr, info.job_id)
        assert done.status is JobStatus.FAILED
        assert done.error["code"] == "job_timeout"
        assert done.result is None
    finally:
        mgr.shutdown()


def test_active_memory_bounded(manager, model):
    # After completion, in-memory tracking must be released.
    done = wait_terminal(manager, manager.submit(model).job_id)
    assert done.status is JobStatus.COMPLETED
    assert done.job_id not in manager._active
