"""Job manager: concurrency isolation, cancellation, limits, no partial state."""

from __future__ import annotations

import time

import pytest

from fem.errors import (
    JobInvalidStateError,
    JobLimitExceededError,
    JobNotFoundError,
)
from fem.models import (
    JobState,
    MaterialInput,
)

from .conftest import make_model


def _wait_terminal(mgr, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = mgr.get(job_id)
        if info.state in (JobState.COMPLETED, JobState.FAILED,
                          JobState.CANCELLED, JobState.TIMEOUT):
            return info
        time.sleep(0.005)
    raise AssertionError(f"job {job_id} did not finish")


def test_submit_completes_with_result(manager_factory, single_element_model):
    mgr = manager_factory()
    info = mgr.submit(single_element_model)
    final = _wait_terminal(mgr, info.job_id)
    assert final.state == JobState.COMPLETED
    assert final.results is not None
    assert mgr.get_result(info.job_id).model_hash == final.model_hash


def test_invalid_submit_has_no_side_effects(manager_factory):
    mgr = manager_factory()
    bad = make_model()
    bad.materials = []
    with pytest.raises(Exception):
        mgr.submit(bad)
    assert mgr.list_jobs() == []


def test_failed_job_exposes_distinct_error_and_no_result(manager_factory):
    mgr = manager_factory()
    # Over-tight tolerance -> runtime solver failure, job is created first.
    m = make_model()
    m.options.equilibrium_tolerance = 1e-300
    info = mgr.submit(m)
    final = _wait_terminal(mgr, info.job_id)
    assert final.state == JobState.FAILED
    assert final.results is None
    assert final.error is not None
    assert final.error.code == "equilibrium_not_satisfied"
    with pytest.raises(JobInvalidStateError):
        mgr.get_result(info.job_id)


def test_concurrent_jobs_are_isolated(manager_factory):
    mgr = manager_factory(worker_threads=4)
    submitted = []
    for v in [1.0, 2.0, 3.0, 4.0]:
        submitted.append(mgr.submit(make_model(load=v)))
    finals = {f.job_id: _wait_terminal(mgr, f.job_id) for f in submitted}
    assert all(f.state == JobState.COMPLETED for f in finals.values())
    # Independent results scale with load; no cross-job contamination.
    ux = {f.results.displacements[2].ux for f in finals.values()}
    assert len(ux) == 4


def test_one_job_failure_does_not_affect_others(manager_factory):
    mgr = manager_factory(worker_threads=2)
    good = mgr.submit(make_model(load=1.0))
    bad_model = make_model(load=1.0)
    bad_model.options.equilibrium_tolerance = 1e-300
    # Rebuild options properly through the model API.
    bad_model = make_model(load=1.0)
    from fem.models import SolveOptions
    bad_model.options = SolveOptions(equilibrium_tolerance=1e-300,
                                     timeout_seconds=None)
    bad = mgr.submit(bad_model)
    good_final = _wait_terminal(mgr, good.job_id)
    bad_final = _wait_terminal(mgr, bad.job_id)
    assert good_final.state == JobState.COMPLETED
    assert bad_final.state == JobState.FAILED
    # The good job's result is still intact and queryable.
    assert mgr.get_result(good.job_id) is not None


def test_cancel_pending_or_running_job(manager_factory):
    # Single worker forces the second job to queue; cancelling it must not
    # impact the first, and no result is exposed.
    mgr = manager_factory(worker_threads=1)
    first = mgr.submit(make_model(load=1.0))
    queued = mgr.submit(make_model(load=2.0))
    mgr.cancel(queued.job_id)
    f1 = _wait_terminal(mgr, first.job_id)
    f2 = _wait_terminal(mgr, queued.job_id)
    assert f1.state == JobState.COMPLETED
    assert f2.state == JobState.CANCELLED
    assert f2.results is None
    with pytest.raises(JobInvalidStateError):
        mgr.get_result(queued.job_id)


def test_cancel_completed_job_is_invalid(manager_factory, single_element_model):
    mgr = manager_factory()
    info = mgr.submit(single_element_model)
    _wait_terminal(mgr, info.job_id)
    with pytest.raises(JobInvalidStateError):
        mgr.cancel(info.job_id)


def test_active_job_limit_enforced_before_execution(manager_factory):
    mgr = manager_factory(max_jobs=1, worker_threads=1)
    first = mgr.submit(make_model(load=1.0))
    # Second distinct model while first active must be rejected up front.
    second_model = make_model(load=5.0)
    with pytest.raises(JobLimitExceededError):
        mgr.submit(second_model)
    _wait_terminal(mgr, first.job_id)
    # After completion its slot frees.
    again = mgr.submit(make_model(load=7.0))
    assert _wait_terminal(mgr, again.job_id).state == JobState.COMPLETED


def test_unknown_job_raises_not_found(manager_factory):
    mgr = manager_factory()
    with pytest.raises(JobNotFoundError):
        mgr.get("does-not-exist")


def test_identical_model_deduplicates_to_same_job(manager_factory):
    mgr = manager_factory()
    i1 = mgr.submit(make_model(load=1.0))
    _wait_terminal(mgr, i1.job_id)
    # Only a *completed* identical model is deduplicated to its result.
    i2 = mgr.submit(make_model(load=1.0))
    assert i2.job_id == i1.job_id
    i3 = mgr.submit(make_model(load=1.0))
    assert i3.job_id == i1.job_id


def test_delete_terminal_job_removes_it(manager_factory, single_element_model):
    mgr = manager_factory()
    info = mgr.submit(single_element_model)
    _wait_terminal(mgr, info.job_id)
    mgr.delete_job(info.job_id)
    with pytest.raises(JobNotFoundError):
        mgr.get(info.job_id)


def test_batch_atomic_when_one_invalid(manager_factory):
    mgr = manager_factory()
    good = make_model(load=1.0)
    bad = make_model(load=2.0)
    bad.materials = [MaterialInput(name="other", youngs_modulus=1.0,
                                   poisson_ratio=0.0, thickness=1.0)]
    # bad references material "s" which no longer exists
    with pytest.raises(Exception):
        mgr.submit_batch([good, bad])
    assert mgr.list_jobs() == []


def test_batch_all_or_none_on_limit(manager_factory):
    mgr = manager_factory(max_jobs=1, worker_threads=1)
    first = mgr.submit(make_model(load=1.0))
    with pytest.raises(JobLimitExceededError):
        mgr.submit_batch([make_model(load=8.0), make_model(load=9.0)])
    _wait_terminal(mgr, first.job_id)


def test_cancel_running_large_job_mid_assembly(manager_factory):
    # A single worker on a large mesh: cancel while RUNNING and confirm no
    # result is ever exposed and the state settles to cancelled.
    from .conftest import make_grid_model

    mgr = manager_factory(worker_threads=1, checkpoint_interval=8)
    info = mgr.submit(make_grid_model(40, 40))

    # Wait until it has actually entered RUNNING, then cancel promptly.
    running_seen = False
    deadline = time.time() + 5
    while time.time() < deadline:
        snap = mgr.get(info.job_id)
        if snap.state == JobState.RUNNING:
            running_seen = True
            break
        time.sleep(0.001)
    assert running_seen
    mgr.cancel(info.job_id)
    final = _wait_terminal(mgr, info.job_id)
    assert final.state == JobState.CANCELLED
    assert final.results is None


def test_batch_coalesces_duplicate_unfinished_models(manager_factory):
    mgr = manager_factory(worker_threads=1)
    # Two identical, not-yet-completed models in one batch share one job.
    jobs = mgr.submit_batch([make_model(load=9.0), make_model(load=9.0)])
    assert len(jobs) == 2
    assert jobs[0].job_id == jobs[1].job_id
    assert _wait_terminal(mgr, jobs[0].job_id).state == JobState.COMPLETED
