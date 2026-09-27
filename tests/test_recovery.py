"""Checkpoint & restart recovery, timeout isolation and determinism across
process restarts (simulated by new manager instances over the same store)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from fem.checkpoint import CheckpointStore
from fem.config import ServiceConfig
from fem.errors import JobInvalidStateError
from fem.manager import JobManager
from fem.models import JobState, SolveOptions

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


def _new_manager(directory: str, **kw) -> JobManager:
    cfg = ServiceConfig(storage_dir=directory,
                        worker_threads=kw.pop("worker_threads", 2), **kw)
    mgr = JobManager(config=cfg)
    mgr.start()
    return mgr


def test_completed_job_survives_restart_without_resolve(tmp_path):
    d = str(tmp_path / "store")
    mgr = _new_manager(d)
    info = mgr.submit(make_model(load=1.5))
    final = _wait_terminal(mgr, mgr.get(info.job_id).job_id)
    expected_json = final.results.model_dump_json()
    mgr.shutdown()

    # Simulate a fresh process over the same directory.
    store = CheckpointStore(d)
    on_disk = store.load(info.job_id)
    assert on_disk.state == JobState.COMPLETED

    mgr2 = _new_manager(d)
    restarted = mgr2.get(info.job_id)
    assert restarted.state == JobState.COMPLETED
    assert restarted.results.model_dump_json() == expected_json
    assert mgr2.get_result(info.job_id).model_dump_json() == expected_json
    mgr2.shutdown()


def test_unfinished_job_resumes_after_restart(tmp_path):
    d = str(tmp_path / "store")
    mgr = _new_manager(d, worker_threads=1)
    # Occupy the single worker so the queued job stays PENDING on disk.
    blocker = mgr.submit(make_model(load=1.0))
    queued = mgr.submit(make_model(load=3.0))
    time.sleep(0.05)
    # Hard "crash": drop the manager without graceful cancellation bookkeeping.
    mgr.shutdown(wait=False)
    # The queued job still has a persisted model to resume from.
    assert CheckpointStore(d).load_model_raw(queued.job_id) is not None

    mgr2 = _new_manager(d, worker_threads=2)
    final = _wait_terminal(mgr2, queued.job_id)
    assert final.state == JobState.COMPLETED
    assert final.results is not None
    _wait_terminal(mgr2, blocker.job_id)
    mgr2.shutdown()


def test_repeated_resolve_does_not_recompute_completed(tmp_path):
    d = str(tmp_path / "store")
    mgr = _new_manager(d)
    info = mgr.submit(make_model(load=2.25))
    _wait_terminal(mgr, info.job_id)
    mgr.shutdown()

    mgr2 = _new_manager(d)
    again = mgr2.submit(make_model(load=2.25))
    assert again.job_id == info.job_id
    assert again.state == JobState.COMPLETED
    mgr2.shutdown()


def test_results_deterministic_across_restart(tmp_path):
    d = str(tmp_path / "store")
    mgr = _new_manager(d)
    info = mgr.submit(make_model(load=4.0))
    _wait_terminal(mgr, info.job_id)
    first = mgr.get_result(info.job_id).model_dump_json()
    mgr.shutdown()

    mgr2 = _new_manager(d)
    second = mgr2.get_result(info.job_id).model_dump_json()
    assert first == second
    mgr2.shutdown()


def test_timeout_job_does_not_poison_others_or_store_result(tmp_path):
    d = str(tmp_path / "store")
    mgr = _new_manager(d, worker_threads=2)
    slow = make_model(load=1.0)
    slow.options = SolveOptions(timeout_seconds=1e-9)  # effectively immediate
    good = mgr.submit(make_model(load=2.0))
    timed = mgr.submit(slow)
    tf = _wait_terminal(mgr, timed.job_id)
    gf = _wait_terminal(mgr, good.job_id)
    assert tf.state == JobState.TIMEOUT
    assert tf.results is None
    assert tf.error.code == "job_timeout"
    assert gf.state == JobState.COMPLETED
    with pytest.raises(JobInvalidStateError):
        mgr.get_result(timed.job_id)
    # On disk the timed-out job has no results payload either.
    on_disk = CheckpointStore(d).load(timed.job_id)
    assert on_disk.state == JobState.TIMEOUT
    assert on_disk.results is None
    mgr.shutdown()


def test_no_tmp_files_left_after_runs(tmp_path):
    d = str(tmp_path / "store")
    mgr = _new_manager(d)
    for v in (1.0, 2.0, 3.0):
        info = mgr.submit(make_model(load=v))
        _wait_terminal(mgr, info.job_id)
    leftovers = list(Path(d).glob("*.tmp"))
    assert leftovers == []
    mgr.shutdown()


def test_terminal_failed_job_is_not_resolved_after_restart(tmp_path):
    d = str(tmp_path / "store")
    mgr = _new_manager(d, worker_threads=2)
    bad = make_model(load=1.0)
    bad.options = SolveOptions(equilibrium_tolerance=1e-300)
    info = mgr.submit(bad)
    final = _wait_terminal(mgr, info.job_id)
    assert final.state == JobState.FAILED
    assert final.error.code == "equilibrium_not_satisfied"
    mgr.shutdown()

    # A new process must keep the failed record, not silently recompute it.
    mgr2 = _new_manager(d, worker_threads=2)
    import time as _t
    _t.sleep(0.2)  # give any (incorrect) re-run a chance to happen
    again = mgr2.get(info.job_id)
    assert again.state == JobState.FAILED
    assert again.results is None
    assert again.error is not None
    assert again.error.code == "equilibrium_not_satisfied"
    mgr2.shutdown()
