"""Cross-run float determinism and version binding."""

from __future__ import annotations

import json

from fem.checkpoint import CheckpointStore
from fem.config import ServiceConfig
from fem.manager import JobManager

from .conftest import make_model


def _run_fresh(directory, load=3.5):
    cfg = ServiceConfig(storage_dir=directory)
    store = CheckpointStore(directory)
    mgr = JobManager(config=cfg, store=store)
    mgr.start()
    info = mgr.submit(make_model(load=load))
    mgr.get  # noop
    import time
    deadline = time.time() + 5
    while time.time() < deadline:
        snap = mgr.get(info.job_id)
        if snap.state.value == "completed":
            break
        time.sleep(0.005)
    out = mgr.get_result(info.job_id).model_dump(mode="json")
    mgr.shutdown(wait=True, timeout=5)
    return out


def test_identical_run_bytes_identical(tmp_path):
    d1 = str(tmp_path / "a")
    d2 = str(tmp_path / "b")
    r1 = _run_fresh(d1)
    r2 = _run_fresh(d2)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_different_parameter_gives_different_result(tmp_path):
    r1 = _run_fresh(str(tmp_path / "a"), load=1.0)
    r2 = _run_fresh(str(tmp_path / "b"), load=2.0)
    u1 = [d["ux"] for d in r1["displacements"]]
    u2 = [d["ux"] for d in r2["displacements"]]
    assert u1 != u2


def test_result_carries_solver_version_and_hash(tmp_path):
    r = _run_fresh(str(tmp_path / "a"))
    assert r["solver_version"]
    assert len(r["model_hash"]) == 64
