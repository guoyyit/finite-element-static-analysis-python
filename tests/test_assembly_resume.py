"""Assembly resumes from the latest checkpoint and produces identical output."""

from __future__ import annotations

import time

import numpy as np

from fem.checkpoint import CheckpointStore
from fem.config import ServiceConfig
from fem.manager import JobManager
from fem.models import JobState
from fem.solver import AssembledSystem, assemble, solve
from fem.validation import validate_model

from .conftest import make_model


def _two_element_model():
    from fem.models import ElementInput, NodeInput
    nodes = [NodeInput(id=1, x=0, y=0), NodeInput(id=2, x=1, y=0),
             NodeInput(id=3, x=2, y=0), NodeInput(id=4, x=2, y=1),
             NodeInput(id=5, x=1, y=1), NodeInput(id=6, x=0, y=1)]
    elements = [ElementInput(id=1, type="CPS4", nodes=[1, 2, 5, 6],
                             material="s"),
                ElementInput(id=2, type="CPS4", nodes=[2, 3, 4, 5],
                             material="s")]
    return make_model(nodes=nodes, elements=elements)


def test_resumed_assembly_matches_fresh_assembly():
    cfg = ServiceConfig(checkpoint_interval=1)
    model = _two_element_model()
    nm = validate_model(model, cfg)

    captured: list[AssembledSystem] = []
    fresh = assemble(nm, cfg, checkpoint_cb=captured.append)

    # Simulate a crash after the first element: persist only that partial.
    partial = captured[0]
    assert partial.assembled_through == 1

    resumed = assemble(nm, cfg, partial=partial)
    assert resumed.assembled_through == 2
    assert np.allclose(resumed.kglob, fresh.kglob, atol=1e-14)

    # Solving from the resumed system must equal solving from scratch.
    full_result = solve(nm, cfg, model_hash=nm.canonical_hash())
    resumed_result = solve(nm, cfg, model_hash=nm.canonical_hash(),
                           assembly=resumed)
    assert resumed_result.model_dump_json() == full_result.model_dump_json()


def test_stale_checkpoint_shape_forces_restart(tmp_path):
    cfg = ServiceConfig(checkpoint_interval=1)
    model = _two_element_model()
    nm = validate_model(model, cfg)
    good = assemble(nm, cfg)
    # A matrix with wrong DOF dimensions must be rejected.
    bad = AssembledSystem(kglob=np.zeros((4, 4)), assembled_through=1)
    try:
        AssembledSystem.from_payload(bad.to_payload(), nm.ndof)
        raise AssertionError("expected shape validation")
    except ValueError:
        pass
    assert good.assembled_through == 2


def test_manager_resumes_from_persisted_assembly(tmp_path):
    d = str(tmp_path / "store")
    cfg = ServiceConfig(storage_dir=d, worker_threads=1,
                        checkpoint_interval=1)

    # Phase 1: run to completion once to get reference output.
    ref_mgr = JobManager(config=cfg)
    ref_mgr.start()
    ref_info = ref_mgr.submit(_two_element_model())
    deadline = time.time() + 5
    while time.time() < deadline:
        snap = ref_mgr.get(ref_info.job_id)
        if snap.state == JobState.COMPLETED:
            break
        time.sleep(0.005)
    reference = ref_mgr.get_result(ref_info.job_id).model_dump_json()
    ref_mgr.shutdown()

    # Phase 2: build a partial checkpoint manually then recover a manager.
    model = _two_element_model()
    nm = validate_model(model, cfg)
    captured: list[AssembledSystem] = []
    assemble(nm, cfg, checkpoint_cb=captured.append)
    # Real partial: only the first element accumulated (independent buffer).
    partial = captured[0]
    assert partial.assembled_through == 1

    from fem.models import JobInfo
    store = CheckpointStore(d)
    store.initialize()
    jid = "deadbeef00000001"
    store.save_model(jid, JobManager._model_payload(model))
    store.save_assembly(jid, partial.kglob)
    info = JobInfo(job_id=jid, state=JobState.RUNNING,
                   model_hash=nm.canonical_hash(), progress=0.3,
                   assembled_through=1, assembly_file=f"{jid}.K.npy",
                   created_at=time.time(), updated_at=time.time())
    store.save(info)

    mgr2 = JobManager(config=cfg)
    mgr2.start()
    deadline = time.time() + 5
    while time.time() < deadline:
        snap = mgr2.get(jid)
        if snap.state in (JobState.COMPLETED, JobState.FAILED):
            break
        time.sleep(0.005)
    assert snap.state == JobState.COMPLETED
    assert mgr2.get_result(jid).model_dump_json() == reference
    # Assembly blob cleaned after successful completion.
    assert store.load_assembly(jid) is None
    mgr2.shutdown()
