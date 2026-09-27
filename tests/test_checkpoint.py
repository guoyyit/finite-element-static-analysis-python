"""Checkpoint store: atomicity, corruption handling and path safety."""

from __future__ import annotations

import pytest

from fem.checkpoint import CheckpointStore
from fem.errors import CheckpointError
from fem.models import JobInfo, JobState


def _job(jid="j1"):
    return JobInfo(job_id=jid, state=JobState.RUNNING, model_hash="h",
                   progress=0.5)


def test_roundtrip_and_deterministic_serialization(tmp_path):
    store = CheckpointStore(str(tmp_path))
    store.initialize()
    info = _job()
    store.save(info)
    loaded = store.load("j1")
    assert loaded == info
    assert store.dumps(loaded) == store.dumps(info)


def test_load_missing_returns_none(tmp_path):
    store = CheckpointStore(str(tmp_path))
    store.initialize()
    assert store.load("ghost") is None


def test_load_all_skips_corrupt_files(tmp_path):
    store = CheckpointStore(str(tmp_path))
    store.initialize()
    store.save(_job("good"))
    (tmp_path / "bad.json").write_text("{broken")
    allj = store.load_all()
    assert list(allj) == ["good"]
    with pytest.raises(CheckpointError):
        store.load("bad")


def test_invalid_job_id_rejected(tmp_path):
    store = CheckpointStore(str(tmp_path))
    store.initialize()
    for bad in ("", "../x", "a/b"):
        with pytest.raises(CheckpointError):
            store.job_path(bad)


def test_no_temp_files_after_save(tmp_path):
    store = CheckpointStore(str(tmp_path))
    store.initialize()
    store.save(_job("x"))
    assert list(tmp_path.glob("*.tmp")) == []


def test_model_payload_roundtrip(tmp_path):
    store = CheckpointStore(str(tmp_path))
    store.initialize()
    store.save_model("j1", b'{"a":1}')
    assert store.load_model_raw("j1") == b'{"a":1}'
    store.delete("j1")
    assert store.load("j1") is None
    assert store.load_model_raw("j1") is None
