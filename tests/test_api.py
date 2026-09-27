"""HTTP API: submission, query, cancellation, distinguishable errors."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_model


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FEA_DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("FEA_MAX_CONCURRENT_JOBS", "2")
    from main import app

    with TestClient(app) as c:
        yield c


def _wait(client, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = client.get(f"/jobs/{job_id}").json()
        if info["status"] in ("completed", "failed", "cancelled"):
            return info
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_submit_and_fetch_result(client):
    r = client.post("/jobs", json=make_model().model_dump(mode="json"))
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    info = _wait(client, job_id)
    assert info["status"] == "completed"
    assert info["result"]["max_equilibrium_residual"] <= 1e-6
    assert len(info["result"]["displacements"]) == 4
    assert len(info["result"]["element_results"]) == 1
    assert len(info["result"]["reactions"]) == 2
    assert info["model_hash"]


def test_invalid_model_returns_422_with_code(client):
    bad = make_model(
        elements=[{"id": 1, "nodes": [1, 2, 3, 4], "material_id": 99,
                   "thickness": 0.1}]
    )
    r = client.post("/jobs", json=bad.model_dump(mode="json"))
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "validation_error"
    assert "undefined material" in body["message"]


def test_degenerate_element_returns_422(client):
    bad = make_model(
        elements=[{"id": 1, "nodes": [1, 2, 3, 3], "material_id": 1,
                   "thickness": 0.1}]
    )
    r = client.post("/jobs", json=bad.model_dump(mode="json"))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_dangling_node_returns_422(client):
    bad = make_model(nodes=[
        {"id": 1, "x": 0.0, "y": 0.0}, {"id": 2, "x": 1.0, "y": 0.0},
        {"id": 3, "x": 1.0, "y": 1.0}, {"id": 4, "x": 0.0, "y": 1.0},
        {"id": 5, "x": 5.0, "y": 5.0},
    ])
    r = client.post("/jobs", json=bad.model_dump(mode="json"))
    assert r.status_code == 422
    assert "dangling" in r.json()["error"]["message"]


def test_oversized_model_returns_413(client, monkeypatch):
    # Tighten the limit for this request by restarting the app.
    monkeypatch.setenv("FEA_MAX_NODES", "3")
    from main import app

    with TestClient(app) as c2:
        r = c2.post("/jobs", json=make_model().model_dump(mode="json"))
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "limit_exceeded"


def test_unknown_job(client):
    r = client.get("/jobs/does-not-exist")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "job_not_found"


def test_cancel_via_api(client):
    import threading

    mgr = client.app.state.manager
    gate = threading.Event()
    mgr._pre_solve_hook = lambda _jid: gate.wait(5)
    try:
        r1 = client.post("/jobs", json=make_model().model_dump(mode="json"))
        r2 = client.post("/jobs", json=make_model().model_dump(mode="json"))
        j1, j2 = r1.json()["job_id"], r2.json()["job_id"]
        r = client.post(f"/jobs/{j2}/cancel")
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"
        assert r.json()["result"] is None
        gate.set()
        # The other job is unaffected.
        assert _wait(client, j1)["status"] == "completed"
        assert client.get(f"/jobs/{j2}").json()["status"] == "cancelled"
    finally:
        gate.set()
        mgr._pre_solve_hook = None


def test_cancel_completed_job_via_api(client):
    r = client.post("/jobs", json=make_model().model_dump(mode="json"))
    job_id = r.json()["job_id"]
    assert _wait(client, job_id)["status"] == "completed"
    r = client.post(f"/jobs/{job_id}/cancel")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "job_state_error"


def test_list_jobs(client):
    client.post("/jobs", json=make_model().model_dump(mode="json"))
    client.post("/jobs", json=make_model().model_dump(mode="json"))
    jobs = client.get("/jobs").json()
    assert len(jobs) == 2
    assert all(j["model_hash"] for j in jobs)


def test_determinism_via_api(client):
    ids = [
        client.post("/jobs", json=make_model().model_dump(mode="json"))
        .json()["job_id"]
        for _ in range(2)
    ]
    results = [_wait(client, i)["result"] for i in ids]
    assert results[0] == results[1]
