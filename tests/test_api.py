"""HTTP API: endpoints, status/error mapping and lifecycle via TestClient."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from fem.api import create_app
from fem.config import ServiceConfig

from .conftest import make_model


@pytest.fixture
def client(tmp_path):
    app = create_app(ServiceConfig(storage_dir=str(tmp_path / "api_jobs"),
                                   worker_threads=2))
    with TestClient(app) as c:
        yield c


def _wait(client, jid):
    for _ in range(500):
        info = client.get(f"/jobs/{jid}").json()
        if info["state"] in ("completed", "failed", "cancelled", "timeout"):
            return info
        time.sleep(0.005)
    raise AssertionError("job did not finish")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_submit_and_fetch_result(client, single_element_model):
    r = client.post("/jobs", json=single_element_model.model_dump(mode="json"))
    assert r.status_code == 201
    jid = r.json()["job"]["job_id"]
    info = _wait(client, jid)
    assert info["state"] == "completed"
    res = client.get(f"/jobs/{jid}/result")
    assert res.status_code == 200
    assert len(res.json()["displacements"]) == 4


def test_validation_error_envelope_and_status(client):
    payload = make_model().model_dump(mode="json")
    payload["elements"][0]["material"] = "ghost"
    r = client.post("/jobs", json=payload)
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "undefined_material"


def test_dangling_node_distinct_code(client):
    payload = make_model().model_dump(mode="json")
    payload["nodes"].append({"id": 99, "x": 10, "y": 10})
    r = client.post("/jobs", json=payload)
    assert r.json()["error"]["code"] == "dangling_node"


def test_schema_error_envelope(client):
    payload = make_model().model_dump(mode="json")
    payload["loads"][0]["dof"] = 7
    r = client.post("/jobs", json=payload)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request_schema"


def test_missing_job_404(client):
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/jobs/nope/result").status_code == 404


def test_cancel_and_list(client):
    # Submit several, list, and cancel a queued job under single worker.
    app2 = create_app(ServiceConfig(
        storage_dir=client.app.state.fem_config.storage_dir + "_b",
        worker_threads=1))
    with TestClient(app2) as c:
        ids = []
        for v in (1.0, 2.0, 3.0):
            r = c.post("/jobs", json=make_model(load=v).model_dump(mode="json"))
            ids.append(r.json()["job"]["job_id"])
        c.post(f"/jobs/{ids[2]}/cancel")
        info = _wait(c, ids[2])
        assert info["state"] == "cancelled"
        listed = c.get("/jobs").json()
        assert len(listed) == 3


def test_batch_endpoint_atomic(client):
    good = make_model(load=1.0).model_dump(mode="json")
    bad = make_model(load=2.0).model_dump(mode="json")
    bad["materials"] = []
    r = client.post("/jobs/batch", json={"models": [good, bad]})
    assert r.status_code == 422
    assert client.get("/jobs").json() == []


def test_failed_job_result_endpoint_conflict(client):
    payload = make_model().model_dump(mode="json")
    payload["options"] = {"equilibrium_tolerance": 1e-300}
    jid = client.post("/jobs", json=payload).json()["job"]["job_id"]
    info = _wait(client, jid)
    assert info["state"] == "failed"
    r = client.get(f"/jobs/{jid}/result")
    assert r.status_code == 409
