from __future__ import annotations

import pytest

from fea.limits import ResourceLimits
from fea.models import ModelInput
from fea.storage import JobStore
from fea.jobs import JobManager


def make_model(**overrides) -> ModelInput:
    """A minimal, valid cantilever-like single-Q4 model."""
    data = dict(
        analysis_type="plane_stress",
        nodes=[
            {"id": 1, "x": 0.0, "y": 0.0},
            {"id": 2, "x": 1.0, "y": 0.0},
            {"id": 3, "x": 1.0, "y": 1.0},
            {"id": 4, "x": 0.0, "y": 1.0},
        ],
        elements=[{"id": 1, "nodes": [1, 2, 3, 4], "material_id": 1,
                   "thickness": 0.1}],
        materials=[{"id": 1, "E": 2.0e11, "nu": 0.3}],
        displacement_bcs=[
            {"node_id": 1, "direction": "x"},
            {"node_id": 1, "direction": "y"},
            {"node_id": 4, "direction": "x"},
            {"node_id": 4, "direction": "y"},
        ],
        point_loads=[
            {"node_id": 2, "fy": -1000.0},
            {"node_id": 3, "fy": -1000.0},
        ],
    )
    data.update(overrides)
    return ModelInput.model_validate(data)


@pytest.fixture
def model() -> ModelInput:
    return make_model()


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.db")
    s.initialize()
    yield s
    s.close()


@pytest.fixture
def manager(store):
    m = JobManager(store, limits=ResourceLimits(max_concurrent_jobs=2))
    m.start()
    yield m
    m.shutdown(wait=True)


def wait_terminal(manager: JobManager, job_id: str, timeout: float = 10.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = manager.get(job_id)
        if info.status.value in ("completed", "failed", "cancelled"):
            return info
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal state")
