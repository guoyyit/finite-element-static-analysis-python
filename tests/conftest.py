"""Shared pytest fixtures: small validated FEM models and temp managers."""

from __future__ import annotations

import pytest

from fem.config import ServiceConfig
from fem.manager import JobManager
from fem.models import (
    BoundaryConditionInput,
    ElementInput,
    MaterialInput,
    ModelInput,
    NodalLoadInput,
    NodeInput,
)


def make_model(*, load: float = 1.0, material: str = "s",
               element_type: str = "CPS4", poisson: float = 0.0,
               young: float = 1.0, thickness: float = 1.0,
               nodes=None, elements=None, bcs=None, loads=None,
               extra_nodes=None) -> ModelInput:
    default_nodes = [
        NodeInput(id=1, x=0, y=0), NodeInput(id=2, x=1, y=0),
        NodeInput(id=3, x=1, y=1), NodeInput(id=4, x=0, y=1),
    ]
    default_elements = [
        ElementInput(id=1, type=element_type, nodes=[1, 2, 3, 4],
                     material=material)
    ]
    default_bcs = [
        BoundaryConditionInput(node=1, dof=0),
        BoundaryConditionInput(node=1, dof=1),
        BoundaryConditionInput(node=2, dof=1),
        BoundaryConditionInput(node=4, dof=1),
    ]
    default_loads = [NodalLoadInput(node=3, dof=0, value=load)]
    mats = [MaterialInput(name="s", youngs_modulus=young,
                          poisson_ratio=poisson, thickness=thickness)]
    ns = nodes if nodes is not None else list(default_nodes)
    if extra_nodes:
        ns = list(ns) + list(extra_nodes)
    return ModelInput(
        nodes=ns,
        elements=elements if elements is not None else default_elements,
        materials=mats,
        boundary_conditions=bcs if bcs is not None else default_bcs,
        loads=loads if loads is not None else default_loads,
    )


@pytest.fixture
def single_element_model() -> ModelInput:
    return make_model()


def make_grid_model(nx: int = 30, ny: int = 30) -> ModelInput:
    """A (nx x ny) Q4 cantilever-ish plate for larger/heavier jobs."""
    nodes: list[NodeInput] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append(NodeInput(id=j * (nx + 1) + i + 1,
                                   x=float(i), y=float(j)))

    def nid(i: int, j: int) -> int:
        return j * (nx + 1) + i + 1

    elements = [
        ElementInput(
            id=j * nx + i + 1, type="CPS4",
            nodes=[nid(i, j), nid(i + 1, j),
                   nid(i + 1, j + 1), nid(i, j + 1)],
            material="s")
        for j in range(ny) for i in range(nx)
    ]
    bcs = []
    for j in range(ny + 1):
        bcs.append(BoundaryConditionInput(node=nid(0, j), dof=0))
        bcs.append(BoundaryConditionInput(node=nid(0, j), dof=1))
    loads = [NodalLoadInput(node=nid(nx, ny), dof=1, value=-1000.0)]
    return ModelInput(
        nodes=nodes, elements=elements,
        materials=[MaterialInput(name="s", youngs_modulus=2.1e8,
                                 poisson_ratio=0.3, thickness=0.05)],
        boundary_conditions=bcs, loads=loads)


@pytest.fixture
def manager_factory(tmp_path):
    created: list[JobManager] = []

    def _factory(**config_kwargs) -> JobManager:
        kwargs = dict(storage_dir=str(tmp_path / ("jobs%d" % len(created))),
                      worker_threads=2)
        kwargs.update(config_kwargs)
        mgr = JobManager(config=ServiceConfig(**kwargs))
        mgr.start()
        created.append(mgr)
        return mgr

    yield _factory

    for mgr in created:
        mgr.shutdown(wait=False)
