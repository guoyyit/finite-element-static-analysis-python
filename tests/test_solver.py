"""Solver behavior: equilibrium, constraints, recovery and determinism."""

from __future__ import annotations

import threading

import pytest

from fem.config import ServiceConfig
from fem.errors import (
    EquilibriumNotSatisfiedError,
    JobCancelledError,
    SingularMatrixError,
)
from fem.models import (
    BoundaryConditionInput,
    NodalLoadInput,
    NodeInput,
)
from fem.solver import solve
from fem.validation import validate_model

from .conftest import make_model


def _solve(model, cfg=ServiceConfig()):
    nm = validate_model(model, cfg)
    return nm, solve(nm, cfg, model_hash=nm.canonical_hash())


def _global_forces(model, results) -> tuple[float, float]:
    """Sum applied loads and support reactions over both components."""
    fx_load = sum(ld.value for ld in model.loads if ld.dof == 0)
    fy_load = sum(ld.value for ld in model.loads if ld.dof == 1)
    rx = sum(r.fx for r in results.reactions)
    ry = sum(r.fy for r in results.reactions)
    return fx_load + rx, fy_load + ry


def test_force_balance_single_element(single_element_model):
    nm, res = _solve(single_element_model)
    bx, by = _global_forces(single_element_model, res)
    assert abs(bx) < 1e-9
    assert abs(by) < 1e-9
    assert res.residual.max_abs_residual <= res.residual.tolerance


def test_patch_test_uniform_stress():
    # Two Q4s in series; constant traction must give constant stress sxx=1.
    from fem.models import ElementInput, MaterialInput, ModelInput

    nodes = [NodeInput(id=1, x=0, y=0), NodeInput(id=2, x=1, y=0),
             NodeInput(id=3, x=2, y=0), NodeInput(id=4, x=2, y=1),
             NodeInput(id=5, x=1, y=1), NodeInput(id=6, x=0, y=1)]
    elements = [ElementInput(id=1, type="CPS4", nodes=[1, 2, 5, 6],
                             material="s"),
                ElementInput(id=2, type="CPS4", nodes=[2, 3, 4, 5],
                             material="s")]
    m = ModelInput(
        nodes=nodes, elements=elements,
        materials=[MaterialInput(name="s", youngs_modulus=1.0,
                                 poisson_ratio=0.0, thickness=1.0)],
        boundary_conditions=[
            BoundaryConditionInput(node=1, dof=0),
            BoundaryConditionInput(node=1, dof=1),
            BoundaryConditionInput(node=6, dof=0)],
        loads=[NodalLoadInput(node=3, dof=0, value=0.5),
               NodalLoadInput(node=4, dof=0, value=0.5)])
    _nm, res = _solve(m)
    for s in res.stresses:
        assert abs(s.sxx - 1.0) < 1e-9
        assert abs(s.syy) < 1e-9
        assert abs(s.sxy) < 1e-9
    # Exact linear displacement ux = x.
    by_id = {d.node: d for d in res.displacements}
    for nid, x in [(1, 0), (2, 1), (3, 2), (4, 2), (5, 1), (6, 0)]:
        assert abs(by_id[nid].ux - x) < 1e-9


def test_plane_strain_runs_and_balances():
    m = make_model(element_type="CPE4", poisson=0.3, young=10.0)
    nm, res = _solve(m)
    bx, by = _global_forces(m, res)
    assert abs(bx) < 1e-8 and abs(by) < 1e-8
    assert len(res.stresses) == 1


def test_prescribed_nonzero_displacement_is_enforced():
    m = make_model(loads=[])
    m.boundary_conditions = [
        BoundaryConditionInput(node=1, dof=0, value=0.0),
        BoundaryConditionInput(node=1, dof=1, value=0.0),
        BoundaryConditionInput(node=2, dof=1, value=0.0),
        BoundaryConditionInput(node=4, dof=1, value=0.0),
        BoundaryConditionInput(node=3, dof=0, value=0.01),
    ]
    _nm, res = _solve(m)
    by_id = {d.node: d for d in res.displacements}
    assert by_id[3].ux == pytest.approx(0.01)
    # No external loads -> reactions must self-equilibrate to zero.
    assert abs(sum(r.fx for r in res.reactions)) < 1e-9
    assert abs(sum(r.fy for r in res.reactions)) < 1e-9


def test_residual_tolerance_configurable():
    # A strict, essentially-zero tolerance cannot be met for any non-trivial
    # solve once we force an artificial imbalance; use the natural residual
    # against a scale that makes the threshold absurdly tight instead.
    cfg = ServiceConfig(equilibrium_tolerance=1e-300,
                        residual_force_scale=1e-300)
    m = make_model()
    nm = validate_model(m, cfg)
    with pytest.raises(EquilibriumNotSatisfiedError):
        solve(nm, cfg, model_hash=nm.canonical_hash())


def test_singular_mechanism_detected():
    # Pin only one translation: rotation/translation remain -> singular.
    m = make_model()
    m.boundary_conditions = [BoundaryConditionInput(node=1, dof=0)]
    nm = validate_model(m, ServiceConfig())
    with pytest.raises(SingularMatrixError):
        solve(nm, ServiceConfig(), model_hash=nm.canonical_hash())


def test_cancel_event_aborts_solve(single_element_model):
    cfg = ServiceConfig()
    nm = validate_model(single_element_model, cfg)
    event = threading.Event()
    event.set()
    with pytest.raises(JobCancelledError):
        solve(nm, cfg, model_hash=nm.canonical_hash(), cancel_event=event)


def test_results_deterministic_across_runs(single_element_model):
    cfg = ServiceConfig()
    nm = validate_model(single_element_model, cfg)
    r1 = solve(nm, cfg, model_hash=nm.canonical_hash())
    r2 = solve(nm, cfg, model_hash=nm.canonical_hash())
    a = r1.model_dump_json()
    b = r2.model_dump_json()
    assert a == b
    # Explicit float reproducibility spot check.
    assert r1.displacements[2].ux == r2.displacements[2].ux


def test_output_version_bound(single_element_model):
    _nm, res = _solve(single_element_model)
    assert res.solver_version
    assert res.model_hash == _nm.canonical_hash()
