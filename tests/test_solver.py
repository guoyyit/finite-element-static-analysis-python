"""Solver semantics: equilibrium, reactions, singularity, determinism."""

from __future__ import annotations

import pytest

from fea.errors import EquilibriumError, SingularMatrixError
from fea.models import ModelInput
from fea.solver import solve_model
from tests.conftest import make_model


def test_equilibrium_residual_within_tolerance(model):
    result = solve_model(model)
    assert result.max_equilibrium_residual <= model.options.equilibrium_tol


def test_global_force_balance(model):
    result = solve_model(model)
    sum_rx = sum(r.rx for r in result.reactions)
    sum_ry = sum(r.ry for r in result.reactions)
    sum_fx = sum(p.fx for p in model.point_loads)
    sum_fy = sum(p.fy for p in model.point_loads)
    assert abs(sum_rx + sum_fx) < 1e-6
    assert abs(sum_ry + sum_fy) < 1e-6


def test_reactions_only_on_constrained_nodes(model):
    result = solve_model(model)
    assert {r.node_id for r in result.reactions} == {1, 4}


def test_prescribed_displacements_respected():
    m = make_model(displacement_bcs=[
        {"node_id": 1, "direction": "x", "value": 0.0},
        {"node_id": 1, "direction": "y", "value": 0.0},
        {"node_id": 4, "direction": "x", "value": 0.0},
        {"node_id": 4, "direction": "y", "value": 1.0e-3},
    ])
    result = solve_model(m)
    d4 = next(d for d in result.displacements if d.node_id == 4)
    assert d4.uy == pytest.approx(1.0e-3)


def test_unsupported_model_is_singular():
    m = make_model(displacement_bcs=[])
    with pytest.raises(SingularMatrixError) as ei:
        solve_model(m)
    assert ei.value.code == "singular_matrix"


def test_mechanism_is_singular():
    # Only one constrained DOF: rigid body rotations remain.
    m = make_model(displacement_bcs=[{"node_id": 1, "direction": "x"}])
    with pytest.raises(SingularMatrixError):
        solve_model(m)


def test_equilibrium_tolerance_enforced():
    # An absurdly tight tolerance must trigger the equilibrium guard.
    m = make_model(options={"equilibrium_tol": 1e-18})
    with pytest.raises(EquilibriumError) as ei:
        solve_model(m)
    assert ei.value.code == "equilibrium_violation"


def test_deterministic_bit_identical_results(model):
    r1 = solve_model(model)
    r2 = solve_model(model)
    assert r1.model_dump_json() == r2.model_dump_json()


def test_plane_strain_supported():
    m = make_model(analysis_type="plane_strain")
    result = solve_model(m)
    assert result.max_equilibrium_residual <= m.options.equilibrium_tol
    assert all(er.von_mises >= 0.0 for er in result.element_results)


def test_two_element_patch_moment_balance():
    # 2x1 mesh cantilever, check moment balance about node 1 as well.
    m = ModelInput.model_validate({
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0}, {"id": 2, "x": 1.0, "y": 0.0},
            {"id": 3, "x": 2.0, "y": 0.0}, {"id": 4, "x": 0.0, "y": 1.0},
            {"id": 5, "x": 1.0, "y": 1.0}, {"id": 6, "x": 2.0, "y": 1.0},
        ],
        "elements": [
            {"id": 1, "nodes": [1, 2, 5, 4], "material_id": 1,
             "thickness": 0.1},
            {"id": 2, "nodes": [2, 3, 6, 5], "material_id": 1,
             "thickness": 0.1},
        ],
        "materials": [{"id": 1, "E": 2.0e11, "nu": 0.3}],
        "displacement_bcs": [
            {"node_id": 1, "direction": "x"},
            {"node_id": 1, "direction": "y"},
            {"node_id": 4, "direction": "x"},
            {"node_id": 4, "direction": "y"},
        ],
        "point_loads": [{"node_id": 3, "fy": -1000.0},
                        {"node_id": 6, "fy": -1000.0}],
    })
    result = solve_model(m)
    sum_ry = sum(r.ry for r in result.reactions)
    assert abs(sum_ry - 2000.0) < 1e-6
    # Moment about origin: applied loads at x=2 give M = 2*(-2000) = -4000;
    # only the horizontal reaction at node 4 (y=1) can balance it:
    # -1 * rx4 + (-4000) = 0  =>  rx4 = -4000.
    r4 = next(r for r in result.reactions if r.node_id == 4)
    assert abs(r4.rx + 4000.0) < 1e-4
