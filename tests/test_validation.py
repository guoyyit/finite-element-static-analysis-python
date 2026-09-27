"""Input validation: distinct errors, strict rejection, no silent repair."""

from __future__ import annotations

import pytest

from fem.config import ServiceConfig
from fem.errors import (
    DanglingNodeError,
    DegenerateElementError,
    InvalidBoundaryConditionError,
    InvalidElementError,
    InvalidLoadError,
    InvalidNodeError,
    UndefinedMaterialError,
    UnconstrainedModelError,
)
from fem.models import (
    BoundaryConditionInput,
    ElementInput,
    NodalLoadInput,
    NodeInput,
)
from fem.validation import validate_model

from .conftest import make_model

CFG = ServiceConfig()


def test_valid_single_element_accepted(single_element_model):
    nm = validate_model(single_element_model, CFG)
    assert nm.nn == 4
    assert nm.ne == 1
    assert nm.ndof == 8


def test_undefined_material_rejected():
    m = make_model()
    m.elements[0].material = "ghost"
    with pytest.raises(UndefinedMaterialError):
        validate_model(m, CFG)


def test_dangling_node_rejected():
    m = make_model(extra_nodes=[NodeInput(id=99, x=5, y=5)])
    with pytest.raises(DanglingNodeError) as exc:
        validate_model(m, CFG)
    assert 99 in exc.value.details["nodes"]


def test_degenerate_duplicate_node_rejected():
    m = make_model()
    m.nodes[3] = NodeInput(id=4, x=0, y=0)  # collapse 4 onto 1
    with pytest.raises(DegenerateElementError):
        validate_model(m, CFG)


def test_degenerate_bowtie_rejected():
    m = make_model()
    # Reorder to a self-intersecting polygon (still distinct nodes).
    m.elements[0].nodes = [1, 3, 2, 4]
    with pytest.raises(DegenerateElementError):
        validate_model(m, CFG)


def test_unsupported_element_type_rejected():
    m = make_model()
    m.elements[0].type = "HEX8"
    with pytest.raises(InvalidElementError) as exc:
        validate_model(m, CFG)
    assert "HEX8" in str(exc.value)


def test_element_undefined_node_rejected():
    m = make_model()
    m.elements[0].nodes = [1, 2, 3, 77]
    with pytest.raises(InvalidElementError):
        validate_model(m, CFG)


def test_duplicate_node_id_rejected():
    m = make_model()
    m.nodes.append(NodeInput(id=1, x=2, y=2))
    with pytest.raises(InvalidNodeError):
        validate_model(m, CFG)


def test_non_finite_coordinate_rejected():
    m = make_model()
    m.nodes[0].x = float("nan")
    with pytest.raises(InvalidNodeError):
        validate_model(m, CFG)


def test_duplicate_element_id_rejected():
    m = make_model()
    m.elements.append(ElementInput(
        id=1, type="CPS4", nodes=[1, 2, 3, 4], material="s"))
    with pytest.raises(InvalidElementError):
        validate_model(m, CFG)


def test_bc_undefined_node_rejected():
    m = make_model()
    m.boundary_conditions.append(
        BoundaryConditionInput(node=42, dof=0))
    with pytest.raises(InvalidBoundaryConditionError):
        validate_model(m, CFG)


def test_duplicate_bc_rejected():
    m = make_model()
    m.boundary_conditions.append(
        BoundaryConditionInput(node=1, dof=0, value=0.0))
    with pytest.raises(InvalidBoundaryConditionError):
        validate_model(m, CFG)


def test_load_undefined_node_rejected():
    m = make_model()
    m.loads.append(NodalLoadInput(node=42, dof=0, value=1.0))
    with pytest.raises(InvalidLoadError):
        validate_model(m, CFG)


def test_missing_constraints_rejected():
    m = make_model(bcs=[])
    with pytest.raises(UnconstrainedModelError):
        validate_model(m, CFG)


def test_node_count_limit_rejected():
    cfg = ServiceConfig(max_nodes=3)
    with pytest.raises(InvalidNodeError):
        validate_model(make_model(), cfg)


def test_memory_limit_rejected():
    cfg = ServiceConfig(max_memory_bytes=100)
    with pytest.raises(InvalidNodeError):
        validate_model(make_model(), cfg)


def test_canonical_hash_order_independent_and_bound(single_element_model):
    import copy

    nm1 = validate_model(single_element_model, CFG)
    reordered = copy.deepcopy(single_element_model)
    reordered.nodes = list(reversed(reordered.nodes))
    reordered.elements = list(reversed(reordered.elements))
    nm2 = validate_model(reordered, CFG)
    assert nm1.canonical_hash() == nm2.canonical_hash()


def test_hash_bound_to_solver_version(single_element_model):
    import fem.models as models_mod
    import copy

    h1 = validate_model(single_element_model, CFG).canonical_hash()
    m2 = copy.deepcopy(single_element_model)
    old = models_mod.SOLVER_VERSION
    try:
        models_mod.SOLVER_VERSION = "fem-solver-9.9.9"
        # Rebuild a model referencing the bumped version payload.
        nm2 = validate_model(m2, CFG)
        assert nm2.canonical_hash() != h1
    finally:
        models_mod.SOLVER_VERSION = old


def test_hash_changes_when_load_changes(single_element_model):
    import copy
    h1 = validate_model(single_element_model, CFG).canonical_hash()
    m2 = copy.deepcopy(single_element_model)
    m2.loads[0].value = 2.5
    assert validate_model(m2, CFG).canonical_hash() != h1
