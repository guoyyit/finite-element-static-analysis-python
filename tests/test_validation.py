"""Input validation: degenerate elements, dangling nodes, undefined refs."""

from __future__ import annotations

import pytest

from fea.errors import ValidationError
from fea.validation import validate_model
from tests.conftest import make_model


def test_valid_model_passes(model):
    validate_model(model)


def test_dangling_node_rejected():
    m = make_model(nodes=[
        {"id": 1, "x": 0.0, "y": 0.0},
        {"id": 2, "x": 1.0, "y": 0.0},
        {"id": 3, "x": 1.0, "y": 1.0},
        {"id": 4, "x": 0.0, "y": 1.0},
        {"id": 5, "x": 9.0, "y": 9.0},  # not referenced by any element
    ])
    with pytest.raises(ValidationError) as ei:
        validate_model(m)
    assert ei.value.code == "validation_error"
    assert "dangling" in ei.value.message


def test_undefined_material_rejected():
    m = make_model(elements=[{"id": 1, "nodes": [1, 2, 3, 4],
                              "material_id": 99, "thickness": 0.1}])
    with pytest.raises(ValidationError, match="undefined material"):
        validate_model(m)


def test_undefined_node_reference_rejected():
    m = make_model(elements=[{"id": 1, "nodes": [1, 2, 3, 42],
                              "material_id": 1, "thickness": 0.1}])
    with pytest.raises(ValidationError, match="undefined node"):
        validate_model(m)


def test_degenerate_repeated_node_rejected():
    m = make_model(elements=[{"id": 1, "nodes": [1, 2, 3, 3],
                              "material_id": 1, "thickness": 0.1}])
    with pytest.raises(ValidationError, match="repeated node"):
        validate_model(m)


def test_degenerate_collinear_element_rejected():
    m = make_model(
        nodes=[
            {"id": 1, "x": 0.0, "y": 0.0},
            {"id": 2, "x": 1.0, "y": 0.0},
            {"id": 3, "x": 2.0, "y": 0.0},
            {"id": 4, "x": 3.0, "y": 0.0},
        ],
        elements=[{"id": 1, "nodes": [1, 2, 3, 4], "material_id": 1,
                   "thickness": 0.1}],
    )
    with pytest.raises(ValidationError, match="degenerate"):
        validate_model(m)


def test_self_intersecting_element_rejected():
    m = make_model(elements=[{"id": 1, "nodes": [1, 3, 2, 4],
                              "material_id": 1, "thickness": 0.1}])
    with pytest.raises(ValidationError, match="non-convex|self-intersecting"):
        validate_model(m)


def test_duplicate_node_id_rejected():
    m = make_model(nodes=[
        {"id": 1, "x": 0.0, "y": 0.0},
        {"id": 1, "x": 5.0, "y": 5.0},
        {"id": 2, "x": 1.0, "y": 0.0},
        {"id": 3, "x": 1.0, "y": 1.0},
        {"id": 4, "x": 0.0, "y": 1.0},
    ])
    with pytest.raises(ValidationError, match="duplicate node id"):
        validate_model(m)


def test_conflicting_bcs_rejected():
    m = make_model(displacement_bcs=[
        {"node_id": 1, "direction": "x", "value": 0.0},
        {"node_id": 1, "direction": "x", "value": 1.0},
    ])
    with pytest.raises(ValidationError, match="conflicting"):
        validate_model(m)


def test_bc_on_undefined_node_rejected():
    m = make_model(displacement_bcs=[{"node_id": 77, "direction": "x"}])
    with pytest.raises(ValidationError, match="undefined node"):
        validate_model(m)


def test_load_on_undefined_node_rejected():
    m = make_model(point_loads=[{"node_id": 77, "fx": 1.0}])
    with pytest.raises(ValidationError, match="undefined node"):
        validate_model(m)


def test_bad_material_params_rejected():
    m = make_model(materials=[{"id": 1, "E": -1.0, "nu": 0.3}])
    with pytest.raises(ValidationError, match="Young"):
        validate_model(m)
    m = make_model(materials=[{"id": 1, "E": 1.0, "nu": 0.6}])
    with pytest.raises(ValidationError, match="Poisson"):
        validate_model(m)
