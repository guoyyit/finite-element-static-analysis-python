"""Structural validation of model input. Rejects, never silently fixes."""

from __future__ import annotations

from .errors import ValidationError
from .models import ModelInput

#: Absolute tolerance below which a quad corner is considered degenerate.
_AREA_TOL = 1e-12


def _check_unique_ids(kind: str, ids: list[int]) -> None:
    seen: set[int] = set()
    for i in ids:
        if i in seen:
            raise ValidationError(
                f"duplicate {kind} id {i}", details={kind: i}
            )
        seen.add(i)


def _cross(oa: tuple[float, float], ob: tuple[float, float]) -> float:
    return oa[0] * ob[1] - oa[1] * ob[0]


def _check_element_geometry(model: ModelInput) -> None:
    coords = {n.id: (n.x, n.y) for n in model.nodes}
    for el in model.elements:
        if len(set(el.nodes)) != 4:
            raise ValidationError(
                f"element {el.id} has repeated node references",
                details={"element_id": el.id, "nodes": el.nodes},
            )
        pts = [coords[nid] for nid in el.nodes]
        # Strict convexity: every corner triangle must turn the same way and
        # have non-negligible area. Catches degenerate, collinear and
        # self-intersecting (bow-tie) quads.
        signs: list[float] = []
        for i in range(4):
            a = pts[i]
            b = pts[(i + 1) % 4]
            c = pts[(i + 2) % 4]
            cross = _cross(
                (b[0] - a[0], b[1] - a[1]),
                (c[0] - b[0], c[1] - b[1]),
            )
            if abs(cross) <= _AREA_TOL:
                raise ValidationError(
                    f"element {el.id} is degenerate (zero-area corner at "
                    f"node {el.nodes[i]})",
                    details={"element_id": el.id},
                )
            signs.append(cross)
        if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
            raise ValidationError(
                f"element {el.id} is non-convex or self-intersecting",
                details={"element_id": el.id},
            )


def validate_model(model: ModelInput) -> None:
    """Raise ValidationError on any structural problem with the model."""
    if not model.nodes:
        raise ValidationError("model has no nodes")
    if not model.elements:
        raise ValidationError("model has no elements")

    _check_unique_ids("node", [n.id for n in model.nodes])
    _check_unique_ids("element", [e.id for e in model.elements])
    _check_unique_ids("material", [m.id for m in model.materials])

    node_ids = {n.id for n in model.nodes}
    material_ids = {m.id for m in model.materials}

    for m in model.materials:
        if not (m.E > 0.0):
            raise ValidationError(
                f"material {m.id} has non-positive Young's modulus",
                details={"material_id": m.id, "E": m.E},
            )
        if not (-1.0 < m.nu < 0.5):
            raise ValidationError(
                f"material {m.id} has Poisson's ratio outside (-1, 0.5)",
                details={"material_id": m.id, "nu": m.nu},
            )

    referenced: set[int] = set()
    for el in model.elements:
        for nid in el.nodes:
            if nid not in node_ids:
                raise ValidationError(
                    f"element {el.id} references undefined node {nid}",
                    details={"element_id": el.id, "node_id": nid},
                )
            referenced.add(nid)
        if el.material_id not in material_ids:
            raise ValidationError(
                f"element {el.id} references undefined material "
                f"{el.material_id}",
                details={"element_id": el.id, "material_id": el.material_id},
            )

    _check_element_geometry(model)

    dangling = sorted(node_ids - referenced)
    if dangling:
        raise ValidationError(
            f"dangling nodes not used by any element: {dangling}",
            details={"node_ids": dangling},
        )

    seen_bc: set[tuple[int, str]] = set()
    for bc in model.displacement_bcs:
        if bc.node_id not in node_ids:
            raise ValidationError(
                f"displacement BC references undefined node {bc.node_id}",
                details={"node_id": bc.node_id},
            )
        key = (bc.node_id, bc.direction)
        if key in seen_bc:
            raise ValidationError(
                f"conflicting displacement BCs on node {bc.node_id} "
                f"direction {bc.direction}",
                details={"node_id": bc.node_id, "direction": bc.direction},
            )
        seen_bc.add(key)

    for load in model.point_loads:
        if load.node_id not in node_ids:
            raise ValidationError(
                f"point load references undefined node {load.node_id}",
                details={"node_id": load.node_id},
            )
