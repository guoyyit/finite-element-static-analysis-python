"""Strict input validation and canonical normalization of FEM models.

Validation never repairs input: degenerate elements, dangling nodes and
undefined materials are rejected with distinct error codes.  The normalized
form uses compact 0-based node indexing and contiguously packed numpy arrays.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

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
from fem.mesh import q4_is_degenerate
from fem import models as models_mod
from fem.models import (
    SUPPORTED_ELEMENT_TYPES,
    ModelInput,
)


@dataclass(frozen=True)
class NormalizedModel:
    """Contiguous-array view of a validated model, indexed by compact ids."""

    coordinates: np.ndarray          # (nn, 2)
    connectivity: np.ndarray         # (ne, 4) compact node indices
    element_types: tuple[str, ...]
    thickness_by_element: np.ndarray
    dmat_by_element: tuple[np.ndarray, ...]
    prescribed_dof: np.ndarray      # constrained DOF global indices
    prescribed_values: np.ndarray
    force_vector: np.ndarray        # full ndof external force vector
    original_node_id: np.ndarray     # compact index -> original id
    element_ids: np.ndarray          # compact elem index -> original id
    nn: int
    ne: int

    @property
    def ndof(self) -> int:
        return 2 * self.nn

    def canonical_hash(self) -> str:
        return self._canonical_hash

    # assigned in validate_model (frozen dataclass via object.__setattr__)
    _canonical_hash: str = ""


def _fail(exc, message: str, **details):
    raise exc(message, details=details) if details else exc(message)


def _canonical_hash(model: ModelInput) -> str:
    """Deterministic hash over the *semantic* model content.

    Independent of list ordering (entities are sorted by their integer ids)
    and of JSON whitespace / key order.  Bound to the input schema and solver
    versions so cached results cannot outlive a semantics change.
    """
    payload = {
        "solver_version": models_mod.SOLVER_VERSION,
        "input_version": model.version,
        "nodes": sorted(
            (n.id, float(n.x), float(n.y)) for n in model.nodes
        ),
        "elements": sorted(
            (e.id, e.type, list(e.nodes), e.material) for e in model.elements
        ),
        "materials": sorted(
            (m.name, float(m.youngs_modulus),
             float(m.poisson_ratio), float(m.thickness))
            for m in model.materials
        ),
        "boundary_conditions": sorted(
            (b.node, b.dof, float(b.value)) for b in model.boundary_conditions
        ),
        "loads": sorted(
            (ld.node, ld.dof, float(ld.value)) for ld in model.loads
        ),
        "options": {
            "tolerance": model.options.equilibrium_tolerance,
            "timeout": model.options.timeout_seconds,
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_model(model: ModelInput, config: ServiceConfig) -> NormalizedModel:
    """Validate raw API input and produce a normalized model."""
    if not model.nodes:
        _fail(InvalidNodeError, "model must contain at least one node")
    if not model.elements:
        _fail(InvalidElementError, "model must contain at least one element")

    nn, ne = len(model.nodes), len(model.elements)
    if nn > config.max_nodes:
        _fail(InvalidNodeError,
              f"node count {nn} exceeds limit {config.max_nodes}",
              count=nn, limit=config.max_nodes)
    if ne > config.max_elements:
        _fail(InvalidElementError,
              f"element count {ne} exceeds limit {config.max_elements}",
              count=ne, limit=config.max_elements)
    ndof = 2 * nn
    if config.memory_estimate_bytes(ndof) > config.max_memory_bytes:
        _fail(InvalidNodeError,
              f"estimated memory for {ndof} DOFs exceeds "
              f"{config.max_memory_bytes} bytes",
              ndof=ndof, limit_bytes=config.max_memory_bytes)

    # ---- nodes: unique ids, finite coordinates ---------------------------
    seen: set[int] = set()
    coordinates: dict[int, tuple[float, float]] = {}
    for n in model.nodes:
        if n.id in seen:
            _fail(InvalidNodeError, f"duplicate node id {n.id}", node_id=n.id)
        seen.add(n.id)
        if not (np.isfinite(n.x) and np.isfinite(n.y)):
            _fail(InvalidNodeError,
                  f"node {n.id} has non-finite coordinates", node_id=n.id)
        coordinates[n.id] = (float(n.x), float(n.y))

    # ---- materials --------------------------------------------------------
    materials: dict[str, object] = {}
    for m in model.materials:
        if m.name in materials:
            _fail(UndefinedMaterialError,
                  f"duplicate material name {m.name!r}", material=m.name)
        if not np.isfinite(m.youngs_modulus) or m.youngs_modulus <= 0.0:
            _fail(UndefinedMaterialError,
                  f"material {m.name!r}: Young's modulus must be positive",
                  material=m.name)
        if not np.isfinite(m.poisson_ratio) or not (-1.0 < m.poisson_ratio < 0.5):
            _fail(UndefinedMaterialError,
                  f"material {m.name!r}: Poisson's ratio must be in (-1, 0.5)",
                  material=m.name)
        if not np.isfinite(m.thickness) or m.thickness <= 0.0:
            _fail(UndefinedMaterialError,
                  f"material {m.name!r}: thickness must be positive",
                  material=m.name)
        materials[m.name] = m

    # ---- elements ---------------------------------------------------------
    elem_seen: set[int] = set()
    used_nodes: set[int] = set()
    compact_id = {nid: i for i, nid in enumerate(sorted(coordinates))}
    original_node_id = np.array(sorted(coordinates), dtype=np.int64)
    conn = np.zeros((ne, 4), dtype=np.int64)
    element_types: list[str] = []
    thicknesses = np.ones(ne, dtype=float)
    dmats: list[np.ndarray] = []
    element_ids = np.zeros(ne, dtype=np.int64)

    # Local import avoids an import cycle (mesh is dependency free).
    from fem.mesh import plane_constitutive

    for k, e in enumerate(sorted(model.elements, key=lambda x: x.id)):
        if e.id in elem_seen:
            _fail(InvalidElementError,
                  f"duplicate element id {e.id}", element_id=e.id)
        elem_seen.add(e.id)
        element_ids[k] = e.id

        if e.type not in SUPPORTED_ELEMENT_TYPES:
            _fail(InvalidElementError,
                  f"element {e.id}: unsupported type {e.type!r}",
                  element_id=e.id, type=e.type,
                  supported=sorted(SUPPORTED_ELEMENT_TYPES))
        element_types.append(e.type)

        if e.material not in materials:
            _fail(UndefinedMaterialError,
                  f"element {e.id} references undefined material "
                  f"{e.material!r}",
                  element_id=e.id, material=e.material)
        mat = materials[e.material]
        thicknesses[k] = float(mat.thickness)
        dmats.append(plane_constitutive(
            float(mat.youngs_modulus), float(mat.poisson_ratio),
            float(mat.thickness), e.type))

        if len(e.nodes) != 4 or len(set(e.nodes)) != 4:
            _fail(InvalidElementError,
                  f"element {e.id} must reference 4 distinct nodes",
                  element_id=e.id, nodes=list(e.nodes))
        for nid in e.nodes:
            if nid not in coordinates:
                _fail(InvalidElementError,
                      f"element {e.id} references undefined node {nid}",
                      element_id=e.id, node_id=nid)
            used_nodes.add(nid)

        coords = np.array([coordinates[nid] for nid in e.nodes], dtype=float)
        if q4_is_degenerate(coords):
            _fail(DegenerateElementError,
                  f"element {e.id} is degenerate (zero/negative area, "
                  f"folded or duplicate corner)",
                  element_id=e.id, nodes=list(e.nodes))
        conn[k] = [compact_id[nid] for nid in e.nodes]

    # ---- dangling nodes: defined but unused by any element ----------------
    dangling = sorted(set(coordinates) - used_nodes)
    if dangling:
        _fail(DanglingNodeError,
              f"{len(dangling)} node(s) are not connected to any element",
              nodes=dangling[:20], count=len(dangling))

    # ---- boundary conditions ---------------------------------------------
    bc_dof: set[int] = set()
    prescribed_dof_list: list[int] = []
    prescribed_value_list: list[float] = []
    for b in model.boundary_conditions:
        if b.node not in coordinates:
            _fail(InvalidBoundaryConditionError,
                  f"boundary condition references undefined node {b.node}",
                  node_id=b.node)
        if not np.isfinite(b.value):
            _fail(InvalidBoundaryConditionError,
                  f"non-finite prescribed displacement on node {b.node}",
                  node_id=b.node)
        gdof = 2 * compact_id[b.node] + b.dof
        if gdof in bc_dof:
            _fail(InvalidBoundaryConditionError,
                  f"duplicate boundary condition on node {b.node} dof {b.dof}",
                  node_id=b.node, dof=b.dof)
        bc_dof.add(gdof)
        prescribed_dof_list.append(gdof)
        prescribed_value_list.append(float(b.value))

    # ---- loads ------------------------------------------------------------
    force_vector = np.zeros(ndof, dtype=float)
    load_seen: dict[int, float] = {}
    for ld in model.loads:
        if ld.node not in coordinates:
            _fail(InvalidLoadError,
                  f"load references undefined node {ld.node}", node_id=ld.node)
        if not np.isfinite(ld.value):
            _fail(InvalidLoadError,
                  f"non-finite load on node {ld.node}", node_id=ld.node)
        gdof = 2 * compact_id[ld.node] + ld.dof
        load_seen[gdof] = load_seen.get(gdof, 0.0) + float(ld.value)
    for gdof, value in load_seen.items():
        force_vector[gdof] = value

    if not bc_dof:
        _fail(UnconstrainedModelError,
              "model has no boundary conditions; rigid body motion would "
              "make the system singular")

    prescribed_dof = np.array(sorted(prescribed_dof_list), dtype=np.int64)
    prescribed_values = np.array(
        [v for _, v in sorted(zip(prescribed_dof_list, prescribed_value_list))],
        dtype=float,
    )
    coords_array = np.array(
        [coordinates[nid] for nid in original_node_id], dtype=float)

    normalized = NormalizedModel(
        coordinates=coords_array,
        connectivity=conn,
        element_types=tuple(element_types),
        thickness_by_element=thicknesses,
        dmat_by_element=tuple(dmats),
        prescribed_dof=prescribed_dof,
        prescribed_values=prescribed_values,
        force_vector=force_vector,
        original_node_id=original_node_id,
        element_ids=element_ids,
        nn=nn,
        ne=ne,
    )
    object.__setattr__(normalized, "_canonical_hash", _canonical_hash(model))
    return normalized
