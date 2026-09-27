"""Deterministic plane-stress / plane-strain Q4 finite element solver.

Determinism: DOFs are ordered by sorted node id, elements processed in
sorted id order, dense float64 linear algebra via numpy. Identical inputs
therefore produce bit-identical outputs across runs on the same platform.
"""

from __future__ import annotations

import numpy as np

from .errors import EquilibriumError, SingularMatrixError
from .models import (
    AnalysisType,
    ElementResult,
    ModelInput,
    NodeResult,
    Reaction,
    SolveResult,
    StressComponents,
)

_GAUSS_PTS = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))


def _constitutive(analysis: AnalysisType, E: float, nu: float) -> np.ndarray:
    if analysis is AnalysisType.PLANE_STRESS:
        c = E / (1.0 - nu * nu)
        return c * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
        )
    # plane strain
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array(
        [
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
        ]
    )


def _shape_derivs(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """dN/d(xi, eta) for the bilinear Q4, nodes ordered (-,-),(+,-),(+,+),(-,+)."""
    dN_dxi = 0.25 * np.array(
        [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
    )
    dN_deta = 0.25 * np.array(
        [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
    )
    return dN_dxi, dN_deta


def _b_matrix(dN_dx: np.ndarray, dN_dy: np.ndarray) -> np.ndarray:
    B = np.zeros((3, 8))
    B[0, 0::2] = dN_dx
    B[1, 1::2] = dN_dy
    B[2, 0::2] = dN_dy
    B[2, 1::2] = dN_dx
    return B


def _element_stiffness(
    xy: np.ndarray, D: np.ndarray, thickness: float
) -> np.ndarray:
    """8x8 stiffness of one Q4 via 2x2 Gauss quadrature."""
    Ke = np.zeros((8, 8))
    for xi in _GAUSS_PTS:
        for eta in _GAUSS_PTS:
            dN_dxi, dN_deta = _shape_derivs(xi, eta)
            J = np.array(
                [
                    [dN_dxi @ xy[:, 0], dN_deta @ xy[:, 0]],
                    [dN_dxi @ xy[:, 1], dN_deta @ xy[:, 1]],
                ]
            )
            detJ = float(np.linalg.det(J))
            if detJ <= 0.0:
                # Validation guarantees convexity; a non-positive Jacobian
                # here means clockwise ordering, which we normalize away by
                # using the absolute value below. Keep it defensive anyway.
                detJ = abs(detJ)
            invJ = np.linalg.inv(J)
            grads = invJ @ np.vstack([dN_dxi, dN_deta])
            B = _b_matrix(grads[0], grads[1])
            Ke += B.T @ D @ B * detJ * thickness
    return Ke


def _stress_at_center(
    xy: np.ndarray, D: np.ndarray, ue: np.ndarray
) -> np.ndarray:
    dN_dxi, dN_deta = _shape_derivs(0.0, 0.0)
    J = np.array(
        [
            [dN_dxi @ xy[:, 0], dN_deta @ xy[:, 0]],
            [dN_dxi @ xy[:, 1], dN_deta @ xy[:, 1]],
        ]
    )
    invJ = np.linalg.inv(J)
    grads = invJ @ np.vstack([dN_dxi, dN_deta])
    B = _b_matrix(grads[0], grads[1])
    return D @ B @ ue


def solve_model(model: ModelInput) -> SolveResult:
    """Assemble, solve and post-process. Deterministic for identical inputs."""
    node_ids = sorted(n.id for n in model.nodes)
    dof_of = {nid: (2 * i, 2 * i + 1) for i, nid in enumerate(node_ids)}
    ndof = 2 * len(node_ids)

    coords = {n.id: np.array([n.x, n.y], dtype=np.float64) for n in model.nodes}
    materials = {m.id: m for m in model.materials}
    D_of = {
        m.id: _constitutive(model.analysis_type, m.E, m.nu)
        for m in model.materials
    }

    K = np.zeros((ndof, ndof), dtype=np.float64)
    F = np.zeros(ndof, dtype=np.float64)

    elements = sorted(model.elements, key=lambda e: e.id)
    for el in elements:
        xy = np.array([coords[nid] for nid in el.nodes])
        Ke = _element_stiffness(xy, D_of[el.material_id], el.thickness)
        dofs: list[int] = []
        for nid in el.nodes:
            dofs.extend(dof_of[nid])
        idx = np.ix_(dofs, dofs)
        K[idx] += Ke

    for load in model.point_loads:
        dx, dy = dof_of[load.node_id]
        F[dx] += load.fx
        F[dy] += load.fy

    # Prescribed displacements via free/constrained partitioning (exact).
    prescribed: dict[int, float] = {}
    for bc in model.displacement_bcs:
        dx, dy = dof_of[bc.node_id]
        prescribed[dx if bc.direction == "x" else dy] = bc.value

    if not prescribed:
        raise SingularMatrixError(
            "model has no displacement boundary conditions; "
            "the stiffness matrix is singular (rigid body modes)"
        )

    fixed = sorted(prescribed)
    free = [d for d in range(ndof) if d not in prescribed]
    u = np.zeros(ndof, dtype=np.float64)
    for d in fixed:
        u[d] = prescribed[d]

    if free:
        Kff = K[np.ix_(free, free)]
        rhs = F[free] - K[np.ix_(free, fixed)] @ u[fixed]
        try:
            u[free] = np.linalg.solve(Kff, rhs)
        except np.linalg.LinAlgError as exc:
            raise SingularMatrixError(
                "stiffness matrix is singular; check supports and mechanisms"
            ) from exc
        if not np.all(np.isfinite(u[free])):
            raise SingularMatrixError(
                "stiffness matrix is numerically singular; "
                "check supports and mechanisms"
            )

    # Global equilibrium residual on free DOFs (reactions balance the rest).
    residual = K @ u - F
    max_res = float(np.max(np.abs(residual[free]))) if free else 0.0
    if max_res > model.options.equilibrium_tol:
        # A huge residual on a supposedly solvable system means the matrix
        # was numerically singular (LAPACK does not always raise). Confirm
        # via the condition number so callers get a distinguishable error.
        if free and np.linalg.cond(Kff) > 1.0e12:
            raise SingularMatrixError(
                "stiffness matrix is ill-conditioned (mechanism or "
                "insufficient supports)",
                details={"max_residual": max_res},
            )
        raise EquilibriumError(
            f"force equilibrium residual {max_res:.3e} exceeds tolerance "
            f"{model.options.equilibrium_tol:.3e}",
            details={"max_residual": max_res},
        )

    displacements = [
        NodeResult(node_id=nid, ux=float(u[dx]), uy=float(u[dy]))
        for nid, (dx, dy) in sorted(dof_of.items())
    ]

    reactions = []
    fixed_by_node: dict[int, dict[str, float]] = {}
    for d in fixed:
        nid = node_ids[d // 2]
        comp = "rx" if d % 2 == 0 else "ry"
        fixed_by_node.setdefault(nid, {"rx": 0.0, "ry": 0.0})[comp] = float(
            residual[d]
        )
    for nid in sorted(fixed_by_node):
        r = fixed_by_node[nid]
        reactions.append(Reaction(node_id=nid, rx=r["rx"], ry=r["ry"]))

    element_results = []
    for el in elements:
        xy = np.array([coords[nid] for nid in el.nodes])
        ue = np.array(
            [u[d] for nid in el.nodes for d in dof_of[nid]], dtype=np.float64
        )
        s = _stress_at_center(xy, D_of[el.material_id], ue)
        sxx, syy, txy = (float(v) for v in s)
        if model.analysis_type is AnalysisType.PLANE_STRAIN:
            nu = materials[el.material_id].nu
            szz = nu * (sxx + syy)
        else:
            szz = 0.0
        von_mises = float(
            np.sqrt(
                0.5
                * (
                    (sxx - syy) ** 2
                    + (syy - szz) ** 2
                    + (szz - sxx) ** 2
                    + 6.0 * txy**2
                )
            )
        )
        element_results.append(
            ElementResult(
                element_id=el.id,
                stress=StressComponents(sxx=sxx, syy=syy, txy=txy),
                von_mises=von_mises,
            )
        )

    return SolveResult(
        displacements=displacements,
        element_results=element_results,
        reactions=reactions,
        max_equilibrium_residual=max_res,
    )
