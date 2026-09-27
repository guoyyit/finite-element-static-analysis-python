"""Element level computations for supported plane elements (Q4).

Only the 4-node bilinear isoparametric quadrilateral is implemented, in plane
stress (``CPS4``) and plane strain (``CPE4``) variants.  2x2 Gauss integration
is used for the stiffness; element stress is evaluated once at the centroid.
"""

from __future__ import annotations

import math

import numpy as np

from fem.models import ELEMENT_PLANE_STRAIN, ELEMENT_PLANE_STRESS

# 2x2 Gauss points / weights on the parent bi-unit square.
_GP = 1.0 / math.sqrt(3.0)
GAUSS_POINTS = ((-_GP, -_GP, 1.0), (_GP, -_GP, 1.0),
                (_GP, _GP, 1.0), (-_GP, _GP, 1.0))
_CENTER = (0.0, 0.0)


def _cross2(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _polygon_signed_area(coords: np.ndarray) -> float:
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def q4_is_degenerate(coords: np.ndarray, area_eps: float = 1e-12) -> bool:
    """Return True when the quadrilateral is geometrically degenerate.

    Rejected shapes (never repaired):
      * duplicate corner nodes;
      * a corner triangle with non-positive oriented area (fold / inversion);
      * a self-intersecting (bow-tie) ordering;
      * a total signed area not larger than ``area_eps``.
    """
    c = np.asarray(coords, dtype=float)
    if c.shape != (4, 2):
        return True

    # Duplicate nodes.
    for i in range(4):
        for j in range(i + 1, 4):
            if c[i, 0] == c[j, 0] and c[i, 1] == c[j, 1]:
                return True

    # Every consecutive corner triangle must be strictly positively oriented;
    # a non-positive value means a folded / inverted / collinear corner.
    area = 0.0
    for i in range(4):
        p = c[(i + 1) % 4] - c[i]
        q = c[(i + 2) % 4] - c[(i + 1) % 4]
        tri = _cross2(float(p[0]), float(p[1]), float(q[0]), float(q[1]))
        if tri <= 0.0:
            return True
        area += tri

    # Reject numerically zero-area elements without any snapping/correction.
    if 0.5 * area <= area_eps:
        return True

    if _polygon_signed_area(c) <= area_eps:
        return True
    return False


def plane_constitutive(young: float, poisson: float, thickness: float,
                       kind: str) -> np.ndarray:
    """Return the 3x3 plane stress or plane strain constitutive matrix D.

    ``thickness`` is accepted for symmetry of call sites but only plane stress
    multiplies stiffness by it during integration (see :func:`q4_stiffness`).
    """
    if young <= 0.0:
        raise ValueError("Young's modulus must be positive")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("Poisson's ratio must be in (-1, 0.5)")

    if kind == ELEMENT_PLANE_STRESS:
        factor = young / (1.0 - poisson * poisson)
        dmat = factor * np.array(
            [[1.0, poisson, 0.0],
             [poisson, 1.0, 0.0],
             [0.0, 0.0, (1.0 - poisson) / 2.0]],
            dtype=float,
        )
    elif kind == ELEMENT_PLANE_STRAIN:
        factor = young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        dmat = factor * np.array(
            [[1.0 - poisson, poisson, 0.0],
             [poisson, 1.0 - poisson, 0.0],
             [0.0, 0.0, (1.0 - 2.0 * poisson) / 2.0]],
            dtype=float,
        )
    else:  # pragma: no cover - guarded by validation
        raise ValueError(f"unsupported element type {kind!r}")
    return dmat


def _shape_derivatives(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (shape functions N(4), parent derivatives dN/dxi (4x2))."""
    shapes = 0.25 * np.array(
        [(1 - xi) * (1 - eta),
         (1 + xi) * (1 - eta),
         (1 + xi) * (1 + eta),
         (1 - xi) * (1 + eta)],
        dtype=float,
    )
    dndxi = 0.25 * np.array(
        [[-(1 - eta), -(1 - xi)],
         [ (1 - eta), -(1 + xi)],
         [ (1 + eta),  (1 + xi)],
         [-(1 + eta),  (1 - xi)]],
        dtype=float,
    )
    return shapes, dndxi


def _strain_displacement(xi: float, eta: float, coords: np.ndarray
                         ) -> tuple[np.ndarray, float]:
    """Return (B 3x8, detJ) at a parent point."""
    _, dndxi = _shape_derivatives(xi, eta)
    jac = dndxi.T @ coords                     # 2x2
    det_j = float(np.linalg.det(jac))
    dndxy = np.linalg.solve(jac, dndxi.T).T    # dN/dx, dN/dy per node
    bmat = np.zeros((3, 8), dtype=float)
    for i in range(4):
        bmat[0, 2 * i] = dndxy[i, 0]
        bmat[1, 2 * i + 1] = dndxy[i, 1]
        bmat[2, 2 * i] = dndxy[i, 1]
        bmat[2, 2 * i + 1] = dndxy[i, 0]
    return bmat, det_j


def q4_stiffness(coords: np.ndarray, dmat: np.ndarray, thickness: float,
                 kind: str) -> np.ndarray:
    """Return the 8x8 stiffness matrix of a 4-node bilinear quadrilateral."""
    ke = np.zeros((8, 8), dtype=float)
    for xi, eta, weight in GAUSS_POINTS:
        bmat, det_j = _strain_displacement(xi, eta, coords)
        # Plane strain is a unit-thickness slice; plane stress scales by t.
        t = thickness if kind == ELEMENT_PLANE_STRESS else 1.0
        bdb = (bmat.T @ dmat @ bmat) * (weight * det_j * t)
        ke += bdb
    # Force exact symmetry to avoid round-off asymmetry entering assembly.
    return 0.5 * (ke + ke.T)


def q4_stress(coords: np.ndarray, dmat: np.ndarray,
              element_displacements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(stress_3, B_center)`` centroid stress for a Q4."""
    bmat, _ = _strain_displacement(_CENTER[0], _CENTER[1], coords)
    strain = bmat @ element_displacements
    stress = dmat @ strain
    return stress, bmat
