"""Static finite element solver: assemble, constrain, solve, verify balance.

Pipeline: element stiffness assembly -> free/prescribed DOF partition ->
dense symmetric elimination -> reaction recovery -> centroid stresses ->
global force-balance residual check against the configured tolerance.

Checkpointing: the assembly phase is resumable.  ``assemble`` returns an
:class:`AssembledSystem` after every ``checkpoint_interval`` elements and can
resume from a previously persisted partial system, so an interrupted large
job continues from the last checkpoint instead of starting over.  The linear
solve and equilibrium verification always run over the *complete* system, so
resumed results are identical to uninterrupted runs.

Determinism: element iteration order, DOF ordering and numpy elimination are
all deterministic for a given validated model; floating-point output is thus
reproducible across runs for identical input and parameters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from fem.config import ServiceConfig
from fem.errors import (
    EquilibriumNotSatisfiedError,
    JobCancelledError,
    JobTimeoutError,
    SingularMatrixError,
)
from fem.mesh import q4_stiffness, q4_stress
from fem.models import (
    SOLVER_VERSION,
    ElementStress,
    NodeDisplacement,
    Reaction,
    ResidualReport,
    SolveResults,
)
from fem.validation import NormalizedModel


@dataclass
class AssembledSystem:
    """Resumable assembly state."""

    kglob: np.ndarray
    assembled_through: int          # number of elements already accumulated

    def to_payload(self) -> dict:
        return {"assembled_through": int(self.assembled_through),
                "kglob": self.kglob.tolist()}

    @classmethod
    def from_payload(cls, payload: dict, ndof: int) -> "AssembledSystem":
        kglob = np.asarray(payload["kglob"], dtype=float)
        if kglob.shape != (ndof, ndof):
            raise ValueError("checkpoint stiffness has wrong shape")
        return cls(kglob=kglob,
                   assembled_through=int(payload["assembled_through"]))


def _check_deadline(start: float, timeout: float, cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelledError("job cancelled during solve")
    if timeout and time.monotonic() - start > timeout:
        raise JobTimeoutError(
            f"solve exceeded {timeout:.3f}s budget",
            details={"timeout_seconds": timeout})


def assemble(
    model: NormalizedModel,
    config: ServiceConfig,
    *,
    start_monotonic: float | None = None,
    cancel_event=None,
    partial: AssembledSystem | None = None,
    checkpoint_cb=None,
) -> AssembledSystem:
    """Assemble (or resume assembling) the global stiffness matrix.

    Invokes ``checkpoint_cb(AssembledSystem)`` at fixed element intervals and
    at completion.  Resuming from ``partial`` skips already accumulated
    elements and yields the same final matrix as a fresh run, because each
    element's contribution is added exactly once in a fixed order.
    """
    ndof, ne = model.ndof, model.ne
    start = start_monotonic if start_monotonic is not None else time.monotonic()
    if partial is None:
        kglob = np.zeros((ndof, ndof), dtype=float)
        begin = 0
    else:
        kglob = np.array(partial.kglob, copy=True)
        begin = partial.assembled_through
        if not (0 <= begin <= ne):
            raise ValueError("checkpoint beyond element count")

    interval = max(1, config.checkpoint_interval or ne)
    for k in range(begin, ne):
        nodes = model.connectivity[k]
        ke = q4_stiffness(
            model.coordinates[nodes],
            model.dmat_by_element[k],
            float(model.thickness_by_element[k]),
            model.element_types[k],
        )
        gdof = np.empty(8, dtype=np.int64)
        gdof[0::2] = 2 * nodes
        gdof[1::2] = 2 * nodes + 1
        np.add.at(kglob, (gdof[:, None], gdof[None, :]), ke)
        done = k + 1
        if done == ne or done % interval == 0:
            _check_deadline(start, config.job_timeout_seconds, cancel_event)
            if checkpoint_cb is not None:
                # Hand over an independent copy: the working buffer keeps
                # accumulating and must not mutate an already-emitted
                # checkpoint (which a resume may rely on).
                checkpoint_cb(AssembledSystem(kglob=kglob.copy(),
                                              assembled_through=done))
    return AssembledSystem(kglob=kglob, assembled_through=ne)


def solve(
    model: NormalizedModel,
    config: ServiceConfig,
    *,
    model_hash: str,
    progress_cb=None,
    cancel_event=None,
    assembly: AssembledSystem | None = None,
    start_monotonic: float | None = None,
) -> SolveResults:
    """Solve a static linear FEM system deterministically.

    A pre-assembled (possibly resumed) system may be supplied; otherwise the
    full matrix is assembled here.  The result always derives from a complete
    stiffness matrix and always passes the equilibrium residual check.
    """
    start = start_monotonic if start_monotonic is not None else time.monotonic()
    nn, ne, ndof = model.nn, model.ne, model.ndof
    timeout = config.job_timeout_seconds

    def progress(p: float) -> None:
        if progress_cb is not None:
            progress_cb(float(p))

    _check_deadline(start, timeout, cancel_event)

    # ---- assembly (resumable) --------------------------------------------
    if assembly is None or assembly.assembled_through < ne:
        system = assemble(
            model, config,
            start_monotonic=start, cancel_event=cancel_event,
            partial=assembly,
            checkpoint_cb=lambda sys_: progress(
                0.6 * sys_.assembled_through / ne))
    else:
        system = assembly
    kglob = 0.5 * (system.kglob + system.kglob.T)
    u = np.zeros(ndof, dtype=float)
    reactions = np.zeros(ndof, dtype=float)

    # ---- partition DOFs ---------------------------------------------------
    fixed = model.prescribed_dof
    fixed_mask = np.zeros(ndof, dtype=bool)
    fixed_mask[fixed] = True
    free = np.nonzero(~fixed_mask)[0]

    u[fixed] = model.prescribed_values
    fext = model.force_vector

    # Reduced system: Kff uf = ff - Kfp up
    rhs = fext[free] - kglob[np.ix_(free, fixed)] @ u[fixed]
    kff = kglob[np.ix_(free, free)]
    progress(0.7)
    _check_deadline(start, timeout, cancel_event)

    # ---- deterministic elimination ---------------------------------------
    try:
        uf = np.linalg.solve(kff, rhs)
    except np.linalg.LinAlgError as exc:
        raise SingularMatrixError(
            "stiffness matrix is singular (insufficient constraints or "
            "mechanism)") from exc
    u[free] = uf
    progress(0.9)
    _check_deadline(start, timeout, cancel_event)

    # ---- reactions on constrained DOFs -----------------------------------
    reactions[fixed] = kglob[fixed, :] @ u - fext[fixed]

    # ---- equilibrium residual over FREE dofs ------------------------------
    residual_free = kglob[np.ix_(free, free)] @ u[free] \
        + kglob[np.ix_(free, fixed)] @ u[fixed] - fext[free]
    scale = max(float(np.max(np.abs(fext))), config.residual_force_scale, 1e-30)
    max_abs = float(np.max(np.abs(residual_free))) if free.size else 0.0
    worst_local = int(np.argmax(np.abs(residual_free))) if free.size else 0
    worst_gdof = int(free[worst_local]) if free.size else int(fixed[0])
    tol = config.equilibrium_tolerance
    # Tolerance is an absolute force threshold scaled by applied-force size.
    tol_abs = tol * scale
    if max_abs > tol_abs:
        raise EquilibriumNotSatisfiedError(
            f"max free-DOF force residual {max_abs:.6e} exceeds tolerance "
            f"{tol_abs:.6e}",
            details={"max_residual": max_abs, "tolerance": tol_abs})

    # ---- stress recovery at element centroids -----------------------------
    stress_rows: list[ElementStress] = []
    for k in range(ne):
        nodes = model.connectivity[k]
        gdof = np.empty(8, dtype=np.int64)
        gdof[0::2] = 2 * nodes
        gdof[1::2] = 2 * nodes + 1
        s3, _ = q4_stress(model.coordinates[nodes],
                          model.dmat_by_element[k], u[gdof])
        sxx, syy, sxy = float(s3[0]), float(s3[1]), float(s3[2])
        if model.element_types[k] == "CPE4":
            dmat = model.dmat_by_element[k]
            nu = float(dmat[0, 1] / (dmat[0, 0] + dmat[0, 1]))
            szz = nu * (sxx + syy)
        else:
            szz = 0.0
        vm = float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2
                                  + (szz - sxx) ** 2 + 6.0 * sxy * sxy)))
        stress_rows.append(ElementStress(
            element=int(model.element_ids[k]),
            sxx=sxx, syy=syy, sxy=sxy, von_mises=vm))

    # ---- output rows keyed by original ids --------------------------------
    disp_rows = [
        NodeDisplacement(node=int(model.original_node_id[i]),
                         ux=float(u[2 * i]), uy=float(u[2 * i + 1]))
        for i in range(nn)
    ]
    reaction_rows: list[Reaction] = []
    fixed_nodes = sorted({int(g // 2) for g in fixed})
    for compact in fixed_nodes:
        orig = int(model.original_node_id[compact])
        reaction_rows.append(Reaction(
            node=orig,
            fx=float(reactions[2 * compact]),
            fy=float(reactions[2 * compact + 1])))

    results = SolveResults(
        solver_version=SOLVER_VERSION,
        model_hash=model_hash,
        displacements=disp_rows,
        stresses=stress_rows,
        reactions=reaction_rows,
        residual=ResidualReport(
            max_abs_residual=max_abs,
            node=int(model.original_node_id[worst_gdof // 2]),
            dof=worst_gdof % 2,
            tolerance=tol_abs),
    )
    progress(1.0)
    return results
