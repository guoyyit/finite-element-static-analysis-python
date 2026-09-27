"""Service configuration: defaults, resource limits and residual threshold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceConfig:
    # Input size guard rails (a job exceeding any limit is rejected up front)
    max_nodes: int = 100_000
    max_elements: int = 100_000
    max_jobs: int = 256
    # Rough memory ceiling (bytes) for assembled global system
    max_memory_bytes: int = 512 * 1024 * 1024
    # Solver wall time budget per job (seconds); 0 disables
    job_timeout_seconds: float = 60.0
    # Equilibrium residual: max |K u - f| over *unconstrained* DOFs.
    equilibrium_tolerance: float = 1e-6
    # Free-DOF residual is compared against max(residual, scale) so zero-load
    # and rigid-body-free models are checked with an absolute floor.
    residual_force_scale: float = 1.0
    # Parallel worker threads running blocking FEM solves
    worker_threads: int = 4
    # Where job checkpoints / results live
    storage_dir: str = ".fem_jobs"
    # Checkpoint progress interval in assembled elements (0 -> only at end)
    checkpoint_interval: int = 4096

    def __post_init__(self) -> None:
        if self.max_nodes <= 0 or self.max_elements <= 0:
            raise ValueError("size limits must be positive")
        if self.max_jobs <= 0 or self.worker_threads <= 0:
            raise ValueError("job/worker limits must be positive")
        if self.max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be positive")
        if self.equilibrium_tolerance <= 0:
            raise ValueError("equilibrium_tolerance must be positive")
        if self.job_timeout_seconds < 0:
            raise ValueError("job_timeout_seconds must be >= 0")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be >= 0")

    def memory_estimate_bytes(self, ndof: int) -> int:
        """Upper-bound memory estimate for a dense symmetric solve.

        Accounts for the global stiffness matrix (ndof x ndof float64), the
        reduced free-DOF working matrix and a constant number of per-DOF
        vectors / integer index arrays.
        """
        matrix = 8 * ndof * ndof          # global K
        workspace = 256 * ndof            # RHS, solution, reactions, indexes
        return matrix + workspace
