"""Distinguishable error types for the FEA service.

Every failure carries a stable machine-readable ``code`` so that clients can
tell apart validation failures, limit rejections, solver failures, etc.
"""

from __future__ import annotations


class FEAError(Exception):
    """Base class for all service errors."""

    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class ValidationError(FEAError):
    """Input model failed structural validation.

    Raised for degenerate elements, dangling nodes, undefined materials,
    duplicate ids, invalid boundary conditions, etc. Never silently fixed.
    """

    code = "validation_error"


class LimitExceededError(FEAError):
    """Job rejected before start because it exceeds configured size limits."""

    code = "limit_exceeded"


class SingularMatrixError(FEAError):
    """Global stiffness matrix is singular (mechanism / insufficient supports)."""

    code = "singular_matrix"


class EquilibriumError(FEAError):
    """Solved solution violates global force equilibrium beyond tolerance."""

    code = "equilibrium_violation"


class JobNotFoundError(FEAError):
    code = "job_not_found"


class JobStateError(FEAError):
    """Illegal state transition, e.g. cancelling a finished job."""

    code = "job_state_error"


class JobTimeoutError(FEAError):
    code = "job_timeout"


class JobCancelledError(FEAError):
    code = "job_cancelled"
