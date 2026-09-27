"""Distinguishable domain error types for the FEM service.

Each validation / execution failure maps to a stable ``code`` so callers can
programmatically distinguish error conditions (never silently corrected).
"""

from __future__ import annotations


class FemError(Exception):
    """Base class for all service errors with a stable machine readable code."""

    code = "fem_error"

    def __init__(self, message: str = "", *, details: dict | None = None) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details: dict = details or {}


class ValidationError(FemError):
    code = "validation_error"


class InvalidNodeError(ValidationError):
    code = "invalid_node"


class InvalidElementError(ValidationError):
    code = "invalid_element"


class DegenerateElementError(ValidationError):
    code = "degenerate_element"


class DanglingNodeError(ValidationError):
    code = "dangling_node"


class UndefinedMaterialError(ValidationError):
    code = "undefined_material"


class InvalidBoundaryConditionError(ValidationError):
    code = "invalid_boundary_condition"


class InvalidLoadError(ValidationError):
    code = "invalid_load"


class UnconstrainedModelError(ValidationError):
    code = "unconstrained_model"


class SingularMatrixError(FemError):
    code = "singular_matrix"


class EquilibriumNotSatisfiedError(FemError):
    code = "equilibrium_not_satisfied"


class JobNotFoundError(FemError):
    code = "job_not_found"


class JobInvalidStateError(FemError):
    code = "job_invalid_state"


class JobTimeoutError(FemError):
    code = "job_timeout"


class JobCancelledError(FemError):
    code = "job_cancelled"


class JobLimitExceededError(FemError):
    code = "job_limit_exceeded"


class ModelTooLargeError(ValidationError):
    code = "model_too_large"


class CheckpointError(FemError):
    code = "checkpoint_error"
