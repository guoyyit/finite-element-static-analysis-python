"""Input / output data models for the FEM service."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# Bumped whenever the solver semantics or output format changes in a way that
# invalidates previously cached results.
SOLVER_VERSION = "fem-solver-1.0.0"
INPUT_SCHEMA_VERSION = 1

# Supported element families. At least one plane stress / plane strain element
# is required; we support the 4-node bilinear quadrilateral in both flavors.
ELEMENT_PLANE_STRESS = "CPS4"
ELEMENT_PLANE_STRAIN = "CPE4"
SUPPORTED_ELEMENT_TYPES = frozenset({ELEMENT_PLANE_STRESS, ELEMENT_PLANE_STRAIN})


class NodeInput(BaseModel):
    id: int
    x: float
    y: float


class ElementInput(BaseModel):
    id: int
    type: str
    nodes: list[int] = Field(min_length=4, max_length=4)
    material: str


class MaterialInput(BaseModel):
    name: str
    youngs_modulus: float
    poisson_ratio: float
    # Thickness only used by plane stress elements; accepted for both.
    thickness: float = 1.0


class BoundaryConditionInput(BaseModel):
    node: int
    # 0 -> ux, 1 -> uy
    dof: Literal[0, 1]
    value: float = 0.0


class NodalLoadInput(BaseModel):
    node: int
    dof: Literal[0, 1]
    value: float


class SolveOptions(BaseModel):
    equilibrium_tolerance: float | None = None
    timeout_seconds: float | None = None


class ModelInput(BaseModel):
    version: int = INPUT_SCHEMA_VERSION
    nodes: list[NodeInput]
    elements: list[ElementInput]
    materials: list[MaterialInput]
    boundary_conditions: list[BoundaryConditionInput]
    loads: list[NodalLoadInput]
    options: SolveOptions = Field(default_factory=SolveOptions)


class NodeDisplacement(BaseModel):
    node: int
    ux: float
    uy: float


class ElementStress(BaseModel):
    element: int
    sxx: float
    syy: float
    sxy: float
    von_mises: float


class Reaction(BaseModel):
    node: int
    fx: float
    fy: float


class ResidualReport(BaseModel):
    max_abs_residual: float
    # Global identifier of the DOF carrying the worst residual.
    node: int
    dof: int
    tolerance: float


class SolveResults(BaseModel):
    solver_version: str
    model_hash: str
    displacements: list[NodeDisplacement]
    stresses: list[ElementStress]
    reactions: list[Reaction]
    residual: ResidualReport


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


TERMINAL_STATES = frozenset(
    {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT}
)
# States whose on-disk payload must never be mistaken for a valid result.
NON_RESULT_STATES = frozenset(
    {JobState.PENDING, JobState.RUNNING, JobState.FAILED,
     JobState.CANCELLED, JobState.TIMEOUT}
)


class JobErrorInfo(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class JobInfo(BaseModel):
    job_id: str
    state: JobState
    model_hash: str
    progress: float = 0.0
    error: JobErrorInfo | None = None
    results: SolveResults | None = None
    # Resumable assembly checkpoint: number of elements accumulated and the
    # file (under the store directory) holding the partial stiffness matrix.
    assembled_through: int = 0
    assembly_file: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
