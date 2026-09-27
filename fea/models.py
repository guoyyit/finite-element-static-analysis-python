"""Pydantic schemas for FEA job inputs, outputs and job metadata."""

from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, Field


class AnalysisType(str, enum.Enum):
    PLANE_STRESS = "plane_stress"
    PLANE_STRAIN = "plane_strain"


class Node(BaseModel):
    id: int
    x: float
    y: float


class Material(BaseModel):
    id: int
    E: float = Field(description="Young's modulus, must be > 0")
    nu: float = Field(description="Poisson's ratio, must be in (-1, 0.5)")


class Element(BaseModel):
    """Four-node quadrilateral (Q4) plane element, nodes in CCW or CW order."""

    id: int
    type: Literal["Q4"] = "Q4"
    nodes: list[int] = Field(min_length=4, max_length=4)
    material_id: int
    thickness: float = Field(gt=0)


class DisplacementBC(BaseModel):
    """Prescribed displacement on one node along one direction."""

    node_id: int
    direction: Literal["x", "y"]
    value: float = 0.0


class PointLoad(BaseModel):
    node_id: int
    fx: float = 0.0
    fy: float = 0.0


class SolveOptions(BaseModel):
    equilibrium_tol: float = Field(
        default=1e-6,
        gt=0,
        description="Max allowed absolute residual of global force equilibrium.",
    )


class ModelInput(BaseModel):
    """Complete static-analysis model description."""

    analysis_type: AnalysisType = AnalysisType.PLANE_STRESS
    nodes: list[Node]
    elements: list[Element]
    materials: list[Material]
    displacement_bcs: list[DisplacementBC] = []
    point_loads: list[PointLoad] = []
    options: SolveOptions = SolveOptions()


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


class NodeResult(BaseModel):
    node_id: int
    ux: float
    uy: float


class StressComponents(BaseModel):
    sxx: float
    syy: float
    txy: float


class ElementResult(BaseModel):
    element_id: int
    stress: StressComponents
    von_mises: float


class Reaction(BaseModel):
    node_id: int
    rx: float
    ry: float


class SolveResult(BaseModel):
    displacements: list[NodeResult]
    element_results: list[ElementResult]
    reactions: list[Reaction]
    max_equilibrium_residual: float


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    model_hash: str
    created_at: str
    updated_at: str
    error: dict | None = None
    result: SolveResult | None = None
