"""Configurable resource limits. Over-limit jobs are rejected before start."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResourceLimits(BaseModel):
    max_nodes: int = Field(default=50_000, gt=0)
    max_elements: int = Field(default=50_000, gt=0)
    max_dofs: int = Field(default=100_000, gt=0)
    max_concurrent_jobs: int = Field(default=4, gt=0)
    max_queued_jobs: int = Field(default=64, gt=0)
    job_timeout_seconds: float = Field(default=300.0, gt=0)

    def check_model_size(self, n_nodes: int, n_elements: int) -> None:
        from .errors import LimitExceededError

        n_dofs = 2 * n_nodes
        if n_nodes > self.max_nodes:
            raise LimitExceededError(
                f"node count {n_nodes} exceeds limit {self.max_nodes}",
                details={"nodes": n_nodes, "limit": self.max_nodes},
            )
        if n_elements > self.max_elements:
            raise LimitExceededError(
                f"element count {n_elements} exceeds limit "
                f"{self.max_elements}",
                details={"elements": n_elements, "limit": self.max_elements},
            )
        if n_dofs > self.max_dofs:
            raise LimitExceededError(
                f"DOF count {n_dofs} exceeds limit {self.max_dofs}",
                details={"dofs": n_dofs, "limit": self.max_dofs},
            )
