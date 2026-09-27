"""FastAPI HTTP surface for the FEM static analysis service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from fem.config import ServiceConfig
from fem.errors import FemError, JobNotFoundError, ValidationError
from fem.manager import JobManager
from fem.models import (
    SOLVER_VERSION,
    JobInfo,
    ModelInput,
    SolveResults,
)

logger = logging.getLogger("fem.api")


class HealthResponse(BaseModel):
    status: str
    solver_version: str


class SubmitResponse(BaseModel):
    job: JobInfo


class BatchSubmitRequest(BaseModel):
    # Batch submission is atomic: if any model is invalid the whole batch
    # is rejected and no job is created.
    models: list[ModelInput] = Field(min_length=1)


class BatchSubmitResponse(BaseModel):
    jobs: list[JobInfo]


def create_app(config: ServiceConfig | None = None,
               manager: JobManager | None = None) -> FastAPI:
    config = config or ServiceConfig()
    own_manager = manager is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mgr.start()
        try:
            yield
        finally:
            if own_manager:
                mgr.shutdown(wait=False)

    mgr = manager or JobManager(config=config)
    application = FastAPI(title="Local FEM Static Analysis",
                          version="1.0.0", lifespan=lifespan)
    application.state.fem_config = config
    application.state.fem_manager = mgr

    @application.exception_handler(FemError)
    async def _fem_error_handler(_request, exc: FemError):
        # Validation problems are client errors; runtime solver failures are
        # 409/500-ish. Validation -> 422, unknown job -> 404, state -> 409,
        # limits -> 413, everything else 500.
        status = 400
        if isinstance(exc, ValidationError):
            status = 422
        elif isinstance(exc, JobNotFoundError):
            status = 404
        elif exc.code == "job_invalid_state":
            status = 409
        elif exc.code in ("job_limit_exceeded", "model_too_large"):
            status = 413
        return JSONResponse(
            status_code=status,
            content={"error": {"code": exc.code, "message": exc.message,
                               "details": exc.details}})

    @application.exception_handler(RequestValidationError)
    async def _schema_error_handler(_request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request_schema",
                               "message": "request payload failed schema "
                                          "validation",
                               "details": {"errors": exc.errors()}}})

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", solver_version=SOLVER_VERSION)

    @application.post("/jobs", response_model=SubmitResponse, status_code=201)
    async def submit_job(model: ModelInput) -> SubmitResponse:
        return SubmitResponse(job=mgr.submit(model))

    @application.post("/jobs/batch", response_model=BatchSubmitResponse,
                      status_code=201)
    async def submit_batch(payload: BatchSubmitRequest) -> BatchSubmitResponse:
        return BatchSubmitResponse(jobs=mgr.submit_batch(payload.models))

    @application.get("/jobs", response_model=list[JobInfo])
    async def list_jobs() -> list[JobInfo]:
        return mgr.list_jobs()

    @application.get("/jobs/{job_id}", response_model=JobInfo)
    async def get_job(job_id: str) -> JobInfo:
        return mgr.get(job_id)

    @application.get("/jobs/{job_id}/result", response_model=SolveResults)
    async def get_result(job_id: str) -> SolveResults:
        return mgr.get_result(job_id)

    @application.post("/jobs/{job_id}/cancel", response_model=JobInfo)
    async def cancel_job(job_id: str) -> JobInfo:
        return mgr.cancel(job_id)

    @application.delete("/jobs/{job_id}", status_code=204)
    async def delete_job(job_id: str) -> None:
        mgr.delete_job(job_id)

    return application


app = create_app()
