"""Local structural finite-element static analysis service.

Endpoints
---------
POST   /jobs            submit a solve job (async, returns job handle)
GET    /jobs            list all jobs
GET    /jobs/{job_id}   query status / result (results only when completed)
POST   /jobs/{job_id}/cancel  cancel a pending/running job
GET    /health          liveness probe
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from fea.errors import FEAError, LimitExceededError, ValidationError
from fea.jobs import JobManager
from fea.limits import ResourceLimits
from fea.models import JobInfo, ModelInput
from fea.storage import JobStore


def _limits_from_env() -> ResourceLimits:
    env = os.environ
    return ResourceLimits(
        max_nodes=int(env.get("FEA_MAX_NODES", 50_000)),
        max_elements=int(env.get("FEA_MAX_ELEMENTS", 50_000)),
        max_dofs=int(env.get("FEA_MAX_DOFS", 100_000)),
        max_concurrent_jobs=int(env.get("FEA_MAX_CONCURRENT_JOBS", 4)),
        max_queued_jobs=int(env.get("FEA_MAX_QUEUED_JOBS", 64)),
        job_timeout_seconds=float(env.get("FEA_JOB_TIMEOUT_SECONDS", 300.0)),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = os.environ.get("FEA_DB_PATH", "fea_jobs.db")
    store = JobStore(db_path)
    store.initialize()
    manager = JobManager(store, limits=_limits_from_env())
    manager.start()  # recovers unfinished jobs from the previous process
    app.state.store = store
    app.state.manager = manager
    yield
    manager.shutdown(wait=True)
    store.close()


app = FastAPI(title="FEA Static Analysis Service", lifespan=lifespan)


@app.exception_handler(FEAError)
async def fea_error_handler(request: Request, exc: FEAError) -> JSONResponse:
    if isinstance(exc, ValidationError):
        code = status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(exc, LimitExceededError):
        code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    else:
        code = status.HTTP_400_BAD_REQUEST
    return JSONResponse(status_code=code, content={"error": exc.to_dict()})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
def submit_job(model: ModelInput, request: Request) -> JobInfo:
    return request.app.state.manager.submit(model)


@app.get("/jobs")
def list_jobs(request: Request) -> list[JobInfo]:
    return request.app.state.manager.list_jobs()


@app.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> JobInfo:
    return request.app.state.manager.get(job_id)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> JobInfo:
    return request.app.state.manager.cancel(job_id)
