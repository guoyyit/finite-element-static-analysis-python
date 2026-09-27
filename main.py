"""Service entry point: ``uv run uvicorn main:app``."""

from fem.api import app

__all__ = ["app"]
