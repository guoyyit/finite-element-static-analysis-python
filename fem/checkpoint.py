"""Crash-safe checkpoint storage: atomic JSON payloads per job.

Incomplete runs resume from the last checkpoint; completed jobs stay
queryable after restart.  Every save is a ``write-temp -> fsync -> rename``
sequence, so a crash can leave either the previous complete file or the new
complete file, never a truncated payload that could be mistaken for a result.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fem.errors import CheckpointError
from fem.models import JobInfo

_SUFFIX = ".json"
_TMP_SUFFIX = ".tmp"
_MODEL_SUFFIX = ".model.json"


class CheckpointStore:
    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)

    # -- paths -------------------------------------------------------------
    def job_path(self, job_id: str) -> Path:
        self._validate_id(job_id)
        return self.directory / f"{job_id}{_SUFFIX}"

    @staticmethod
    def _validate_id(job_id: str) -> None:
        if not job_id or "/" in job_id or os.sep in job_id or ".." in job_id:
            raise CheckpointError(f"invalid job id {job_id!r}")

    # -- lifecycle ---------------------------------------------------------
    def initialize(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, info: JobInfo) -> None:
        """Atomically persist a job snapshot and sync it to disk."""
        try:
            self.initialize()
            target = self.job_path(info.job_id)
            data = self.dumps(info)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{info.job_id}.", suffix=_TMP_SUFFIX,
                dir=self.directory)
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, target)
                self._fsync_dir()
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
        except OSError as exc:
            raise CheckpointError(
                f"failed to persist job {info.job_id}: {exc}") from exc

    def _fsync_dir(self) -> None:
        try:
            dir_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Some platforms do not support fsync on directories.
            pass

    def load(self, job_id: str) -> JobInfo | None:
        path = self.job_path(job_id)
        if not path.exists():
            return None
        try:
            return self.loads(path.read_bytes())
        except (OSError, ValueError, KeyError) as exc:
            raise CheckpointError(
                f"corrupt checkpoint for job {job_id}: {exc}") from exc

    def load_all(self) -> dict[str, JobInfo]:
        if not self.directory.exists():
            return {}
        out: dict[str, JobInfo] = {}
        for path in sorted(self.directory.glob(f"*{_SUFFIX}")):
            try:
                info = self.loads(path.read_bytes())
            except (ValueError, KeyError):
                # Ignore a torn/legacy file rather than crash recovery; it is
                # never returned as a result.
                continue
            out[info.job_id] = info
        return out

    def delete(self, job_id: str) -> None:
        path = self.job_path(job_id)
        try:
            path.unlink(missing_ok=True)
            self.model_path(job_id).unlink(missing_ok=True)
            self.assembly_path(job_id).unlink(missing_ok=True)
            self._fsync_dir()
        except OSError as exc:
            raise CheckpointError(
                f"failed to delete job {job_id}: {exc}") from exc

    # -- raw model payload (needed to resume after process restart) --------
    def model_path(self, job_id: str) -> Path:
        self._validate_id(job_id)
        return self.directory / f"{job_id}{_MODEL_SUFFIX}"

    def save_model(self, job_id: str, raw: bytes) -> None:
        try:
            self.initialize()
            target = self.model_path(job_id)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{job_id}.model.", suffix=_TMP_SUFFIX,
                dir=self.directory)
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(raw)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, target)
                self._fsync_dir()
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
        except OSError as exc:
            raise CheckpointError(
                f"failed to persist model for job {job_id}: {exc}") from exc

    def load_model_raw(self, job_id: str) -> bytes | None:
        path = self.model_path(job_id)
        if not path.exists():
            return None
        try:
            return path.read_bytes()
        except OSError as exc:
            raise CheckpointError(
                f"failed to read model for job {job_id}: {exc}") from exc

    def delete_model(self, job_id: str) -> None:
        self.model_path(job_id).unlink(missing_ok=True)

    # -- resumable assembly matrix (large binary blob) ---------------------
    def assembly_path(self, job_id: str) -> Path:
        self._validate_id(job_id)
        return self.directory / f"{job_id}.K.npy"

    def _atomic_write_bytes(self, name: str, data: bytes) -> None:
        self.initialize()
        target = self.directory / name
        fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.",
                                        suffix=_TMP_SUFFIX,
                                        dir=self.directory)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
            self._fsync_dir()
        except OSError as exc:
            raise CheckpointError(f"failed to write {name}: {exc}") from exc
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def save_assembly(self, job_id: str, matrix) -> None:
        import io
        import numpy as np

        buf = io.BytesIO()
        np.save(buf, np.asarray(matrix, dtype=np.float64), allow_pickle=False)
        self._atomic_write_bytes(self.assembly_path(job_id).name,
                                 buf.getvalue())

    def load_assembly(self, job_id: str):
        import numpy as np

        path = self.assembly_path(job_id)
        if not path.exists():
            return None
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise CheckpointError(
                f"corrupt assembly checkpoint for job {job_id}: {exc}") from exc

    def delete_assembly(self, job_id: str) -> None:
        self.assembly_path(job_id).unlink(missing_ok=True)

    @staticmethod
    def dumps(info: JobInfo) -> bytes:
        return (json.dumps(info.model_dump(mode="json"),
                           sort_keys=True, separators=(",", ":"),
                           allow_nan=False) + "\n").encode("utf-8")

    @staticmethod
    def loads(raw: bytes | str) -> JobInfo:
        data = json.loads(raw)
        return JobInfo.model_validate(data)
