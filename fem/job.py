"""In-memory job state holder with a lock and a cancellation event."""

from __future__ import annotations

import threading

from fem.models import JobInfo, JobState, TERMINAL_STATES


class Job:
    """Thread-safe mutable envelope around an immutable :class:`JobInfo`.

    The lock serializes state transitions; ``cancel_event`` is polled by the
    solver to abort cooperatively.  Completed/failed payloads are frozen once
    the job reaches a terminal state.
    """

    def __init__(self, info: JobInfo) -> None:
        self.info = info
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        # Set when a worker has fully finished (including persistence).
        self.done_event = threading.Event()

    @property
    def job_id(self) -> str:
        return self.info.job_id

    def snapshot(self) -> JobInfo:
        with self.lock:
            return self.info.model_copy(deep=True)

    def update(self, **changes) -> JobInfo:
        """Apply changes under the lock; terminal jobs no longer change."""
        with self.lock:
            if self.info.state in TERMINAL_STATES:
                return self.snapshot()
            data = self.info.model_dump()
            data.update(changes)
            self.info = JobInfo.model_validate(data)
            if self.info.state in TERMINAL_STATES:
                self.done_event.set()
            return self.snapshot()

    def request_cancel(self) -> bool:
        """Signal cooperative cancellation.

        Returns True when the caller is the one that cancels a non-terminal
        job; terminal or already-cancelling jobs return False.
        """
        with self.lock:
            if self.info.state in TERMINAL_STATES:
                return False
            if self.cancel_event.is_set():
                return False
            self.cancel_event.set()
            return True

    def is_cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def state(self) -> JobState:
        with self.lock:
            return self.info.state

    def wait_done(self, timeout: float | None = None) -> bool:
        return self.done_event.wait(timeout)
