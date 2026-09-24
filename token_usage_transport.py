"""Backend-neutral asynchronous token-usage transport.

The transport owns only queueing, batching, coalescing, lifecycle, and failure
isolation. Persistence is injected as a callback; no database connection or SQL
surface crosses this boundary.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, Protocol

logger = logging.getLogger("hermes_state")

TokenDelta = tuple[str, dict[str, Any]]


class TokenUsagePersistence(Protocol):
    """Persistence boundary for one ordered token-usage delta."""

    def __call__(self, session_id: str, **kwargs: Any) -> None: ...


class TokenUsageTransport:
    """Ordered, coalescing, best-effort token-usage writer.

    ``persist`` is the only backend-specific operation. It must apply one
    delta atomically in the selected store; it never receives a raw connection.
    """

    def __init__(
        self,
        persist: TokenUsagePersistence,
        *,
        sum_fields: Sequence[str],
        cost_fields: Sequence[str],
        route_fields: Sequence[str],
        idle_seconds: Callable[[], float],
        coalesce: Callable[[list[TokenDelta]], list[TokenDelta]] | None = None,
    ) -> None:
        self._persist = persist
        self._sum_fields = tuple(sum_fields)
        self._cost_fields = tuple(cost_fields)
        self._route_fields = tuple(route_fields)
        self._idle_seconds = idle_seconds
        self._coalesce_override = coalesce
        self.queue: deque[TokenDelta] = deque()
        self.condition = threading.Condition(threading.Lock())
        self.writer_thread: threading.Thread | None = None
        self.stop_requested = self.busy = False
        self._atexit_hook: Callable[[], None] | None = None

    def queue_delta(self, session_id: str, kwargs: dict[str, Any]) -> None:
        """Accept a delta or synchronously persist it after shutdown."""
        with self.condition:
            thread = self.writer_thread
            writer_alive = thread is not None and thread.is_alive()
            writer_stopped = self.stop_requested and not writer_alive
            if not writer_stopped:
                self.queue.append((session_id, kwargs))
                if not writer_alive:
                    thread = threading.Thread(
                        target=self._writer_loop, name="session-db-token-writer", daemon=True,
                    )
                    self.writer_thread = thread
                    thread.start()
                    self._register_atexit_locked()
                self.condition.notify_all()
        if writer_stopped:
            self._persist(session_id, **kwargs)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait for queue plus in-flight work; dead writers are synchronously drained."""
        if not self.queue and not self.busy:
            return True
        batch: list[TokenDelta] | None = None
        with self.condition:
            deadline = time.monotonic() + timeout
            while self.queue or self.busy:
                thread = self.writer_thread
                if (thread is None or not thread.is_alive()) and not self.busy:
                    self.busy = True
                    batch = list(self.queue)
                    self.queue.clear()
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
        if batch:
            self._apply_claimed_batch(batch)
        return True

    def stop(self, join_timeout: float = 10.0) -> None:
        """Stop the worker and drain leftovers without racing an in-flight batch."""
        with self.condition:
            self.stop_requested = True
            self.condition.notify_all()
            thread = self.writer_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                logger.warning(
                    "async token accounting: writer did not stop within %.0fs; %d queued delta(s) not persisted",
                    join_timeout, len(self.queue),
                )
                return
        with self.condition:
            deadline = time.monotonic() + join_timeout
            while self.busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "async token accounting: concurrent drain did not finish within %.0fs; "
                        "%d queued delta(s) not persisted", join_timeout, len(self.queue),
                    )
                    return
                self.condition.wait(remaining)
            batch = list(self.queue)
            if batch:
                self.busy = True
                self.queue.clear()
        if batch:
            self._apply_claimed_batch(batch)

    def close(self) -> None:
        self.stop()
        hook, self._atexit_hook = self._atexit_hook, None
        if hook is not None:
            atexit.unregister(hook)

    def apply_batch(self, batch: list[TokenDelta]) -> None:
        """Apply a claimed batch in order; one failed delta never kills the writer."""
        try:
            coalesced = self._coalesce(batch)
        except Exception as exc:
            logger.warning("async token accounting: coalesce failed, applying raw batch: %s", exc)
            coalesced = batch
        for session_id, kwargs in coalesced:
            try:
                self._persist(session_id, **kwargs)
            except Exception as exc:
                logger.warning("async token accounting: apply failed (session=%s): %s", session_id, exc)

    def coalesce(self, batch: list[TokenDelta]) -> list[TokenDelta]:
        """Merge only contiguous, incremental, equal-route deltas."""
        groups: list[tuple[tuple[Any, ...] | None, str, dict[str, Any]]] = []
        for session_id, kwargs in batch:
            key = None
            if not kwargs.get("absolute"):
                key = (session_id, *(kwargs.get(field) for field in self._route_fields))
            if groups and key is not None and groups[-1][0] == key:
                merged = groups[-1][2]
                for field in self._sum_fields:
                    merged[field] = merged.get(field, 0) + kwargs.get(field, 0)
                for field in self._cost_fields:
                    value = kwargs.get(field)
                    if value is not None:
                        merged[field] = (merged.get(field) or 0.0) + value
            else:
                groups.append((key, session_id, dict(kwargs)))
        return [(session_id, kwargs) for _, session_id, kwargs in groups]

    def _coalesce(self, batch: list[TokenDelta]) -> list[TokenDelta]:
        return self._coalesce_override(batch) if self._coalesce_override is not None else self.coalesce(batch)

    def _apply_claimed_batch(self, batch: list[TokenDelta]) -> None:
        try:
            self.apply_batch(batch)
        finally:
            with self.condition:
                self.busy = False
                self.condition.notify_all()

    def _writer_loop(self) -> None:
        while True:
            with self.condition:
                idle_deadline = time.monotonic() + self._idle_seconds()
                while not self.queue and not self.stop_requested:
                    remaining = idle_deadline - time.monotonic()
                    if remaining <= 0:
                        self.writer_thread = None
                        return
                    self.condition.wait(remaining)
                if not self.queue:
                    self.writer_thread = None
                    return
                self.busy = True
                batch = list(self.queue)
                self.queue.clear()
            self._apply_claimed_batch(batch)

    def _register_atexit_locked(self) -> None:
        if self._atexit_hook is not None:
            return
        self_ref = weakref.ref(self)

        def _drain_at_exit() -> None:
            transport = self_ref()
            if transport is not None:
                with contextlib.suppress(Exception):
                    transport.stop()

        self._atexit_hook = _drain_at_exit
        atexit.register(_drain_at_exit)
