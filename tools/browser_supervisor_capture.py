"""Trusted-Python captured CDP transport; not target ownership authorization."""
from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tools.browser_supervisor import CDPSupervisor, _SupervisorRegistry


class CapturedCDPInvalid(RuntimeError):
    """The captured supervisor, connection, or default attachment changed."""


def _timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class CapturedCDPIdentity:
    """Opaque comparison tokens. No endpoint, credentials, or ownership claim."""

    supervisor: str
    connection: str
    attachment: str


class _DispatchFence:
    """Caller abandonment/deadline must be observed before admitting a send."""
    def __init__(self, timeout):
        self.deadline = time.monotonic() + timeout
        self.abandoned = threading.Event()

    def begin(self):
        if self.abandoned.is_set() or time.monotonic() >= self.deadline:
            raise CapturedCDPInvalid("Captured CDP call abandoned before dispatch")

    def abandon(self):
        self.abandoned.set()


class _AttachmentHandoff:
    """One reply owner until the caller claims it or old-wire disposal takes it."""

    def __init__(self, capture, session_id, cleanup):
        self.capture = capture
        self.parent_session_id = session_id
        self.cleanup = cleanup
        self.future = concurrent.futures.Future()
        self._lock = threading.Lock()
        self._abandoned = False
        self._claimed = False
        self._session_id = None
        self._disposal_started = False

    def begin(self):
        """Linearize dispatch against caller abandonment before sending."""
        with self._lock:
            if self._abandoned:
                raise CapturedCDPInvalid("Attachment acquisition abandoned before dispatch")
            if not self.cleanup:
                self.capture.check_valid()

    def _take_disposal_locked(self):
        if self._abandoned and self._session_id and not self._claimed and not self._disposal_started:
            self._disposal_started = True
            return self._session_id
        return None

    def abandon(self):
        with self._lock:
            if not self._claimed:
                self._abandoned = True
            session_id = self._take_disposal_locked()
        self._dispose(session_id)

    def fail(self, error):
        with self._lock:
            self._abandoned = True
            if not self.future.done():
                self.future.set_exception(error)
            session_id = self._take_disposal_locked()
        self._dispose(session_id)

    def publish(self, result):
        with self._lock:
            session_id = result.get("result", {}).get("sessionId")
            if isinstance(session_id, str) and session_id:
                self._session_id = session_id
            try:
                if not self.cleanup:
                    self.capture.check_valid()
            except CapturedCDPInvalid as error:
                self._abandoned = True
                if not self.future.done():
                    self.future.set_exception(error)
            else:
                if not self.future.done():
                    self.future.set_result(result)
            session_id = self._take_disposal_locked()
        self._dispose(session_id)

    def claim(self):
        with self._lock:
            if self._abandoned:
                raise CapturedCDPInvalid("Attachment acquisition abandoned")
            if not self.cleanup:
                self.capture.check_valid()
            self._claimed = True

    def _dispose(self, session_id):
        if session_id is None:
            return
        from agent.async_utils import safe_schedule_threadsafe

        async def detach():
            try:
                await self.capture._supervisor._captured_request(
                    self.capture, "Target.detachFromTarget", {"sessionId": session_id},
                    self.parent_session_id, 10.0, True)
            except Exception:
                pass

        safe_schedule_threadsafe(detach(), self.capture._loop)


class CapturedCDP:
    """A non-retargeting handle. Construct through SUPERVISOR_REGISTRY.capture().

    Sync calls must run off the supervisor thread. Async calls may run on any
    event loop. Every call requires an explicit session_id (None = browser).
    Cleanup deliberately skips current-identity fences, but uses only the old
    wire. Neither path discovers an endpoint, connects, or focuses a page.
    """

    def __init__(self, supervisor: CDPSupervisor, registry: _SupervisorRegistry):
        self._supervisor = supervisor
        self._registry = registry
        self._loop = supervisor._loop
        self._wire = supervisor._ws
        self._identity = CapturedCDPIdentity(supervisor._capture_id, supervisor._connection_id,
                                             supervisor._attachment_id)
        self._page_session_id = supervisor._page_session_id

    @property
    def identity(self) -> CapturedCDPIdentity:
        return self._identity

    @property
    def page_session_id(self) -> str:
        return self._page_session_id

    @property
    def valid(self) -> bool:
        sup = self._supervisor
        return (self._registry.get(sup.task_id) is sup and sup._active and not sup._stop_requested
                and sup._loop is self._loop and self._loop is not None and self._loop.is_running()
                and sup._ws is self._wire and self._wire is not None
                and sup._connection_id == self.identity.connection
                and sup._attachment_id == self.identity.attachment
                and sup._page_session_id == self.page_session_id and not self._wire_closed())

    def check_valid(self) -> None:
        if not self.valid:
            raise CapturedCDPInvalid("Captured CDP connection or default attachment changed")

    def _wire_closed(self) -> bool:
        return (getattr(self._wire, "closed", False) is True
                or getattr(getattr(self._wire, "state", None), "name", None) == "CLOSED")

    def _submit(self, method, params, session_id, timeout, cleanup):
        from agent.async_utils import safe_schedule_threadsafe

        timeout = _timeout(timeout)
        if not isinstance(method, str) or not method:
            raise ValueError("method must be a nonempty CDP method")
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise ValueError("session_id must be None or a nonempty string")
        if self._loop is None or not self._loop.is_running():
            raise CapturedCDPInvalid("Captured CDP loop unavailable")
        handoff = _AttachmentHandoff(self, session_id, cleanup) if method == "Target.attachToTarget" else None
        dispatch = _DispatchFence(timeout) if not cleanup else None
        future = safe_schedule_threadsafe(
            self._supervisor._captured_request(self, method, params, session_id, timeout, cleanup, handoff, dispatch),
            self._loop)
        if future is None:
            raise CapturedCDPInvalid("Captured CDP loop unavailable")
        return handoff.future if handoff else future, timeout, handoff, dispatch

    def _sync_call(self, method, params, session_id, timeout, cleanup):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._loop:
            raise RuntimeError("Use acall/acleanup_call on the supervisor loop")
        future, timeout, handoff, dispatch = self._submit(method, params, session_id, timeout, cleanup)
        try:
            result = future.result(timeout)
            if handoff:
                handoff.claim()
            elif not cleanup:
                self.check_valid()
            return result
        except BaseException:
            if dispatch:
                dispatch.abandon()
            if handoff:
                handoff.abandon()
            elif not cleanup:
                future.cancel()
            raise

    async def _async_call(self, method, params, session_id, timeout, cleanup):
        from agent.async_utils import consume_detached_task_result

        future, timeout, handoff, dispatch = self._submit(method, params, session_id, timeout, cleanup)
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(consume_detached_task_result)
        try:
            result = await asyncio.wait_for(asyncio.shield(wrapped), timeout)
            if handoff:
                handoff.claim()
            elif not cleanup:
                self.check_valid()
            return result
        except BaseException:
            if dispatch:
                dispatch.abandon()
            if handoff:
                handoff.abandon()
            elif not cleanup:
                future.cancel()
            raise

    def call(self, method: str, params: Optional[Dict[str, Any]] = None, *,
             session_id: Optional[str], timeout: float = 10.0) -> Dict[str, Any]:
        """Call on the captured wire; reject invalidation before/after effects."""
        return self._sync_call(method, params, session_id, timeout, False)

    async def acall(self, method: str, params: Optional[Dict[str, Any]] = None, *,
                    session_id: Optional[str], timeout: float = 10.0) -> Dict[str, Any]:
        """Async call, including cancellation from another event loop."""
        return await self._async_call(method, params, session_id, timeout, False)

    def cleanup_call(self, method: str, params: Optional[Dict[str, Any]] = None, *,
                     session_id: Optional[str], timeout: float = 10.0) -> Dict[str, Any]:
        """Explicit disposal on the old wire, even after invalidation; errors raise."""
        return self._sync_call(method, params, session_id, timeout, True)

    async def acleanup_call(self, method: str, params: Optional[Dict[str, Any]] = None, *,
                            session_id: Optional[str], timeout: float = 10.0) -> Dict[str, Any]:
        """Async equivalent of cleanup_call."""
        return await self._async_call(method, params, session_id, timeout, True)
