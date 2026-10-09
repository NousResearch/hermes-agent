"""Captured CDP connection handle for trusted in-process plugins.

``SUPERVISOR_REGISTRY.capture(task_id)`` pins one supervisor's current WebSocket and the
default page session attached on it. ``call()`` sends only over that socket. Once the
supervisor reconnects, stops, or is replaced in the registry, the handle is invalid and
every call raises ``CapturedCDPInvalid``: it never retargets a newer connection.

This grants no new capability. In-process plugin code can already reach the supervisor;
the handle exists so plugins stop writing its private call table and socket (catalog
review on #132272). It performs no origin, consent or target-ownership checks.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from tools.browser_supervisor import CDPSupervisor, _SupervisorRegistry


class CapturedCDPInvalid(RuntimeError):
    """The captured connection is gone (reconnect, stop, or registry replacement)."""


class CapturedCDP:
    """A CDP channel pinned to one supervisor connection. Build via ``SUPERVISOR_REGISTRY.capture``."""

    def __init__(self, supervisor: CDPSupervisor, registry: _SupervisorRegistry) -> None:
        # Runs on the supervisor loop (see ``capture``) so ws and session id are one snapshot.
        self._supervisor = supervisor
        self._registry = registry
        self._loop: asyncio.AbstractEventLoop = supervisor._loop  # type: ignore[assignment]
        self._ws = supervisor._ws
        self.task_id: str = supervisor.task_id
        self.cdp_url: str = supervisor.cdp_url
        # The default page session at capture time. A later ``focus_page`` on the supervisor
        # does not change it: calls with this id keep reaching the page captured here.
        self.page_session_id: str = supervisor._page_session_id or ""

    def is_valid(self) -> bool:
        """True while the captured connection is still the supervisor's live one."""
        sup = self._supervisor
        return (self._registry.get(self.task_id) is sup and sup._active and not sup._stop_requested
                and sup._ws is self._ws and self._loop.is_running())

    def _invalid(self) -> CapturedCDPInvalid:
        return CapturedCDPInvalid(f"captured CDP connection for task {self.task_id!r} is gone "
                                  "(supervisor reconnected, stopped or was replaced); capture again")

    async def _send(self, method: str, params: Optional[dict[str, Any]],
                    session_id: Optional[str], timeout: float) -> dict[str, Any]:
        # ``_cdp`` reaches its send without awaiting, so this check and the send run in one
        # loop step: a reconnect cannot swap the socket in between.
        if not self.is_valid():
            raise self._invalid()
        try:
            return await self._supervisor._cdp(method, params, session_id=session_id, timeout=timeout)
        except Exception:
            # A reply lost to a dropped socket (timeout, ConnectionError, ConnectionClosed)
            # surfaces as invalidation; a CDP error on a live connection passes through.
            if not self.is_valid():
                raise self._invalid() from None
            raise

    def call(self, method: str, params: Optional[dict[str, Any]] = None, *,
             session_id: Optional[str] = None, timeout: float = 10.0) -> dict[str, Any]:
        """Send ``method`` on the captured connection and return the raw CDP reply
        (``{"id", "result"}``). ``session_id=None`` targets the browser endpoint;
        pass ``page_session_id`` (or a session you attached) for page domains.

        Raises ``CapturedCDPInvalid`` when the connection is gone, ``TimeoutError`` when no
        reply arrives in ``timeout`` seconds, ``RuntimeError`` for a CDP error reply.
        Blocks the calling thread; never call it from the supervisor's own loop.
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            raise RuntimeError("CapturedCDP.call() blocks and cannot run on the supervisor loop")
        if not self.is_valid():
            raise self._invalid()
        from agent.async_utils import safe_schedule_threadsafe

        fut = safe_schedule_threadsafe(self._send(method, params, session_id, timeout), self._loop)
        if fut is None:
            raise self._invalid()
        try:
            return fut.result(timeout=timeout + 1.0)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise TimeoutError(f"CDP {method} got no reply within {timeout}s") from None


def capture(registry: _SupervisorRegistry, task_id: str, *, timeout: float = 10.0) -> CapturedCDP:
    """Pin the running supervisor for ``task_id``; never starts, reconnects or refocuses one."""
    from tools.browser_supervisor import _LoopUnavailable, _schedule

    found = registry.get(task_id)
    loop = found._loop if found is not None else None
    if found is None or loop is None or not loop.is_running():
        raise CapturedCDPInvalid(f"no running browser supervisor for task {task_id!r}")
    sup = found

    async def _snapshot() -> CapturedCDP:
        if not sup._active or sup._ws is None or not sup._page_session_id:
            raise CapturedCDPInvalid(f"browser supervisor for task {task_id!r} has no attached page")
        return CapturedCDP(sup, registry)

    try:
        return _schedule(_snapshot(), loop, timeout=timeout)
    except _LoopUnavailable:
        raise CapturedCDPInvalid(f"no running browser supervisor for task {task_id!r}") from None
