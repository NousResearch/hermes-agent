"""Session health for MCPServerTask: dynamic tool refresh on list_changed notifications, server
log forwarding, keepalive probes, suspect-mark / lazy-verify, in-flight call fail-fast, stdio
child liveness and stdio idle/lifetime recycling."""

import asyncio
import json
import logging
import time
from typing import Iterable, Optional
from tools.mcp_tool_errors import _is_method_not_found_error, _unwrap_exception_group
from tools.mcp_tool_schema import mcp_prefixed_tool_name
from tools.mcp_tool_registration import _forget_mcp_tool_server
from tools.mcp_tool_common import _core
from tools import mcp_tool_registration as _registration

logger = logging.getLogger("tools.mcp_tool")

_KEEPALIVE_RPC_TIMEOUT = 30.0


class MCPServerHealthMixin:
    """Methods of :class:`tools.mcp_tool.MCPServerTask` (mixed in; relies on its attributes)."""

    __slots__ = ()

    def _is_http(self) -> bool:
        return "url" in self._config

    def _is_recycled_stdio(self) -> bool:
        """True when a stdio server was intentionally recycled."""
        return not self._is_http() and self._recycled_reason is not None

    def mark_tool_call(self) -> None:
        """Record that a user-visible MCP operation is starting."""
        self._last_tool_call_at = time.monotonic()

    def _mark_lifecycle_started(self) -> None:
        self._lifecycle_started_at = self._last_tool_call_at = time.monotonic()
        self._recycled_reason = None

    def _stdio_recycle_deadlines(self):
        """``[(deadline, reason), ...]`` for the lifetime/idle limits; empty for HTTP or while an RPC holds the lock."""
        if self._is_http() or self._rpc_lock.locked():
            return []
        limits = ((self._lifecycle_started_at, self._max_lifetime_seconds, "max_lifetime_seconds"),
                  (self._last_tool_call_at, self._idle_timeout_seconds, "idle_timeout_seconds"))
        return [(start + limit, reason) for start, limit, reason in limits if limit is not None]

    def _stdio_recycle_reason(self, now: Optional[float] = None) -> Optional[str]:
        """The stdio recycle reason if idle/age limits have elapsed (lifetime wins), else None."""
        now = time.monotonic() if now is None else now
        return next((reason for deadline, reason in self._stdio_recycle_deadlines() if now >= deadline), None)

    def _next_stdio_recycle_deadline(self) -> Optional[float]:
        return min((d for d, _ in self._stdio_recycle_deadlines()), default=None)

    def _mark_stdio_recycled(self, reason: str) -> None:
        """Mark a stdio session dormant before its transport finishes closing."""
        self._recycled_reason = reason
        self.session = None

    def _schedule_tools_refresh(self) -> asyncio.Task:
        """Schedule a background tool refresh (failures logged) and keep it strongly referenced."""
        async def _run():
            try:
                await self._refresh_tools()
            except Exception:
                logger.exception("MCP server '%s': dynamic tool refresh failed", self.name)
        task = asyncio.create_task(_run())
        self._pending_refresh_tasks.add(task)
        task.add_done_callback(self._pending_refresh_tasks.discard)
        return task

    def _make_logging_callback(self):
        """``logging_callback`` forwarding server ``notifications/message`` into Hermes logging (SDK default drops them).

        Routes MCP ``notifications/message`` log notifications from the server into Hermes' logging
        (agent.log via hermes_logging), tagged with the server name. Without this, the SDK's default
        callback silently discards them, so server-side warnings/errors during a tool call were invisible.
        Port of anomalyco/opencode#34529.
        """
        async def _on_log(params):
            try:
                level = _core._MCP_LOG_LEVEL_MAP.get(str(getattr(params, "level", "info")).lower(), logging.INFO)
                data = getattr(params, "data", None)
                if not isinstance(data, str):
                    try:
                        data = json.dumps(data, ensure_ascii=False, default=str)
                    except (TypeError, ValueError):
                        data = str(data)
                if len(data) > 2000:  # cap payloads so a chatty server can't flood agent.log
                    data = data[:2000] + "... [truncated]"
                logger_name = getattr(params, "logger", None)
                origin = f"{self.name}/{logger_name}" if logger_name else self.name
                logger.log(level, "MCP server log [%s]: %s", origin, data)
            except Exception:
                logger.debug("Failed to handle MCP log notification from '%s'", self.name, exc_info=True)
        return _on_log

    def _make_message_handler(self):
        """``message_handler``: only ``ToolListChangedNotification`` triggers a refresh; prompt/resource changes log."""
        async def _handler(message):
            try:
                if isinstance(message, Exception):
                    logger.debug("MCP message handler (%s): exception: %s", self.name, message)
                    return
                if not (_core._MCP_NOTIFICATION_TYPES and isinstance(message, _core.ServerNotification)):
                    return
                # mcp 2.0 made ServerNotification a plain union (payload IS the message) instead of
                # a RootModel (payload under ``.root``); without this unwrap refreshes silently stop.
                payload = getattr(message, "root", message)
                if isinstance(payload, _core.ToolListChangedNotification):
                    logger.info("MCP server '%s': received tools/list_changed notification", self.name)
                    # Separate task: refreshing synchronously inside the handler can wedge the stdio
                    # JSON-RPC stream when list_changed arrives while another request is in flight.
                    self._schedule_tools_refresh()
                    await asyncio.sleep(0)  # one tick so short-lived contexts (and tests) observe it
                elif isinstance(payload, _core.PromptListChangedNotification):
                    logger.debug("MCP server '%s': prompts/list_changed (ignored)", self.name)
                elif isinstance(payload, _core.ResourceListChangedNotification):
                    logger.debug("MCP server '%s': resources/list_changed (ignored)", self.name)
            except Exception:
                logger.exception("Error in MCP message handler for '%s'", self.name)
        return _handler

    def _deregister_owned(self, tool_names: Iterable[str]) -> None:
        """Deregister *tool_names* this server's toolset still owns (never a colliding name owned by another server)."""
        from tools.registry import registry
        for tool_name in tool_names:
            if registry.get_toolset_for_tool(tool_name) == f"mcp-{self.name}":
                registry.deregister(tool_name, scope=_core._server_registry_scope(self.name))
                _forget_mcp_tool_server(tool_name)

    async def _refresh_tools(self):
        """Re-fetch tools on ``tools/list_changed`` and update the registry. The lock serializes rapid-fire
        notifications; after the list_tools ``await`` all mutations are synchronous — atomic on the event loop."""
        if not self._advertises_tools():
            return  # tools/list would raise MCPError(-32601)
        async with self._refresh_lock:
            old_tool_names = set(self._registered_tool_names)
            async with self._rpc_lock:
                new_mcp_tools = await _core._paginate_full_list(self.session.list_tools, "tools", self.name)
            # Remove only stale names first — no nuke-and-repave: live turns may hold tool-call
            # IDs pointing at existing handlers; in-place replacement avoids "not connected" races.
            self._deregister_owned(old_tool_names - {mcp_prefixed_tool_name(self.name, tool.name) for tool in new_mcp_tools})
            # Re-register; a raw name can become ambiguous after normalization without changing
            # its normalized name, so also drop old entries the final registration no longer owns.
            self._tools = new_mcp_tools
            registered_names = _registration._register_server_tools(self.name, self, self._config)
            self._deregister_owned(old_tool_names - set(registered_names))
            self._registered_tool_names = registered_names
            new_tool_names = set(registered_names)
            changes = [f"{label}: {', '.join(sorted(names))}" for label, names in
                       (("added", new_tool_names - old_tool_names), ("removed", old_tool_names - new_tool_names)) if names]
            if changes:
                logger.warning("MCP server '%s': tools changed dynamically — %s. "
                               "Verify these changes are expected.", self.name, "; ".join(changes))
            else:
                logger.info("MCP server '%s': dynamically refreshed %d tool(s) (no changes)",
                            self.name, len(self._registered_tool_names))

    async def _keepalive_probe(self) -> None:
        """Exercise the session; raise on a genuine connection failure. ``ping`` first (cheap,
        OPTIONAL); on -32601 latch ``_ping_unsupported`` (reset per transport connection) and fall
        back to ``list_tools`` when the server advertises tools, else the -32601 propagates."""
        async def list_tools():
            await asyncio.wait_for(self.session.list_tools(), timeout=_KEEPALIVE_RPC_TIMEOUT)
        if not self._ping_unsupported:
            try:
                await asyncio.wait_for(self.session.send_ping(), timeout=_KEEPALIVE_RPC_TIMEOUT)
                return
            except Exception as exc:
                if _is_method_not_found_error(exc):
                    if not self._advertises_tools():  # ping definitively unsupported, nothing to fall back to
                        raise
                    self._ping_unsupported = True
                    logger.info("MCP server '%s': does not implement the optional 'ping' utility (-32601); "
                                "using 'list_tools' for keepalive on this connection.", self.name)
                elif isinstance(exc, (TimeoutError, asyncio.TimeoutError)) and self._advertises_tools():
                    # A server that silently drops ping looks like a dead transport: confirm with
                    # list_tools before declaring it dead, else propagate the original failure.
                    try:
                        await list_tools()
                    except Exception:
                        raise exc from None
                    self._ping_unsupported = True  # latch so later keepalives skip the 30s wait
                    logger.info("MCP server '%s': ping timed out but list_tools succeeded — server "
                                "silently drops ping; using 'list_tools' for keepalive on this connection.", self.name)
                    return
                else:
                    raise  # closed transport, expired session, etc. — real failure
        await list_tools()

    def _mark_session_proven(self) -> None:
        """Record that the session demonstrated real health (keepalive or tool-call success).
        Only then is the reconnect budget cleared: a handshake that drops moments later must keep
        consuming ``_reconnect_retries`` so a flapping transport still reaches the park.

        Called from the keepalive success path (session survived at least one full keepalive interval) and
        the tool-call success path. See #62212.
        """
        if self._session_proven:
            return
        self._session_proven = True
        self._reconnect_retries = 0
        if self._was_parked:
            self._was_parked = False
            logger.warning("MCP server '%s': revived — session healthy again after "
                           "parking (state: parked → connected)", self.name)
        # A proven fresh transport clears the one-time permanent-failure grace and any race bookkeeping.
        self._permanent_grace_used = self._teardown_race = False

    def mark_suspect(self, reason: str) -> None:
        """Latch a suspicion (no I/O); the NEXT call verifies via :meth:`ensure_healthy` and recycles on failure.

        The NEXT call verifies via :meth:`ensure_healthy` and recycles the transport if the probe fails,
        instead of the connection silently staying poisoned until process restart (#81051/#77765/#84132).
        """
        if self._suspect_reason is None and reason:
            logger.warning("MCP server '%s': connection marked suspect (%s); next call will health-check it",
                           self.name, reason)
        self._suspect_reason = reason or None

    async def ensure_healthy(self, timeout: float = 5.0) -> bool:
        """Verify a suspect connection before reuse; recycle if dead. True when healthy (suspicion
        cleared). On failure requests a reconnect, drops the stale session so the caller's
        no-session path takes over, and returns False. Never raises."""
        reason = self._suspect_reason
        if not reason:
            return True
        if self.session is None:  # nothing to verify — the reconnect path owns recovery now
            self._suspect_reason = None
            self._reconnect_event.set()
            return False
        try:
            await asyncio.wait_for(self._keepalive_probe(), timeout=timeout)
        except Exception as exc:
            root = _unwrap_exception_group(exc)
            logger.warning("MCP server '%s': suspect connection (%s) failed health check (%s: %s) — "
                           "requesting reconnect (state: suspect → degraded)",
                           self.name, reason, type(root).__name__, root)
            self._suspect_reason = None
            self.mark_suspect(f"health check failed after {reason}")
            self.session = None
            self._ready.clear()
            self._reconnect_event.set()
            return False
        logger.info("MCP server '%s': suspect connection passed health check (%s) — clearing suspicion",
                    self.name, reason)
        self._suspect_reason = None
        self._mark_session_proven()
        return True

    def _fail_inflight_calls(self, reason: str) -> None:
        """Cancel every in-flight RPC BEFORE the transport unwinds: the SDK does not always fail
        pending requests when streams close, so a call would otherwise wait out the full tool
        timeout. Cancelling anything flags ``_teardown_race`` so run() treats the next reconnect
        as recovery rather than charging the rapid-drop budget."""
        victims = [t for t in self._inflight_tasks if not t.done()]
        if not victims:
            return
        self._reconnecting = self._teardown_race = True
        self.mark_suspect(f"{reason} tore down {len(victims)} in-flight call(s)")
        for task in victims:
            task.cancel()

    def _stdio_children_dead(self) -> bool:
        """True when every stdio child we spawned has exited. Best-effort: False (unknown → don't
        fail fast) for HTTP, no captured PIDs, missing psutil, or a failed probe."""
        pids = getattr(self, "_stdio_child_pids", None)
        if not pids or self._is_http():
            return False
        try:
            import psutil
            return not any(psutil.pid_exists(pid) for pid in pids)  # Windows-safe, no signal noise
        except Exception:  # missing psutil or failed probe → unknown → don't fail fast
            return False

    async def _watch_stdio_children(self) -> None:
        """Poll child liveness during a stdio RPC; resolves when a tracked child dies so the caller cancels the RPC.

        See #81995.
        """
        while not self._stdio_children_dead():
            # Async context — never block the loop (#36163).
            await asyncio.sleep(0.25)
