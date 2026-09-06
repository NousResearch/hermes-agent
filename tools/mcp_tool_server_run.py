"""Lifecycle of :class:`tools.mcp_tool.MCPServerTask`: the long-lived ``run`` state machine
(connect -> serve -> reconnect/park/recycle), keepalive-driven lifecycle waits, start/shutdown
and tool deregistration. Origin state and patchable helpers are read through ``_core`` so
``mock.patch("tools.mcp_tool.X")`` keeps working."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional
from tools.mcp_tool_common import _core, _get_lifecycle_seconds, _jittered, _resolve_tool_timeout
from tools import mcp_tool_errors as _errors
from tools import mcp_tool_registration as _registration
from tools import mcp_tool_sampling as _sampling

logger = logging.getLogger("tools.mcp_tool")


@dataclass
class _RetryBudget:
    """Per-run() retry counters (``_reconnect_retries`` stays on the task: handlers/tests read it)."""

    initial_retries: int = 0
    backoff: float = 1.0


class MCPServerRunMixin:
    """Methods of :class:`tools.mcp_tool.MCPServerTask` (mixed in; relies on its attributes)."""

    @staticmethod
    async def _cancel_waiters(*tasks: asyncio.Task) -> None:
        for t in tasks:
            if not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    def _event_waiters(self) -> tuple:
        """Fresh ``(shutdown, reconnect)`` wait tasks; cancel them via ``_cancel_waiters``."""
        return (asyncio.ensure_future(self._shutdown_event.wait()),
                asyncio.ensure_future(self._reconnect_event.wait()))

    def _recycle_if_due(self) -> bool:
        """Latch a stdio idle/lifetime recycle when its deadline has passed."""
        recycle_reason = self._stdio_recycle_reason()
        if recycle_reason is None:
            return False
        self._mark_stdio_recycled(recycle_reason)
        return True

    async def _wait_for_lifecycle_event(self) -> str:
        """Serve until a lifecycle event: ``"shutdown"`` (exits run), ``"reconnect"`` (session torn
        down, transport re-entered; event cleared first) or ``"recycle"`` (stdio idle/lifetime
        limit; restarts lazily on next call). Shutdown wins a tie. A keepalive (``ping``,
        list_tools fallback) runs every ``keepalive_interval`` (must stay below the server's
        session TTL); a failure triggers a reconnect.

        Periodically sends a lightweight keepalive (``ping``, with a ``list_tools`` fallback for servers
        that don't implement the optional ping utility — see :meth:`_keepalive_probe`) to prevent
        TCP/session state from going stale during idle periods (#17003).
        """
        keepalive_interval = max(
            _core._MIN_KEEPALIVE_INTERVAL,
            float(self._config.get("keepalive_interval", _core._DEFAULT_KEEPALIVE_INTERVAL)))
        shutdown_task, reconnect_task = self._event_waiters()
        try:
            while True:
                if self._recycle_if_due():
                    return "recycle"
                timeout = keepalive_interval
                recycle_deadline = self._next_stdio_recycle_deadline()
                if recycle_deadline is not None:
                    timeout = max(0.0, min(timeout, recycle_deadline - time.monotonic()))
                done, _pending = await asyncio.wait(
                    {shutdown_task, reconnect_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if done:
                    break
                if self._recycle_if_due():
                    return "recycle"
                # Timeout: probe for a stale session — NEVER while an RPC is in flight (a
                # concurrent ping can wedge the stdio stream; a busy server is alive anyway).
                # Timeout — no lifecycle event fired. See #48069.
                if self.session:
                    if self._rpc_lock.locked() or any(not t.done() for t in self._inflight_tasks):
                        continue
                    try:
                        async with self._rpc_lock:
                            await self._keepalive_probe()
                    except Exception as exc:
                        root = _errors._unwrap_exception_group(exc)
                        logger.warning("MCP server '%s' keepalive failed, triggering reconnect (state: connected → "
                                       "degraded): %s: %s", self.name, type(root).__name__, root)
                        self.mark_suspect(f"keepalive failed: {type(root).__name__}: {root}")
                        self._reconnect_event.set()
                        break
                    # Survived a full keepalive interval: real proof of health.
                    # Clear the rapid-drop budget (#62212).
                    self._mark_session_proven()
        finally:
            await self._cancel_waiters(shutdown_task, reconnect_task)
        if self._shutdown_event.is_set():
            self._fail_inflight_calls("shutdown")
            return "shutdown"
        # Deliberate teardown: fail in-flight RPCs NOW instead of riding out the tool timeout.
        # See #48069, #81995.
        self._fail_inflight_calls("reconnect")
        self._reconnect_event.clear()
        return "reconnect"

    async def _wait_for_reconnect_or_shutdown(self, timeout: Optional[float] = None) -> str:
        """Parked wait: ``"shutdown"`` or ``"reconnect"`` (explicit, or the ``timeout`` self-probe;
        event cleared first). Shutdown wins a tie."""
        shutdown_task, reconnect_task = self._event_waiters()
        try:
            await asyncio.wait({shutdown_task, reconnect_task}, return_when=asyncio.FIRST_COMPLETED, timeout=timeout)
        finally:
            await self._cancel_waiters(shutdown_task, reconnect_task)
        if self._shutdown_event.is_set():
            return "shutdown"
        self._reconnect_event.clear()
        return "reconnect"

    async def _park(self, revival_reason: str) -> bool:
        """Drop this server's tools and wait for a reconnect request; True when shutdown came instead.
        The run task must NOT exit (it is the only ``_reconnect_event`` listener, so returning
        leaves the server unrevivable). With tools deregistered no call can reach the breaker
        probe, so the wait is TIMED (one self-probe per ``_PARKED_RETRY_INTERVAL``); an explicit
        ``_reconnect_event.set()`` wakes it immediately."""
        # Do NOT return — exiting the task orphans the server: nothing would ever listen for
        # _reconnect_event again and the server would be permanently wedged for the life of the process
        # (#16788). Instead, drop the phantom tools from the registry and park. Because parking deregisters
        # the tools, no tool call can reach the circuit-breaker half-open probe or _signal_reconnect — so
        # the park is a TIMED wait: every _PARKED_RETRY_INTERVAL we wake and attempt one reconnect ourselves
        # (#57129). An explicit _reconnect_event.set() (OAuth recovery, manual /mcp refresh) still wakes us
        # immediately.
        self._was_parked = True
        self._deregister_tools()
        self._reconnect_event.clear()
        if await self._wait_for_reconnect_or_shutdown(timeout=_core._PARKED_RETRY_INTERVAL) == "shutdown":
            return True
        logger.debug("MCP server '%s': attempting revival %s (self-probe or explicit reconnect request); "
                     "rebuilding transport.", self.name, revival_reason)
        return False

    async def _prepare_run(self, config: dict) -> bool:
        """Bind config, build sampling/elicitation handlers, validate HTTP. False when the server
        must not start (bad remote URL / non-MCP endpoint: fail fast with ``_error`` set and
        ``_ready`` fired instead of burning the reconnect ladder inside the SDK's httpx layer)."""
        self._config = config
        self.tool_timeout = _resolve_tool_timeout(config)
        self._auth_type = (config.get("auth") or "").lower().strip()
        self._idle_timeout_seconds = _get_lifecycle_seconds(config, "idle_timeout_seconds")
        self._max_lifetime_seconds = _get_lifecycle_seconds(config, "max_lifetime_seconds")
        # The _MCP_*_TYPES flags are False until the lazy SDK import runs.
        _core._ensure_mcp_sdk()
        sampling_config = config.get("sampling", {})
        self._sampling = (_sampling.SamplingHandler(self.name, sampling_config)
                          if sampling_config.get("enabled", True) and _core._MCP_SAMPLING_TYPES else None)
        # elicitation/create lets a server ask for structured input mid-call; the handler
        # routes it through Hermes' approval system.
        elicitation_config = config.get("elicitation", {})
        self._elicitation = (_sampling.ElicitationHandler(self.name, elicitation_config, owner=self)
                             if elicitation_config.get("enabled", True) and _core._MCP_ELICITATION_TYPES else None)
        if "url" in config and "command" in config:
            logger.warning("MCP server '%s' has both 'url' and 'command' in config. Using HTTP transport "
                           "('url'). Remove 'command' to silence this warning.", self.name)
        if not self._is_http():
            return True
        try:
            _errors._validate_remote_mcp_url(self.name, config.get("url"))
            # Content-type preflight (Streamable HTTP only; SSE serves text/event-stream): a
            # web-app root returns HTML and would hang the SDK for connect_timeout. Skipped once
            # _ready was ever set and for OAuth servers (a token-less probe sees HTML/401).
            if (config.get("transport") != "sse" and not config.get("skip_preflight")
                    and not self._ready.is_set() and self._auth_type != "oauth"):
                await self._preflight_content_type(
                    config["url"], headers=dict(config.get("headers") or {}),
                    ssl_verify=config.get("ssl_verify", True),
                    client_cert=_errors._resolve_client_cert(self.name, config))
        except (_errors.InvalidMcpUrlError, _errors.NonMcpEndpointError) as exc:
            logger.warning("%s", exc)
            self._publish_error(exc)  # fail fast and non-retryably
            return False
        return True

    def _publish_error(self, exc: BaseException) -> None:
        """Hand *exc* to the waiting ``start()``."""
        self._error = exc
        self._ready.set()

    async def run(self, config: dict):
        """Long-lived: connecting -> connected -> (degraded -> parked -> revived)*. Unproven drops
        and transport errors charge a rapid-drop budget with jittered backoff; exhausting it (or
        a permanent error) parks via :meth:`_park` rather than exiting, so the server stays
        revivable. Branch helpers return True to keep looping, False to exit."""
        if not await self._prepare_run(config):
            return
        self._reconnect_retries = 0
        budget = _RetryBudget()
        while True:
            try:
                run_transport = self._run_http if self._is_http() else self._run_stdio
                if not await self._on_clean_return(await run_transport(config), budget):
                    break
            except asyncio.CancelledError:
                # Not a connection failure: re-raise so shutdown()'s ``await self._task`` completes.
                # Task was cancelled (shutdown, gateway restart, explicit task.cancel()). Don't treat this
                # as a connection failure — CancelledError inherits from BaseException (not Exception) in
                # Python 3.11+, so the broad ``except Exception`` below would NOT catch it; we'd silently
                # exit the reconnect loop and the MCP server would stay dead until Hermes is fully
                # restarted. See #9930.
                self.session = None
                raise
            except Exception as exc:
                self.session = None
                if not await self._on_transport_error(exc, budget):
                    break
            finally:
                self.session = None
                # Stale PIDs must never fast-fail the NEXT transport's calls.
                self._stdio_child_pids = set()

    async def _on_clean_return(self, lifecycle_reason: str, budget: "_RetryBudget") -> bool:
        """Clean transport return: shutdown, stdio recycle, or a requested rebuild (not a failure
        for the retry counters)."""
        if self._shutdown_event.is_set():
            return False
        if lifecycle_reason == "recycle":
            logger.info("MCP server '%s': stdio session recycled after %s; waiting for lazy reconnect",
                        self.name, self._recycled_reason)
            self.session = None
            # Dormant until a lazy call wakes it (untimed: nothing to self-probe).
            return await self._wait_for_reconnect_or_shutdown() != "shutdown"
        # Per-cycle chatter stays DEBUG; WARNINGs mark state transitions.
        logger.debug("MCP server '%s': reconnecting (OAuth recovery or manual refresh)", self.name)
        # A clean return is NOT proof of health (a flapper handshakes fine, then drops). Only a
        # PROVEN session clears the budget; a teardown race is recovery, never a park charge.
        # A clean transport return means a session was established and then asked to rebuild (auth recovery
        # / manual refresh / keepalive failure / transport TaskGroup drop). That alone is NOT proof of
        # health: a flapping transport handshakes fine and drops moments later, and resetting the budget
        # here let such servers respawn forever (#62212 — 6212 spawns in 63h). Only clear the
        # consecutive-failure budget once the session PROVED healthy — survived >=1 full keepalive interval
        # or served >=1 successful tool call (_mark_session_proven).
        if self._teardown_race and not self._session_proven:
            logger.info("MCP server '%s': reconnect after teardown race (in-flight calls were failed); "
                        "not charging the rapid-drop budget", self.name)
            self._teardown_race, budget.backoff = False, 1.0
        elif self._session_proven:
            self._reconnect_retries, budget.backoff = 0, 1.0
        else:
            self._reconnect_retries += 1
            if self._reconnect_retries > _core._MAX_RECONNECT_RETRIES:
                logger.warning(
                    "MCP server '%s': %d consecutive reconnects without a healthy session (rapid-drop budget "
                    "exhausted), parking; will self-probe every %ds until it recovers (state: degraded → parked)",
                    self.name, _core._MAX_RECONNECT_RETRIES, _core._PARKED_RETRY_INTERVAL)
                if not await self._park_and_rearm("from parked state", budget):
                    return False
        # Clear readiness too: a stale _ready lets handler recovery mistake old for fresh.
        self._ready.clear()
        self.session = None
        return True

    async def _park_and_rearm(self, revival_reason: str, budget: "_RetryBudget") -> bool:
        """Park; on revival leave ONE probe per wake so a still-dead server re-parks instead of
        burning 5 rapid retries. False on shutdown."""
        if await self._park(revival_reason):
            return False
        self._reconnect_retries, budget.backoff = _core._MAX_RECONNECT_RETRIES, 1.0
        return True

    async def _park_initial_failure(self, exc: Exception, revival_reason: str, budget: "_RetryBudget") -> bool:
        """Publish ``exc`` to ``start()``, park, and on revival reset every counter. False on shutdown."""
        self._publish_error(exc)
        if await self._park(revival_reason):
            return False
        budget.initial_retries = self._reconnect_retries = 0
        budget.backoff = 1.0
        self._error = None
        self._ready.clear()
        return True

    async def _backoff_sleep(self, budget: "_RetryBudget") -> None:
        await asyncio.sleep(_jittered(budget.backoff))
        budget.backoff = min(budget.backoff * 2, _core._MAX_BACKOFF_SECONDS)

    async def _on_transport_error(self, exc: Exception, budget: "_RetryBudget") -> bool:
        """Transport raised: classify, then run the initial-connect or reconnect ladder. False = exit."""
        # Unwrap anyio TaskGroup wrappers: the group's str() hides the root cause.
        root = _errors._unwrap_exception_group(exc)
        failure_class = _errors._classify_mcp_failure(root)
        if self._is_recycled_stdio():
            logger.warning("MCP server '%s': lazy reconnect after stdio recycle failed, marking unavailable "
                           "while retrying: %s: %s", self.name, type(root).__name__, root)
            self._recycled_reason = None
        # Initial-connect ladder (a startup blip must not kill the server); gated on
        # _ever_connected, not _ready (which clears every reconnect cycle).
        # If this is the first connection attempt, retry with backoff before giving up. Gated on
        # ``_ever_connected`` rather than ``_ready`` — ``_ready`` is cleared on every reconnect cycle (see
        # below), so a server that already registered tools once and then dropped would otherwise be
        # misclassified as never having connected and re-enter this initial-connect ladder (#94654).
        # ``_ever_connected`` itself is set once and never cleared. (Ported from Kilo Code's MCP resilience
        # fix.)
        if not self._ever_connected:
            return await self._on_initial_connect_error(exc, root, failure_class, budget)
        if self._shutdown_event.is_set():
            logger.debug("MCP server '%s' disconnected during shutdown: %s: %s",
                         self.name, type(root).__name__, root)
            return False
        if failure_class == "permanent":
            return await self._on_permanent_error(root, budget)
        self._reconnect_retries += 1
        if self._reconnect_retries > _core._MAX_RECONNECT_RETRIES:
            logger.warning(
                "MCP server '%s' failed after %d reconnection attempts, parking; will self-probe every %ds "
                "until it recovers (state: degraded → parked): %s: %s",
                self.name, _core._MAX_RECONNECT_RETRIES, _core._PARKED_RETRY_INTERVAL, type(root).__name__, root)
            return await self._park_and_rearm("from parked state", budget)
        logger.debug("MCP server '%s' connection lost (attempt %d/%d), reconnecting in %.0fs: %s: %s",
                     self.name, self._reconnect_retries, _core._MAX_RECONNECT_RETRIES, budget.backoff,
                     type(root).__name__, root)
        await self._backoff_sleep(budget)
        return not self._shutdown_event.is_set()

    async def _on_initial_connect_error(self, exc: Exception, root: BaseException,
                                        failure_class: str, budget: "_RetryBudget") -> bool:
        if failure_class == "permanent":
            # Deterministic failure (bad command, non-MCP URL, 401/403): park at once; auth
            # failures park (not return) so the task can pick up fresh tokens later.
            detail = (f"authentication, parking until credentials change; re-authenticate with "
                      f"`hermes mcp login {self.name}`" if _errors._is_auth_error(root)
                      else "connection with a permanent error, parking without retries")
            logger.warning("MCP server '%s' failed initial %s (state: connecting → parked): %s: %s",
                           self.name, detail, type(root).__name__, root)
            return await self._park_initial_failure(exc, "after permanent initial failure", budget)
        budget.initial_retries += 1
        if budget.initial_retries > _core._MAX_INITIAL_CONNECT_RETRIES:
            logger.warning(
                "MCP server '%s' failed initial connection after %d attempts, parking until a reconnect is "
                "requested (state: connecting → parked): %s: %s",
                self.name, _core._MAX_INITIAL_CONNECT_RETRIES, type(root).__name__, root)
            return await self._park_initial_failure(exc, "after initial connection failures", budget)
        logger.debug(
            "MCP server '%s' initial connection failed (attempt %d/%d), retrying in %.0fs: %s: %s",
            self.name, budget.initial_retries, _core._MAX_INITIAL_CONNECT_RETRIES, budget.backoff,
            type(root).__name__, root)
        await self._backoff_sleep(budget)
        if self._shutdown_event.is_set():
            self._publish_error(exc)
        return not self._shutdown_event.is_set()

    async def _on_permanent_error(self, root: BaseException, budget: "_RetryBudget") -> bool:
        # Auth failure on a PROVEN session is often a raced-teardown OAuth lock, not revoked
        # credentials: grant ONE suspect+reconnect cycle first.
        if _errors._is_auth_error(root) and self._session_proven and not self._permanent_grace_used:
            self._permanent_grace_used = True
            self.mark_suspect(f"auth error on proven session: {root}")
            logger.warning(
                "MCP server '%s': auth error on a previously healthy session — marking suspect and forcing "
                "one reconnect instead of parking (state: connected → suspect): %s: %s",
                self.name, type(root).__name__, root)
            self._reconnect_retries, budget.backoff = 0, 1.0
            await asyncio.sleep(_jittered(1.0))
            return not self._shutdown_event.is_set()
        # Deterministic failure on a working server: park now.
        logger.warning(
            "MCP server '%s' hit a permanent error, parking without retries; will self-probe every %ds "
            "(state: connected → parked): %s: %s", self.name, _core._PARKED_RETRY_INTERVAL, type(root).__name__, root)
        return await self._park_and_rearm("from parked state (permanent error)", budget)

    async def start(self, config: dict):
        """Create the background Task and wait until ready (or failed)."""
        self._task = asyncio.ensure_future(self.run(config))
        try:
            await self._ready.wait()
        except asyncio.CancelledError:
            # The caller's connect timeout cancels *this* coroutine; the ensure_future'd run()
            # task would otherwise keep running detached on a hung transport with no owner.
            # Propagate so the transport context managers unwind and release child / FDs.
            if self._task and not self._task.done():
                self._task.cancel()
            raise
        if self._error:
            raise self._error

    async def shutdown(self):
        """Signal the Task to exit and wait for clean resource teardown."""
        self._shutdown_event.set()
        # Also set reconnect: closes any race where _wait_for_lifecycle_event misses the
        # shutdown flag after returning "reconnect".
        self._reconnect_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning("MCP server '%s' shutdown timed out, cancelling task", self.name)
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        if self._pending_refresh_tasks:
            for task in list(self._pending_refresh_tasks):
                task.cancel()
            await asyncio.gather(*self._pending_refresh_tasks, return_exceptions=True)
            self._pending_refresh_tasks.clear()
        self._deregister_tools()
        self.session = None

    def _deregister_tools(self) -> None:
        """Drop this server's tools from the registry (idempotent); on shutdown AND budget
        exhaustion, so a dead server never leaves phantom tools in the prompt."""
        from tools.registry import registry
        for tool_name in list(getattr(self, "_registered_tool_names", [])):
            registry.deregister(tool_name, scope=_core._server_registry_scope(self.name))
            _registration._forget_mcp_tool_server(tool_name)
        self._registered_tool_names = []
