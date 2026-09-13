#!/usr/bin/env python3
"""MCP (Model Context Protocol) client: connects to the ``mcp_servers`` configured in
~/.hermes/config.yaml (stdio, Streamable HTTP or SSE), discovers their tools and registers them
into the hermes tool registry. The ``mcp`` package is optional (no-op without it).

One background event loop (``_mcp_loop``) in a daemon thread runs each server as a long-lived
Task (``MCPServerTask``) so the transport's anyio cancel scopes enter and exit in one Task; every
``_servers``/loop mutation holds ``_lock``. This module keeps the SDK loader, ``MCPServerTask`` and
all shared state; the ``mcp_tool_*`` siblings read that state back through ``tools.mcp_tool`` at
call time (``_core``) and are imported directly by their callers."""

import asyncio
import contextvars
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

from tools.mcp_tool_common import _DEFAULT_TOOL_TIMEOUT, mcp_field
from tools.mcp_tool_config import _get_mcp_stderr_log, _npx_cached_bin
from tools.mcp_tool_sampling import ElicitationHandler, SamplingHandler
from tools.mcp_tool_transport import MCPServerTransportMixin
from tools.mcp_tool_server_run import MCPServerRunMixin
from tools.mcp_tool_health import MCPServerHealthMixin


# Wall-clock bound on the fail-open OSV malware preflight before a stdio spawn; just ABOVE
# osv_check._TIMEOUT (10s) so it only bites when a stalled SSL handshake defeats that.
_OSV_MALWARE_CHECK_TIMEOUT_S = 12.0


async def _preflight_stdio_command(server_name: str, command: str, args: list) -> tuple[str, list]:
    """OSV malware preflight (off-loop, wall-clock bound, fail-open on timeout), THEN the
    cached-npx swap. The preflight must see the REAL command/args: anything that rewrites argv to a
    wrapper or resolved binary has to happen after it, or the check silently inspects the wrapper
    and becomes a no-op (``_infer_ecosystem`` keys off the command basename being npx/uvx/pipx)."""
    from tools.osv_check import check_package_for_malware
    try:
        malware_error = await asyncio.wait_for(
            asyncio.to_thread(check_package_for_malware, command, args), timeout=_OSV_MALWARE_CHECK_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("MCP server '%s': OSV malware preflight timed out after %.0fs "
                       "(network slow/unreachable) — proceeding without the check.",
                       server_name, _OSV_MALWARE_CHECK_TIMEOUT_S)
        malware_error = None
    if malware_error:
        raise ValueError(f"MCP server '{server_name}': {malware_error}")

    # npx resolves the package and then FORKS, staying resident as the real server's parent for
    # nothing (~48 MB per server, measured). Hermes already supervises the child (shared death
    # supervisor), so a cached package is spawned directly; a cache miss leaves npx untouched.
    if os.path.basename(command).lower().startswith("npx"):
        cached = _npx_cached_bin(args)
        if cached:
            direct_command, direct_args = cached
            logger.debug("MCP server '%s': using cached npx binary %s (skipping the "
                         "resident `npm exec` parent)", server_name, direct_command)
            command, args = direct_command, direct_args
    return command, args


# ---- Optional MCP SDK: availability probe now, symbol import on first use ----

_MCP_AVAILABLE = _MCP_HTTP_AVAILABLE = _MCP_NEW_HTTP = _MCP_LEGACY_HTTP = False
_MCP_SAMPLING_TYPES = _MCP_NOTIFICATION_TYPES = _MCP_ELICITATION_TYPES = False
_MCP_MESSAGE_HANDLER_SUPPORTED = _MCP_LOGGING_CALLBACK_SUPPORTED = False
sse_client = None
# Fallback for SDKs without LATEST_PROTOCOL_VERSION (Streamable HTTP arrived with 2025-03-26).
LATEST_PROTOCOL_VERSION = "2025-03-26"
# Newest revision ``ClientSession.initialize()`` speaks; from 2026-07-28 the handshake is a
# per-request envelope so this can be OLDER than LATEST_PROTOCOL_VERSION, and the
# MCP-Protocol-Version header must be seeded from THIS one.
LATEST_HANDSHAKE_VERSION = LATEST_PROTOCOL_VERSION

# Importing ``mcp`` costs ~260ms, so it is deferred to first use (_ensure_mcp_sdk); availability
# is decided now via find_spec so every ``if not _MCP_AVAILABLE`` gate / patch / skipif holds.
try:
    _MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None
except Exception:
    _MCP_AVAILABLE = False
if not _MCP_AVAILABLE:
    logger.debug("mcp package not installed -- MCP tool support disabled")

ClientSession: Any = None
_MCP_SDK_IMPORT_ATTEMPTED = False
_MCP_SDK_IMPORT_LOCK = threading.Lock()

# Optional SDK type families (module, names, debug message when absent), bound in this order to
# _MCP_SAMPLING_TYPES / _MCP_ELICITATION_TYPES / _MCP_NOTIFICATION_TYPES; an older SDK only
# loses that feature, not MCP.
_OPTIONAL_TYPE_FAMILIES = (
    ("mcp.types", ("CreateMessageResult", "CreateMessageResultWithTools", "ErrorData", "SamplingCapability",
                   "SamplingToolsCapability", "TextContent", "ToolUseContent"),
     "MCP sampling types not available -- sampling disabled"),
    ("mcp.types", ("ElicitRequestParams", "ElicitResult"),
     "MCP elicitation types not available -- elicitation disabled"),
    ("mcp.types", ("ServerNotification", "ToolListChangedNotification", "PromptListChangedNotification",
                   "ResourceListChangedNotification"),
     "MCP notification types not available -- dynamic tool discovery disabled"),
)
# Bound by _ensure_mcp_sdk(); module __getattr__ (PEP 562) imports the SDK on first external
# access so mock.patch("tools.mcp_tool.stdio_client") sees a real original, never clobbered.
_MCP_SDK_LAZY_SYMBOLS = frozenset(
    {"StdioServerParameters", "stdio_client", "streamablehttp_client", "streamable_http_client"}
    | {n for _mod, names, _msg in _OPTIONAL_TYPE_FAMILIES for n in names})


def __getattr__(name: str):
    if name in _MCP_SDK_LAZY_SYMBOLS:
        _ensure_mcp_sdk()
        try:
            return globals()[name]
        except KeyError:
            pass  # SDK missing or symbol absent on this SDK build
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _import_sdk_names(module: str, names: tuple, missing_msg: Optional[str] = None) -> bool:
    """Bind ``names`` from SDK ``module`` into this module's globals; False (nothing bound,
    optional debug line) when this SDK build lacks the module or any of the names."""
    try:
        mod = importlib.import_module(module)
        values = {n: getattr(mod, n) for n in names}
    except (ImportError, AttributeError):
        if missing_msg:
            logger.debug(missing_msg)
        return False
    globals().update(values)
    return True


def _ensure_mcp_sdk() -> bool:
    """Import the optional ``mcp`` SDK on first use; return availability. Idempotent and
    thread-safe; honors a test-patched ``_MCP_AVAILABLE=False`` (no import) and pre-installed
    mocks (``ClientSession`` already set means no re-import)."""
    global _MCP_SDK_IMPORT_ATTEMPTED, _MCP_AVAILABLE, _MCP_HTTP_AVAILABLE, _MCP_NEW_HTTP, _MCP_LEGACY_HTTP
    global _MCP_SAMPLING_TYPES, _MCP_NOTIFICATION_TYPES, _MCP_ELICITATION_TYPES, sse_client
    global _MCP_MESSAGE_HANDLER_SUPPORTED, _MCP_LOGGING_CALLBACK_SUPPORTED, LATEST_HANDSHAKE_VERSION
    global _JSONRPC_METHOD_NOT_FOUND
    if not _MCP_AVAILABLE:
        return False
    if _MCP_SDK_IMPORT_ATTEMPTED or ClientSession is not None:
        return _MCP_AVAILABLE
    with _MCP_SDK_IMPORT_LOCK:
        if _MCP_SDK_IMPORT_ATTEMPTED or ClientSession is not None:
            return _MCP_AVAILABLE
        if (_import_sdk_names("mcp", ("ClientSession", "StdioServerParameters"))
                and _import_sdk_names("mcp.client.stdio", ("stdio_client",))):
            _MCP_AVAILABLE = True
            # mcp >= 1.24 ships streamable_http_client; 2.0 dropped the deprecated
            # streamablehttp_client alias. Either one gives HTTP.
            _MCP_NEW_HTTP = _import_sdk_names("mcp.client.streamable_http", ("streamable_http_client",))
            _MCP_LEGACY_HTTP = _import_sdk_names("mcp.client.streamable_http", ("streamablehttp_client",))
            _MCP_HTTP_AVAILABLE = _MCP_NEW_HTTP or _MCP_LEGACY_HTTP
            _import_sdk_names("mcp.types", ("LATEST_PROTOCOL_VERSION",),
                              "mcp.types.LATEST_PROTOCOL_VERSION not available -- using fallback protocol version")
            if not _import_sdk_names("mcp.client.session", ("LATEST_HANDSHAKE_VERSION",)):
                # Pre-2.x SDKs: newest revision IS the handshake revision.
                LATEST_HANDSHAKE_VERSION = LATEST_PROTOCOL_VERSION
            if not _import_sdk_names("mcp.client.sse", ("sse_client",),
                                     "mcp.client.sse.sse_client not available -- SSE transport disabled"):
                sse_client = None
            _MCP_SAMPLING_TYPES, _MCP_ELICITATION_TYPES, _MCP_NOTIFICATION_TYPES = [
                _import_sdk_names(*family) for family in _OPTIONAL_TYPE_FAMILIES]
        else:
            logger.debug("mcp package not installed -- MCP tool support disabled")
        if _MCP_AVAILABLE:
            try:
                _JSONRPC_METHOD_NOT_FOUND = importlib.import_module("mcp.types").METHOD_NOT_FOUND
            except Exception:  # pragma: no cover — SDK without the constant
                pass
        _MCP_MESSAGE_HANDLER_SUPPORTED = _client_session_accepts("message_handler")
        if _MCP_AVAILABLE and not _MCP_MESSAGE_HANDLER_SUPPORTED:
            logger.debug("MCP SDK does not support message_handler -- dynamic tool discovery disabled")
        _MCP_LOGGING_CALLBACK_SUPPORTED = _client_session_accepts("logging_callback")
        _MCP_SDK_IMPORT_ATTEMPTED = True
        return _MCP_AVAILABLE


_SDK_HTTPX_MOD = None


def sdk_httpx():
    """The httpx module the *installed* MCP SDK is built against (mcp 2.0 moved to ``httpx2``).
    Every object crossing the SDK boundary (AsyncClient, OAuth Request, exception classes) must
    come from the module the SDK itself imports or it fails at the transport layer. Resolved
    from the SDK's transport module, else the newest present; ``None`` if neither imports."""
    global _SDK_HTTPX_MOD
    if _SDK_HTTPX_MOD is not None:
        return _SDK_HTTPX_MOD
    try:
        from mcp.client import streamable_http as _transport
        _SDK_HTTPX_MOD = getattr(_transport, "httpx2", None) or getattr(_transport, "httpx", None)
    except ImportError:
        _SDK_HTTPX_MOD = None
    for fallback in ("httpx2", "httpx"):
        if _SDK_HTTPX_MOD is not None:
            break
        try:
            _SDK_HTTPX_MOD = importlib.import_module(fallback)
        except ImportError:
            pass
    return _SDK_HTTPX_MOD


def _client_session_accepts(kwarg: str) -> bool:
    """Whether this SDK's ``ClientSession.__init__`` takes ``kwarg`` (older SDKs lack
    ``message_handler`` and ``logging_callback``)."""
    if not _MCP_AVAILABLE:
        return False
    try:
        return kwarg in inspect.signature(ClientSession).parameters
    except (TypeError, ValueError):
        return False


# MCP logging levels (RFC 5424 syslog severities) -> Python logging levels.
# Port of anomalyco/opencode#34529's serverLog mapping.
_MCP_LOG_LEVEL_MAP = {
    "debug": logging.DEBUG, "info": logging.INFO, "notice": logging.INFO,
    "warning": logging.WARNING, "error": logging.ERROR, "critical": logging.ERROR,
    "alert": logging.ERROR, "emergency": logging.ERROR}

# ---- Reconnect / keepalive tuning ----

_DEFAULT_CONNECT_TIMEOUT = 60    # seconds for initial connection per server
_MAX_RECONNECT_RETRIES = 5
_MAX_INITIAL_CONNECT_RETRIES = 3 # retries for the very first connection attempt
_MAX_BACKOFF_SECONDS = 60
_RECYCLED_RECONNECT_TIMEOUT = 15.0
# Parked servers (tools deregistered) self-probe on this cadence: nothing else can revive them.
_PARKED_RETRY_INTERVAL = 300
# Bounded wait for a respawned stdio child when a call finds it dead (gateway restarts kill
# every MCP child); bounded so a broken server still parks via run()'s rapid-drop budget.
_STDIO_RESPAWN_WAIT_SEC = 15.0
# The client MUST ping faster than the server's idle-session TTL (short-TTL servers need a
# smaller configured ``keepalive_interval``); the floor stops a tiny interval busy-looping.
_DEFAULT_KEEPALIVE_INTERVAL, _MIN_KEEPALIVE_INTERVAL = 180, 5
# One bounded cancellation cycle at final shutdown so resistant tasks cannot hang exit.
_MCP_LOOP_DRAIN_TIMEOUT = 3.0
# JSON-RPC 2.0 "method not found" (server without optional ``ping``); _ensure_mcp_sdk()
# overrides it from mcp.types once loaded.
_JSONRPC_METHOD_NOT_FOUND = -32601
# nextCursor pagination cap so a forever-cursor cannot spin discovery (50 pages = thousands).
_MCP_LIST_MAX_PAGES = 50


async def _paginate_full_list(list_method, items_attr: str, server_name: str,
                              cache_meta_out: Optional[dict] = None):
    """Drain a paginated ``list_*`` call by following ``nextCursor``; ``cache_meta_out`` gets the
    first page's SEP-2549 hints. Callers must hold the server's ``_rpc_lock``."""
    items: list = []
    cursor = None
    for _ in range(_MCP_LIST_MAX_PAGES):
        if not cursor:
            result = await list_method()
        else:
            # mcp 2.0 takes params=PaginatedRequestParams, 1.x takes cursor=.
            # Inspect before awaiting: an internal TypeError is not a signature mismatch.
            import inspect

            try:
                signature = inspect.signature(list_method)
            except (TypeError, ValueError):
                accepts_params = True  # Opaque callables use the current SDK convention.
            else:
                accepts_params = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    or (p.name == "params" and p.kind != inspect.Parameter.POSITIONAL_ONLY)
                    for p in signature.parameters.values()
                )
            if accepts_params:
                import mcp.types as _types  # late: keeps the SDK import lazy
                _params_cls = getattr(_types, "PaginatedRequestParams", None)
                if _params_cls is not None:
                    result = await list_method(params=_params_cls(cursor=cursor))
                else:
                    result = await list_method(cursor=cursor)
            else:
                result = await list_method(cursor=cursor)
        if cache_meta_out is not None and not items:
            for key, snake, camel in (("ttl_ms", "ttl_ms", "ttlMs"), ("cache_scope", "cache_scope", "cacheScope")):
                hint = mcp_field(result, snake, camel)
                if hint is not None:
                    cache_meta_out[key] = hint
        items.extend(getattr(result, items_attr, None) or [])
        cursor = mcp_field(result, "next_cursor", "nextCursor")
        # Cursor is an opaque string; anything else (incl. mocks) = last page.
        if not isinstance(cursor, str) or not cursor:
            break
    else:
        logger.warning("MCP server '%s': %s pagination exceeded %d pages; truncating at %d items",
                       server_name, items_attr, _MCP_LIST_MAX_PAGES, len(items))
    return items


# ---- Server task -- each MCP server lives in one long-lived asyncio Task ----

class MCPServerTask(MCPServerRunMixin, MCPServerTransportMixin, MCPServerHealthMixin):
    """One MCP server connection in one long-lived asyncio Task (the transport's anyio cancel
    scopes must enter/exit in the same Task). Run state machine, transport bring-up and
    keepalive/liveness live in the three mixins."""

    __slots__ = (
        "name", "session", "tool_timeout", "_task", "_ready", "_shutdown_event", "_reconnect_event",
        "_tools", "_error", "_config", "_sampling", "_elicitation", "_registered_tool_names",
        "_auth_type", "_refresh_lock", "_rpc_lock", "_pending_refresh_tasks", "_pending_call_context",
        "_lifecycle_started_at", "_last_tool_call_at", "_idle_timeout_seconds", "_max_lifetime_seconds",
        "_recycled_reason", "initialize_result", "_ping_unsupported", "_list_cache_meta",
        "_reconnect_retries", "_session_proven", "_was_parked", "_inflight_tasks", "_reconnecting",
        "_suspect_reason", "_teardown_race", "_permanent_grace_used", "_stdio_child_pids",
        "_ever_connected")

    def __init__(self, name: str):
        self.name = name
        self.session: Optional[Any] = None
        self.tool_timeout: float = _DEFAULT_TOOL_TIMEOUT
        self._task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        # Set -> _run_http/_run_stdio exit cleanly and run() re-enters the transport.
        self._reconnect_event = asyncio.Event()
        self._tools: list = []
        self._registered_tool_names: list[str] = []
        self._config: dict = {}
        self._error: Optional[Exception] = None
        self._sampling: Optional[SamplingHandler] = None
        self._elicitation: Optional[ElicitationHandler] = None
        self._reconnect_retries: int = 0
        # Rapid-drop budget: a session is UNPROVEN until it survives a keepalive interval or a
        # successful call; only a proven session clears the budget, so a post-handshake flapper
        # still parks.
        # Rapid-drop budget (#62212): a freshly (re)established session is UNPROVEN until it demonstrates
        # real health — it survived at least one full keepalive interval (keepalive success path) or served
        # at least one successful tool call. Only a proven session clears the reconnect budget; a transport
        # that flaps right after the handshake keeps getting charged and still reaches the park instead of
        # hot-cycling respawns forever.
        self._session_proven: bool = False
        # Never cleared (unlike _ready): separates first-connect from reconnect failures.
        self._ever_connected: bool = False
        # True from park until proven healthy again; logs the revival once.
        self._was_parked: bool = False
        # In-flight RPC tasks so a deliberate teardown fails them fast; _reconnecting is True
        # during that teardown so _track_inflight_rpc turns the cancel into a retryable error.
        # In-flight RPC bookkeeping (#48069 salvage): user-visible requests registered while running so a
        # reconnect/shutdown teardown can fail them fast instead of orphaning them on a dying transport.
        self._inflight_tasks: set = set()
        self._reconnecting: bool = False
        # Latched by races (teardown-vs-keepalive, auth-lock corruption); ensure_healthy()
        # verifies before the next call.
        # See #77765, #81051, #84132.
        self._suspect_reason: Optional[str] = None
        # Teardown that failed in-flight calls => next reconnect is RACE RECOVERY, not a
        # budget charge.
        self._teardown_race: bool = False
        # One-time grace: auth/permanent failure on a PROVEN session gets one suspect+reconnect
        # cycle before parking.
        self._permanent_grace_used: bool = False
        # Children of the current stdio transport: in-flight calls fail FAST when one dies.
        # PIDs of the stdio subprocess spawned for the current transport (captured in _run_stdio). Used to
        # fail in-flight calls FAST when the child dies instead of waiting out the full tool timeout
        # (#81995).
        self._stdio_child_pids: Set[int] = set()
        self._auth_type: str = ""
        self._refresh_lock = asyncio.Lock()
        # A stdio session is one JSON-RPC stream (a concurrent list_tools can wedge a tool
        # call): serialize client-initiated RPCs per server (HTTP too, for ordering).
        self._rpc_lock = asyncio.Lock()
        self._pending_refresh_tasks: set[asyncio.Task] = set()
        # contextvars snapshot inside session.call_tool(): the SDK runs elicitation/create on a
        # task that does not inherit HERMES_SESSION_PLATFORM, so the callback replays this.
        self._pending_call_context: Optional[contextvars.Context] = None
        self._lifecycle_started_at = self._last_tool_call_at = time.monotonic()
        self._idle_timeout_seconds = self._max_lifetime_seconds = self._recycled_reason = None
        # Handshake InitializeResult: the server's REAL advertised capabilities.
        # Captures the ``InitializeResult`` returned by ``await session.initialize()`` so downstream code
        # can inspect the server's real advertised capabilities (``.capabilities.resources``,
        # ``.capabilities.prompts``) instead of assuming every ``ClientSession`` method attribute
        # corresponds to a supported server method. See #18051.
        self.initialize_result: Optional[Any] = None
        # SEP-2549 cache hints from the last tools/list (ttl_ms, cache_scope).
        self._list_cache_meta: dict = {}
        # Latched when ``ping`` returns -32601; keepalives then use list_tools. Reset per connect.
        self._ping_unsupported: bool = False

    def _is_http(self) -> bool:
        """Check if this server uses HTTP transport."""
        return "url" in self._config

    def _advertises_tools(self) -> bool:
        """Whether the server advertises the ``tools`` capability.

        Per the MCP spec, ``InitializeResult.capabilities.tools`` is non-None
        iff the server implements the ``tools/*`` request family. Prompt-only
        or resource-only servers omit it, and calling ``tools/list`` against
        them raises ``MCPError(-32601 Method not found)`` — which previously
        killed the connection during discovery and made every keepalive fail.
        (Ported from anomalyco/opencode#31271.)

        Returns True when no capability info was captured (legacy fallback:
        preserve the old always-call-list_tools behavior rather than regress
        any server that was working before this gate).
        """
        init_result = self.initialize_result
        caps = getattr(init_result, "capabilities", None) if init_result is not None else None
        if caps is None:
            return True
        return getattr(caps, "tools", None) is not None

    async def _negotiate_session(self, session, connect_timeout: float):
        """Negotiate the protocol era with the server and return its result.

        MCP 2026-07-28 replaced the ``initialize``/``initialized`` handshake
        with a stateless core: every request is self-describing and clients
        MAY probe ``server/discover`` up front (SEP-2575). The SDK exposes
        both paths on ``ClientSession`` (``initialize()`` / ``discover()``)
        and ``adopt()``s whichever result installs the outbound stamp, so
        the rest of this file is era-agnostic.

        Per-server ``protocol`` config key:

        - ``auto`` (default): try the legacy handshake FIRST, and fall back
          to ``server/discover`` when the server signals it is modern-only
          (``UnsupportedProtocolVersion`` -32022, or ``initialize`` missing
          -32601). This is the reverse of the SDK's own discover-first auto
          mode, on purpose: nearly every configured/catalog server today
          speaks the handshake era, and initialize-first means ZERO extra
          round-trips and zero behavior change for all of them, while
          stateless-only servers still connect via the fallback.
        - ``stateless``: probe ``server/discover`` first (one legacy retry
          on MCPError, so a handshake-only server still connects).
        - ``legacy``: handshake only, no fallback (escape hatch for servers
          that misbehave on unknown methods).

        Both result types expose ``.capabilities``, so downstream gates
        (``_advertises_tools``, ``_select_utility_schemas``, the config
        probe) work unchanged on either.
        """
        mode = str((self._config or {}).get("protocol", "auto")).lower().strip()
        if mode in ("stateless", "modern", "2026-07-28"):
            try:
                return await asyncio.wait_for(
                    session.discover(), timeout=connect_timeout
                )
            except asyncio.TimeoutError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.info(
                    "MCP server '%s': server/discover rejected (%s) despite "
                    "protocol=%s — falling back to the legacy handshake",
                    self.name, exc, mode,
                )
                return await asyncio.wait_for(
                    session.initialize(), timeout=connect_timeout
                )
        if mode in ("legacy", "handshake"):
            return await asyncio.wait_for(
                session.initialize(), timeout=connect_timeout
            )
        if mode != "auto":
            logger.warning(
                "MCP server '%s': unknown protocol=%r — treating as 'auto' "
                "(valid: auto, stateless, legacy)", self.name, mode,
            )
        try:
            return await asyncio.wait_for(
                session.initialize(), timeout=connect_timeout
            )
        except asyncio.TimeoutError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not _handshake_rejected_as_modern(exc):
                raise
            if not hasattr(session, "discover"):
                # Legacy SDK generation (mcp 1.x) has no server/discover
                # client — nothing to fall back to.
                raise
            logger.info(
                "MCP server '%s': legacy handshake rejected (%s) — "
                "retrying via server/discover (2026-07-28 stateless server)",
                self.name, exc,
            )
            return await asyncio.wait_for(
                session.discover(), timeout=connect_timeout
            )

    def _is_recycled_stdio(self) -> bool:
        """Return True when a stdio server was intentionally recycled."""
        return not self._is_http() and self._recycled_reason is not None

    def mark_tool_call(self) -> None:
        """Record that a user-visible MCP operation is starting."""
        self._last_tool_call_at = time.monotonic()

    def _mark_lifecycle_started(self) -> None:
        now = time.monotonic()
        self._lifecycle_started_at = now
        self._last_tool_call_at = now
        self._recycled_reason = None

    def _stdio_recycle_reason(self, now: Optional[float] = None) -> Optional[str]:
        """Return the stdio recycle reason if idle/age limits have elapsed."""
        if self._is_http() or self._rpc_lock.locked():
            return None
        now = time.monotonic() if now is None else now
        if (
            self._max_lifetime_seconds is not None
            and now - self._lifecycle_started_at >= self._max_lifetime_seconds
        ):
            return "max_lifetime_seconds"
        if (
            self._idle_timeout_seconds is not None
            and now - self._last_tool_call_at >= self._idle_timeout_seconds
        ):
            return "idle_timeout_seconds"
        return None

    def _next_stdio_recycle_deadline(self) -> Optional[float]:
        """Return the next monotonic recycle deadline for stdio, if any."""
        if self._is_http() or self._rpc_lock.locked():
            return None
        deadlines = []
        if self._max_lifetime_seconds is not None:
            deadlines.append(self._lifecycle_started_at + self._max_lifetime_seconds)
        if self._idle_timeout_seconds is not None:
            deadlines.append(self._last_tool_call_at + self._idle_timeout_seconds)
        return min(deadlines) if deadlines else None

    def _mark_stdio_recycled(self, reason: str) -> None:
        """Mark a stdio session dormant before its transport finishes closing."""
        self._recycled_reason = reason
        self.session = None

    # ----- Dynamic tool discovery (notifications/tools/list_changed) -----

    async def _refresh_tools_task(self):
        """Run a dynamic tool refresh and log failures from background tasks."""
        try:
            await self._refresh_tools()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("MCP server '%s': dynamic tool refresh failed", self.name)

    def _schedule_tools_refresh(self) -> asyncio.Task:
        """Schedule a background tool refresh and keep it strongly referenced."""
        task = asyncio.create_task(self._refresh_tools_task())
        self._pending_refresh_tasks.add(task)
        task.add_done_callback(self._pending_refresh_tasks.discard)
        return task

    def _make_logging_callback(self):
        """Build a ``logging_callback`` for ``ClientSession``.

        Routes MCP ``notifications/message`` log notifications from the
        server into Hermes' logging (agent.log via hermes_logging), tagged
        with the server name.  Without this, the SDK's default callback
        silently discards them, so server-side warnings/errors during a
        tool call were invisible.  Port of anomalyco/opencode#34529.
        """
        async def _on_log(params):
            try:
                level = _MCP_LOG_LEVEL_MAP.get(
                    str(getattr(params, "level", "info")).lower(), logging.INFO,
                )
                data = getattr(params, "data", None)
                if not isinstance(data, str):
                    try:
                        data = json.dumps(data, ensure_ascii=False, default=str)
                    except (TypeError, ValueError):
                        data = str(data)
                # Cap pathological payloads so a chatty/broken server can't
                # flood agent.log with megabyte lines.
                if len(data) > 2000:
                    data = data[:2000] + "... [truncated]"
                logger_name = getattr(params, "logger", None)
                origin = f"{self.name}/{logger_name}" if logger_name else self.name
                logger.log(level, "MCP server log [%s]: %s", origin, data)
            except Exception:
                logger.debug(
                    "Failed to handle MCP log notification from '%s'",
                    self.name, exc_info=True,
                )
        return _on_log

    def _make_message_handler(self):
        """Build a ``message_handler`` callback for ``ClientSession``.

        Dispatches on notification type.  Only ``ToolListChangedNotification``
        triggers a refresh; prompt and resource change notifications are
        logged as stubs for future work.
        """
        async def _handler(message):
            try:
                if isinstance(message, Exception):
                    logger.debug("MCP message handler (%s): exception: %s", self.name, message)
                    return
                if _MCP_NOTIFICATION_TYPES and isinstance(message, ServerNotification):
                    # mcp 2.0 turned ServerNotification from a RootModel into
                    # a plain union of the concrete notification types, so the
                    # payload IS the message instead of living under ``.root``.
                    # ``isinstance`` accepts a union, so the guard above still
                    # holds on both generations; only the unwrap changes.
                    # Without this, ``message.root`` raises AttributeError into
                    # the catch-all below and tools/list_changed refreshes stop
                    # firing silently.
                    match getattr(message, "root", message):
                        case ToolListChangedNotification():
                            logger.info(
                                "MCP server '%s': received tools/list_changed notification",
                                self.name,
                            )
                            # Some servers (notably mongodb-mcp-server) emit
                            # tools/list_changed immediately after initialize,
                            # while the client may already be executing another
                            # request. Refreshing synchronously inside the SDK
                            # notification handler can race with that request
                            # and wedge the stdio JSON-RPC stream, making all
                            # subsequent tool calls time out. Do the refresh in
                            # a separate task and let the handler return
                            # promptly.
                            self._schedule_tools_refresh()
                            # Yield one loop tick so tests and short-lived
                            # notification contexts can observe the scheduled
                            # refresh without awaiting the full server RPC.
                            await asyncio.sleep(0)
                        case PromptListChangedNotification():
                            logger.debug("MCP server '%s': prompts/list_changed (ignored)", self.name)
                        case ResourceListChangedNotification():
                            logger.debug("MCP server '%s': resources/list_changed (ignored)", self.name)
                        case _:
                            pass
            except Exception:
                logger.exception("Error in MCP message handler for '%s'", self.name)
        return _handler

    async def _refresh_tools(self):
        """Re-fetch tools from the server and update the registry.

        Called when the server sends ``notifications/tools/list_changed``.
        The lock prevents overlapping refreshes from rapid-fire notifications.
        After the initial ``await`` (list_tools), all mutations are synchronous
        — atomic from the event loop's perspective.
        """
        from tools.registry import registry

        if not self._advertises_tools():
            # A server that doesn't implement tools/* should never send
            # tools/list_changed, but guard anyway — calling tools/list
            # would raise MCPError(-32601).
            return

        async with self._refresh_lock:
            # Capture old tool names for change diff
            old_tool_names = set(self._registered_tool_names)

            # 1. Fetch current tool list from server (follow nextCursor)
            async with self._rpc_lock:
                new_mcp_tools = await _paginate_full_list(
                    self.session.list_tools, "tools", self.name
                )

            # 2. Re-register with fresh tool list. Avoid nuke-and-repave for
            # all names: live agent turns may already have tool-call IDs
            # pointing at existing handler functions. Replacing entries
            # in-place is enough for unchanged names and avoids transient
            # "tool not connected" / stale-handler races during startup
            # notifications. Tools absent from the fresh list are no longer
            # callable, so remove only those stale registry entries first.
            toolset_name = f"mcp-{self.name}"
            stale_tool_names = old_tool_names - {
                mcp_prefixed_tool_name(self.name, tool.name)
                for tool in new_mcp_tools
            }
            for tool_name in stale_tool_names:
                # Never let one server's refresh remove a colliding name that
                # is currently owned by another server.
                if registry.get_toolset_for_tool(tool_name) != toolset_name:
                    continue
                registry.deregister(tool_name)
                _forget_mcp_tool_server(tool_name)

            # 3. Re-register with the fresh list. The helper may skip names that
            # are ambiguous after normalization.
            self._tools = new_mcp_tools
            registered_names = _register_server_tools(
                self.name, self, self._config
            )

            # A previously unique raw name can become ambiguous without changing
            # its normalized registry name. In that case the pre-pass above does
            # not consider it stale, so remove any old entry that the final,
            # collision-checked registration set no longer owns.
            registered_name_set = set(registered_names)
            for tool_name in old_tool_names - registered_name_set:
                if registry.get_toolset_for_tool(tool_name) != toolset_name:
                    continue
                registry.deregister(tool_name)
                _forget_mcp_tool_server(tool_name)
            self._registered_tool_names = registered_names

            # 4. Log what changed (user-visible notification)
            new_tool_names = set(self._registered_tool_names)
            added = new_tool_names - old_tool_names
            removed = old_tool_names - new_tool_names
            changes = []
            if added:
                changes.append(f"added: {', '.join(sorted(added))}")
            if removed:
                changes.append(f"removed: {', '.join(sorted(removed))}")
            if changes:
                logger.warning(
                    "MCP server '%s': tools changed dynamically — %s. "
                    "Verify these changes are expected.",
                    self.name, "; ".join(changes),
                )
            else:
                logger.info(
                    "MCP server '%s': dynamically refreshed %d tool(s) (no changes)",
                    self.name, len(self._registered_tool_names),
                )

    async def _keepalive_probe(self) -> None:
        """Exercise the session to detect a stale/expired connection.

        Uses ``ping`` (cheap, transport-agnostic liveness) by default. ``ping``
        is an OPTIONAL MCP utility: a server that doesn't implement it answers
        JSON-RPC -32601. The first time that happens we latch
        ``_ping_unsupported`` and fall back to the pre-ping probe — capability
        permitting, ``list_tools``; otherwise ``ping`` is the only option and
        the -32601 propagates (a server advertising neither a working ping nor
        tools has no liveness primitive left). The latch resets on each fresh
        transport connection so a server that gains ping support after a
        reconnect is re-probed with the cheap path.

        Raises on a genuine connection failure so the caller triggers a
        reconnect; returns normally when the session is alive.
        """
        # Keepalives are client-initiated RPCs too. Normal tools/resources/
        # prompts are serialized through ``_rpc_lock`` because overlapping
        # requests on one MCP transport can wedge or close the underlying
        # anyio stream. Do not let an idle liveness ping race an active call.
        # If a user RPC is already in flight, that RPC is itself the stronger
        # liveness signal, so simply defer this keepalive cycle.
        if self._rpc_lock.locked():
            logger.debug(
                "MCP server '%s': skipping keepalive while an RPC is in flight",
                self.name,
            )
            return

        async with self._rpc_lock:
            session = self.session
            if session is None:
                raise RuntimeError("MCP session disappeared before keepalive probe")

            if not self._ping_unsupported:
                try:
                    await asyncio.wait_for(session.send_ping(), timeout=30.0)
                    return
                except Exception as exc:
                    # Only a "method not found" means ping is unsupported. Any
                    # other error (timeout, closed transport, session expired) is
                    # a real liveness failure — propagate so we reconnect.
                    if not _is_method_not_found_error(exc):
                        raise
                    if not self._advertises_tools():
                        # No ping, no tools → no cheaper probe to fall back to.
                        raise
                    self._ping_unsupported = True
                    logger.info(
                        "MCP server '%s': does not implement the optional 'ping' "
                        "utility (-32601); using 'list_tools' for keepalive on "
                        "this connection.",
                        self.name,
                    )

            # Fallback probe for servers without ping support.
            await asyncio.wait_for(session.list_tools(), timeout=30.0)

    def _mark_session_proven(self) -> None:
        """Record that the current session demonstrated real health.

        Called from the keepalive success path (session survived at least one
        full keepalive interval) and the tool-call success path. Only then is
        the reconnect budget cleared: a handshake that completes but drops
        moments later must keep consuming ``_reconnect_retries`` so a flapping
        transport still reaches the park instead of respawning forever
        (#62212 — 6212 spawns in 63h).
        """
        if not self._session_proven:
            self._session_proven = True
            self._reconnect_retries = 0
            if self._was_parked:
                self._was_parked = False
                logger.warning(
                    "MCP server '%s': revived — session healthy again after "
                    "parking (state: parked → connected)",
                    self.name,
                )
            # A session that just proved healthy on a fresh transport clears
            # the one-time permanent-failure grace and any race bookkeeping.
            self._permanent_grace_used = False
            self._teardown_race = False

    # -- SuspectableBackend contract (agent.deadline) -----------------------

    def mark_suspect(self, reason: str) -> None:
        """Latch a suspicion about this connection. Cheap — no I/O.

        The NEXT call verifies via :meth:`ensure_healthy` and recycles the
        transport if the probe fails, instead of the connection silently
        staying poisoned until process restart (#81051/#77765/#84132).
        """
        if self._suspect_reason is None and reason:
            logger.warning(
                "MCP server '%s': connection marked suspect (%s); next call "
                "will health-check it",
                self.name, reason,
            )
        self._suspect_reason = reason or None

    async def ensure_healthy(self, timeout: float = 5.0) -> bool:
        """Verify a suspect connection before reuse; recycle if dead.

        Returns True when healthy (suspicion cleared). On failure, requests a
        reconnect, drops the stale session reference so the caller's normal
        no-session path takes over, and returns False. Never raises.
        """
        reason = self._suspect_reason
        if not reason:
            return True
        if self.session is None:
            # Nothing to verify — the reconnect path owns recovery now.
            self._suspect_reason = None
            self._reconnect_event.set()
            return False
        try:
            await asyncio.wait_for(self._keepalive_probe(), timeout=timeout)
        except Exception as exc:
            root = _unwrap_exception_group(exc)
            logger.warning(
                "MCP server '%s': suspect connection (%s) failed health "
                "check (%s: %s) — requesting reconnect (state: suspect → "
                "degraded)",
                self.name, reason, type(root).__name__, root,
            )
            self._suspect_reason = None
            self.mark_suspect(f"health check failed after {reason}")
            self.session = None
            self._ready.clear()
            self._reconnect_event.set()
            return False
        logger.info(
            "MCP server '%s': suspect connection passed health check "
            "(%s) — clearing suspicion",
            self.name, reason,
        )
        self._suspect_reason = None
        self._mark_session_proven()
        return True

    def _fail_inflight_calls(self, reason: str) -> None:
        """Cancel every in-flight RPC attached to this connection.

        Called from the lifecycle exits (reconnect/shutdown/recycle) BEFORE
        the transport unwinds: the MCP SDK does not always fail pending
        requests when its streams close, so without this an in-flight call
        would wait out the full tool timeout on a dying transport. Cancelling
        at least one task flags the cycle as a teardown race
        (``_teardown_race``) so run() treats the following reconnect as
        recovery rather than charging the rapid-drop budget.
        """
        victims = [t for t in self._inflight_tasks if not t.done()]
        if not victims:
            return
        self._reconnecting = True
        self._teardown_race = True
        self.mark_suspect(f"{reason} tore down {len(victims)} in-flight call(s)")
        for task in victims:
            task.cancel()

    def _stdio_children_dead(self) -> bool:
        """True when every stdio child we spawned has exited.

        Best-effort: only meaningful for stdio transports with captured PIDs;
        returns False (unknown → don't fail fast) otherwise.
        """
        pids = getattr(self, "_stdio_child_pids", None)
        if not pids or self._is_http():
            return False
        try:
            import psutil
        except ImportError:
            return False  # unknown → don't fail fast
        for pid in pids:
            # pid_exists handles Windows without signal-permission noise; a
            # probe failure is unknown, not proof that every child exited.
            try:
                alive = psutil.pid_exists(pid)
            except Exception:
                return False  # unknown → don't fail fast
            if alive:
                return False  # at least one child alive → not all dead
        return True  # every tracked child has exited

    async def _watch_stdio_children(self) -> None:
        """Poll child liveness while a stdio RPC is in flight (#81995).

        Resolves when a tracked child dies; the caller then cancels the RPC
        immediately instead of letting it hang for the full tool timeout.
        """
        while True:
            if self._stdio_children_dead():
                return
            await asyncio.sleep(0.25)

    async def _wait_for_lifecycle_event(self) -> str:
        """Block until either _shutdown_event or _reconnect_event fires.

        Returns:
            "shutdown"  if the server should exit the run loop entirely.
            "reconnect" if the server should tear down the current MCP
                        session and re-enter the transport (fresh OAuth
                        tokens, new session ID, etc.). The reconnect event
                        is cleared before return so the next cycle starts
                        with a fresh signal.
            "recycle"   if a stdio idle/max-lifetime limit elapsed. The
                        current transport is torn down and restarted lazily
                        on the next tool call.

        Shutdown takes precedence if both events are set simultaneously.

        Periodically sends a lightweight keepalive (``ping``, with a
        ``list_tools`` fallback for servers that don't implement the optional
        ping utility — see :meth:`_keepalive_probe`) to prevent TCP/session
        state from going stale during idle periods (#17003). If the keepalive
        fails, triggers a reconnect.

        The cadence is ``keepalive_interval`` from server config (default
        :data:`_DEFAULT_KEEPALIVE_INTERVAL`, floored at
        :data:`_MIN_KEEPALIVE_INTERVAL`). Servers that GC idle sessions on a
        short TTL (e.g. Unreal Engine's editor MCP, ~15s) need an interval
        below that TTL, otherwise every idle tool call lands on an
        already-expired session and pays the full reconnect path.
        """
        # Refresh faster than the server's session TTL. ``ping`` (MCP base
        # protocol liveness) is used rather than ``list_tools`` so the probe
        # stays a few bytes regardless of how many tools the server exposes —
        # a ``list_tools`` keepalive against an 830-tool server would pull
        # ~1 MB every cycle. Tool-list changes still arrive out-of-band via
        # ``notifications/tools/list_changed`` → ``_refresh_tools``.
        keepalive_interval = max(
            _MIN_KEEPALIVE_INTERVAL,
            float(self._config.get("keepalive_interval", _DEFAULT_KEEPALIVE_INTERVAL)),
        )

        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
        reconnect_task = asyncio.create_task(self._reconnect_event.wait())
        try:
            while True:
                recycle_reason = self._stdio_recycle_reason()
                if recycle_reason is not None:
                    self._mark_stdio_recycled(recycle_reason)
                    return "recycle"

                timeout = keepalive_interval
                recycle_deadline = self._next_stdio_recycle_deadline()
                if recycle_deadline is not None:
                    timeout = max(0.0, min(timeout, recycle_deadline - time.monotonic()))

                done, _pending = await asyncio.wait(
                    {shutdown_task, reconnect_task},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if done:
                    break

                recycle_reason = self._stdio_recycle_reason()
                if recycle_reason is not None:
                    self._mark_stdio_recycled(recycle_reason)
                    return "recycle"

                # Timeout — no lifecycle event fired.  Probe the connection
                # to detect stale/expired sessions — but NEVER while an RPC
                # is in flight (#48069): the stdio session is a single
                # JSON-RPC stream and a concurrent ping/list_tools can wedge
                # the in-flight request. A busy server is provably alive.
                if self.session:
                    if self._rpc_lock.locked() or any(
                        not t.done() for t in self._inflight_tasks
                    ):
                        continue
                    try:
                        async def _probe_under_lock():
                            async with self._rpc_lock:
                                await self._keepalive_probe()

                        await _probe_under_lock()
                    except Exception as exc:
                        root = _unwrap_exception_group(exc)
                        logger.warning(
                            "MCP server '%s' keepalive failed, triggering "
                            "reconnect (state: connected → degraded): %s: %s",
                            self.name, type(root).__name__, root,
                        )
                        self.mark_suspect(
                            f"keepalive failed: {type(root).__name__}: {root}"
                        )
                        self._reconnect_event.set()
                        break
                    # Keepalive succeeded — the session survived a full
                    # keepalive interval, which is real proof of health.
                    # Clear the rapid-drop budget (#62212).
                    self._mark_session_proven()
        finally:
            for t in (shutdown_task, reconnect_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

        if self._shutdown_event.is_set():
            self._fail_inflight_calls("shutdown")
            return "shutdown"
        # Deliberate teardown: fail any in-flight RPC NOW so it doesn't ride
        # the dying transport to the full tool timeout (#48069/#81995).
        self._fail_inflight_calls("reconnect")
        self._reconnect_event.clear()
        return "reconnect"

    async def _wait_for_reconnect_or_shutdown(
        self, timeout: Optional[float] = None
    ) -> str:
        """Block until a reconnect or shutdown is requested while parked.

        Used by :meth:`run` after the reconnect budget is exhausted. The
        task stays alive (so ``_reconnect_event`` always has a listener) but
        does no work until something explicitly asks it to come back —
        OAuth recovery, a manual ``/mcp`` refresh — or, when ``timeout`` is
        given, until the timeout elapses (a periodic self-probe). The timed
        wake matters because parking deregisters this server's tools, so
        no tool call can ever reach the circuit-breaker's half-open probe
        or ``_signal_reconnect`` — without a self-probe a parked server
        would be unrevivable short of a full reload.

        Returns:
            ``"shutdown"`` if the server should exit the run loop entirely,
            ``"reconnect"`` if it should rebuild the transport (explicit
            request or self-probe timeout). The reconnect event is cleared
            before returning so the next park cycle starts from a fresh
            signal. Shutdown takes precedence.
        """
        shutdown_task = asyncio.ensure_future(self._shutdown_event.wait())
        reconnect_task = asyncio.ensure_future(self._reconnect_event.wait())
        try:
            await asyncio.wait(
                {shutdown_task, reconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout,
            )
        finally:
            for t in (shutdown_task, reconnect_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
        if self._shutdown_event.is_set():
            return "shutdown"
        self._reconnect_event.clear()
        return "reconnect"

    async def _run_stdio(self, config: dict):
        """Run the server using stdio transport."""
        if config.get("identity_header") is not None:
            # Headers don't exist on stdio transports — warn and ignore so a
            # copy-pasted HTTP config block doesn't silently mislead.
            logger.warning(
                "MCP server '%s': identity_header is only supported on "
                "HTTP/SSE transports — ignored for stdio servers", self.name,
            )
        if not _ensure_mcp_sdk():
            raise ImportError(
                f"MCP server '{self.name}' requires the 'mcp' Python SDK, but "
                "it is not installed. Run `hermes setup` to install MCP support, "
                "then retry."
            )

        command = config.get("command")
        args = config.get("args", [])
        user_env = config.get("env")

        if not command:
            raise ValueError(
                f"MCP server '{self.name}' has no 'command' in config"
            )

        safe_env = _build_safe_env(user_env)
        command, safe_env = _resolve_stdio_command(command, safe_env)

        # Check package against OSV malware database before spawning.
        # Run off the event loop (the urllib HTTPS call is blocking) and bound
        # it with a wall-clock timeout so a stalled SSL handshake can't freeze
        # MCP discovery / gateway startup (#29184). The check is fail-open, so
        # on timeout we log and proceed rather than blocking indefinitely.
        # NOTE: must run against the REAL command/args — the watchdog wrap
        # below rewrites argv to `python -m tools.mcp_stdio_watchdog …`,
        # which would silently turn the preflight into a no-op.
        from tools.osv_check import check_package_for_malware
        try:
            malware_error = await asyncio.wait_for(
                asyncio.to_thread(check_package_for_malware, command, args),
                timeout=_OSV_MALWARE_CHECK_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "MCP server '%s': OSV malware preflight timed out after %.0fs "
                "(network slow/unreachable) — proceeding without the check.",
                self.name, _OSV_MALWARE_CHECK_TIMEOUT_S,
            )
            malware_error = None
        if malware_error:
            raise ValueError(
                f"MCP server '{self.name}': {malware_error}"
            )

        # Wrap the real command in a parent-death watchdog supervisor so an
        # ungraceful exit of this Hermes process (kill -9, crash, force-quit)
        # can't leave the stdio MCP child (and its own descendants, e.g.
        # mcp-remote's spawned `node`) running forever. On a clean exit,
        # MCPServerTask.shutdown() / _kill_orphaned_mcp_children() still do
        # the reaping as before -- this only covers the case where that code
        # never gets to run. POSIX-only (relies on process groups); no-op
        # elsewhere, matching existing killpg-based cleanup's platform scope.
        # Applied AFTER the OSV preflight so the check inspects the real
        # package, not the watchdog wrapper.
        command, args = _wrap_command_with_watchdog(command, args)

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=safe_env if safe_env else None,
            cwd=config.get("cwd"),
            # On Windows, pipe I/O can deliver non-UTF-8 bytes at chunk
            # boundaries.  Use "replace" to substitute undecodable bytes
            # with U+FFFD instead of crashing with UnicodeDecodeError.
            encoding_error_handler="replace",
        )

        sampling_kwargs = self._sampling.session_kwargs() if self._sampling else {}
        if self._elicitation:
            sampling_kwargs.update(self._elicitation.session_kwargs())
        if _MCP_NOTIFICATION_TYPES and _MCP_MESSAGE_HANDLER_SUPPORTED:
            sampling_kwargs["message_handler"] = self._make_message_handler()
        if _MCP_LOGGING_CALLBACK_SUPPORTED:
            sampling_kwargs["logging_callback"] = self._make_logging_callback()

        # Reap any orphaned subprocesses from prior failed connection
        # attempts before spawning a new one.  Without this, each retry in
        # the run() reconnect loop spawns a fresh process pair while the
        # previous failed pair lingers — leading to rapid zombie
        # accumulation (see #57355, #57228).  The unscoped sweep also
        # opportunistically reaps orphans left by *other* servers that
        # never reconnect; per-server filtering via ``server_name`` remains
        # available for scoped call sites.  Run in a worker thread: the
        # reaper blocks up to 2s (SIGTERM → wait → SIGKILL) when orphans
        # exist, which would otherwise stall the shared MCP event loop.
        await asyncio.to_thread(_kill_orphaned_mcp_children)

        # Snapshot child PIDs before spawning so we can track the new one.
        pids_before = _snapshot_child_pids()
        new_pids: set = set()
        # Redirect subprocess stderr into a shared log file so MCP servers
        # (FastMCP banners, slack-mcp startup JSON, etc.) don't dump onto
        # the user's TTY and corrupt the TUI.  Preserves debuggability via
        # ~/.hermes/logs/mcp-stderr.log.
        _write_stderr_log_header(self.name)
        _errlog = _get_mcp_stderr_log()
        try:
            async with stdio_client(server_params, errlog=_errlog) as (
                read_stream,
                write_stream,
            ):
                # Capture the newly spawned subprocess PID for force-kill cleanup.
                # Filter out non-MCP children that race into the snapshot window:
                # slash_worker and LSP servers (jdtls/pyright/yaml-ls) are spawned
                # directly by the gateway without start_new_session, so their pgid
                # equals the TUI parent PID. If they leak into _stdio_pgids, the
                # shutdown sweep's killpg() kills the TUI parent itself.
                # See agent/lsp/client.py for the complementary start_new_session fix.
                new_pids = _filter_mcp_children(
                    _snapshot_child_pids() - pids_before
                )
                if new_pids:
                    # Capture pgid while the child is alive — once it exits we
                    # can no longer call ``os.getpgid`` on it, and the cleanup
                    # sweep needs the pgid to reach any reparented descendants
                    # (e.g. ``claude mcp serve`` spawned by a stdio wrapper).
                    new_pgids: Dict[int, int] = {}
                    for _pid in new_pids:
                        try:
                            new_pgids[_pid] = os.getpgid(_pid)
                        except (AttributeError, ProcessLookupError, OSError):
                            # AttributeError: Windows (os.getpgid is POSIX-only)
                            # ProcessLookupError: child raced and already exited
                            pass
                    with _lock:
                        for _pid in new_pids:
                            _stdio_pids[_pid] = self.name
                        _stdio_pgids.update(new_pgids)
                    # Positive identity for the machine spawn ledger (#61514):
                    # record each helper child as (pid, create_time,
                    # 'mcp-helper', spawner=this process) so startup sweeps
                    # can reap orphans left after an unclean parent exit.
                    # Best-effort — never let ledger I/O break MCP startup.
                    for _pid in new_pids:
                        try:
                            from hermes_cli.process_identity import register_child

                            register_child(_pid, "mcp-helper")
                        except Exception:
                            logger.debug(
                                "spawn-ledger register_child failed for MCP "
                                "helper pid %s",
                                _pid,
                                exc_info=True,
                            )
                # Track the spawned children on the connection object for
                # fast-fail of in-flight calls when the subprocess dies
                # (#81995).
                self._stdio_child_pids = set(new_pids)
                async with ClientSession(
                    read_stream, write_stream, **sampling_kwargs
                ) as session:
                    # Bound the MCP handshake. A stdio server that never
                    # completes ``initialize`` (e.g. emits a non-JSON-RPC frame
                    # and then blocks on stdin) otherwise hangs this coroutine
                    # forever on the background loop: ``connect_timeout`` only
                    # bounds the caller's ``.result()`` wait, not the coroutine
                    # itself. Because the connect never unwinds, the cleanup
                    # ``finally`` below never runs, so the spawned child and its
                    # stdio pipes/pidfd leak on every discovery retry — unbounded
                    # until the gateway hits EMFILE. Timing out here converts the
                    # hang into a normal failure, letting the ``finally`` reap the
                    # child. See #59349.
                    connect_timeout = float(
                        config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
                    )
                    self.initialize_result = await self._negotiate_session(
                        session, connect_timeout
                    )
                    self.session = session
                    self._mark_lifecycle_started()
                    await self._discover_tools()
                    self._ready.set()
                    self._ever_connected = True
                    # Session is live again: clear any breaker state from a
                    # prior outage so the first call after recovery isn't
                    # gated on a stale consecutive-failure count (#16788).
                    _reset_server_error(self.name)
                    # A completed handshake alone is NOT proof of health: a
                    # flapping transport can handshake fine and drop moments
                    # later, forever (#62212). The session must prove itself
                    # (keepalive success or a successful tool call) before the
                    # reconnect budget is cleared — see _mark_session_proven.
                    self._session_proven = False
                    # stdio transport does not use OAuth, but we still honor
                    # _reconnect_event (e.g. future manual /mcp refresh) for
                    # consistency with _run_http.
                    return await self._wait_for_lifecycle_event()
        finally:
            # Runs on clean exit, exceptions, AND asyncio cancellation.
            # If any of the spawned PIDs are still alive, the SDK's
            # teardown failed (common when the task is cancelled mid-way
            # on Linux, where setsid() children escape the parent cgroup).
            # Mark them as orphans so the next cleanup sweep can reap them.
            if new_pids:
                from gateway.status import _pid_exists
                _killpg = getattr(os, "killpg", None)
                with _lock:
                    for _pid in new_pids:
                        _stdio_pids.pop(_pid, None)
                    for pid in new_pids:
                        # ``os.kill(pid, 0)`` is NOT a no-op on Windows
                        # (bpo-14484). Use the cross-platform check.
                        pid_alive = _pid_exists(pid)
                        pgroup_alive = False
                        pgid = _stdio_pgids.get(pid)
                        if not pid_alive and pgid is not None and _killpg is not None:
                            # Direct child exited but descendants may still be
                            # in its pgroup (e.g. ``claude mcp serve`` spawned
                            # by an MCP wrapper that exited first).  Probe with
                            # signal 0 — succeeds iff any pgroup member is alive.
                            try:
                                _killpg(pgid, 0)
                                pgroup_alive = True
                            except (ProcessLookupError, PermissionError, OSError):
                                pgroup_alive = False
                        if pid_alive or pgroup_alive:
                            _orphan_stdio_pids.add(pid)
                            _orphan_stdio_pid_servers[pid] = self.name
                        else:
                            # Nothing left to reap — drop the pgid entry so
                            # PID-reuse can't surface stale pgroup state later.
                            _stdio_pgids.pop(pid, None)

    # Content types a real MCP Streamable-HTTP endpoint may return on the
    # initial POST/GET. Anything else on a 2xx response means the URL is not
    # an MCP endpoint.
    _MCP_CONTENT_TYPES = ("application/json", "text/event-stream")


# ---- Module-level state (every mutation under ``_lock``) ----

_servers: Dict[str, MCPServerTask] = {}
# Profile registry scope per live connection (None outside multiplex) so a multiplexed
# /reload-mcp tears down only its own profile's servers.
_server_scope_keys: Dict[str, Optional[str]] = {}
# Registry scopes that have adopted a live server connection. The owning scope above remains
# authoritative for connection teardown; this set preserves visibility for shared connections.
_server_tool_scopes: Dict[str, set] = {}
_server_connecting: set[str] = set()
_server_connect_errors: Dict[str, str] = {}
# Lazy startup: servers registered from the schema cache without connecting; popped on
# first real connection.
# Keyed by server name; entries are popped once a real connection is established on first use. See #56832.
_lazy_server_configs: Dict[str, dict] = {}
_lazy_server_fingerprints: Dict[str, str] = {}
_lazy_server_tool_names: Dict[str, List[str]] = {}
# Task-local claim around ``_connect_server``: discovery retains a recoverable parked task
# while standalone probes never publish failed servers into module-global ownership.
_connect_server_claim: contextvars.ContextVar[Optional[Callable[[MCPServerTask], None]]] = (
    contextvars.ContextVar("mcp_connect_server_claim", default=None))

# Per-server connect cooldown: a server that fails to spawn never reaches ``_servers``, so
# without it every ``discover_mcp_tools()`` (one per worker session) would respawn it — a
# restart storm whose unreaped children destabilise healthy servers. Exponential-backoff
# deadline honoured by ``register_mcp_servers``; cleared on success.
# Connection-retry cooldown (per-server isolation against restart storms). A single stdio MCP server that
# fails to spawn (bad PATH, ``exec: not found``, crash-on-start) is never recorded in ``_servers`` --
# ``start()`` raises and ``_discover_and_register_server`` aborts before the ``_servers[name] = server``
# line. Without a cooldown, EVERY subsequent ``discover_mcp_tools()`` (one per agent worker session, i.e.
# every few seconds) sees the server as "not connected" and re-spawns it from scratch. That is the restart
# storm in #50394: the failing server is re-attempted on the shared MCP event loop on every worker session,
# the subprocesses pile up unreaped, and the churn destabilises the healthy co-located servers (their tools
# intermittently surface as "Unknown tool"). Fix: after a failed connection attempt, stamp a monotonic
# ``retry_after`` deadline with exponential backoff. ``register_mcp_servers`` skips a server whose cooldown
# has not elapsed, so a chronically failing server is retried on a backoff schedule instead of on every
# worker session -- isolating it from the rest of the bridge. A successful connection clears the state.
_server_connect_retry_after: Dict[str, float] = {}   # name -> monotonic deadline
_server_connect_failures: Dict[str, int] = {}        # name -> consecutive failures
_CONNECT_RETRY_BASE_BACKOFF_SEC, _CONNECT_RETRY_MAX_BACKOFF_SEC = 30.0, 600.0

# Per-server circuit breaker: closed -> open (calls short-circuit until the cooldown) ->
# half-open (next call probes). Mutate only via _bump_server_error / _reset_server_error.
# After _CIRCUIT_BREAKER_THRESHOLD consecutive failures, the handler returns a "server unreachable" message
# that tells the model to stop retrying, preventing the 90-iteration burn loop described in #10447. State
# machine: closed    — error count below threshold; all calls go through. open      — threshold reached;
# calls short-circuit until the cooldown elapses. half-open — cooldown elapsed; the next call is a probe
# that actually hits the session. Probe success → closed. Probe failure → reopens (cooldown re-armed).
# ``_server_breaker_opened_at`` records the monotonic timestamp when the breaker most recently transitioned
# into the open state. Use the ``_bump_server_error`` / ``_reset_server_error`` helpers to mutate this state
# — they keep the count and timestamp in sync.
_server_error_counts: Dict[str, int] = {}
_server_breaker_opened_at: Dict[str, float] = {}
_CIRCUIT_BREAKER_THRESHOLD, _CIRCUIT_BREAKER_COOLDOWN_SEC = 3, 60.0

# Trust-tier gating (``trust: full | untrusted``): on an untrusted server every write-capable
# call (discovery-time ``readOnlyHint`` not exactly True; malformed fails closed) needs approval
# before the RPC fires. A lying readOnlyHint can only skip approval for calls the operator was
# already warned about, never widen access. Missing trust = full; unrecognized = untrusted (a
# typo must never disable the gate). Classified at CALL time from DISCOVERY data: no schema
# mutation, prompt cache intact.
_server_trust_levels: Dict[str, str] = {}
_tool_read_only_hints: Dict[str, Dict[str, bool]] = {}

_TRUST_FULL, _TRUST_UNTRUSTED = "full", "untrusted"


def _bump_server_error(server_name: str) -> None:
    """Count a failure; at the threshold (re)stamp the breaker-open time."""
    n = _server_error_counts.get(server_name, 0) + 1
    _server_error_counts[server_name] = n
    if n >= _CIRCUIT_BREAKER_THRESHOLD:
        _server_breaker_opened_at[server_name] = time.monotonic()


def _reset_server_error(server_name: str) -> None:
    """Close the breaker on any unambiguous success signal."""
    _server_error_counts[server_name] = 0
    _server_breaker_opened_at.pop(server_name, None)


# Raw server names opted into parallel tool calls (``foo-bar``/``foo_bar`` sanitize alike but
# must not share policy).
_parallel_safe_servers: set = set()
# registry tool name -> raw server name (the generated name is lossy; never re-parse it).
_mcp_tool_server_names: Dict[str, str] = {}

# Dedicated event loop in a background daemon thread; _lock guards the loop handles, _servers,
# the status maps and the PID ledgers.
_mcp_loop: Optional[asyncio.AbstractEventLoop] = None
_mcp_thread: Optional[threading.Thread] = None
_lock = threading.Lock()


# ---- Shared parent-death supervisor (state lives HERE: tests rebind ``_death_supervisor``) ----
# If this process dies without running its cleanup path (kill -9, OOM, crash, force-quit), stdio
# MCP children reparent to init and run forever; macOS has no PR_SET_PDEATHSIG, so something has
# to outlive us and reap them. ONE supervisor process serves all stdio servers and is told which
# process groups to reap over a pipe; it detects our death as EOF on that pipe (exact, instant)
# rather than polling getppid(). Replaced the per-server watchdog wrapper (~10 MB resident per
# server, plus a signal-forwarding layer because wrapping put the server in a different session
# from the pgid tracked for killpg). See tools/mcp_death_supervisor.py. POSIX-only, matching the
# killpg-based orphan cleanup below.
_death_supervisor = None  # Optional[subprocess.Popen]
_death_supervisor_lock = threading.Lock()
# Groups the supervisor is reaping on our behalf; replayed verbatim on respawn so a respawn never
# silently drops coverage for servers that are still running.
_supervised_pgids: set = set()


def _spawn_death_supervisor():
    """Start the shared supervisor, or None if it cannot be started."""
    import subprocess
    supervisor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_death_supervisor.py")
    try:
        # start_new_session=True is load-bearing: shutdown paths killpg this process's own group,
        # which would kill the supervisor before it could reap anything.
        return subprocess.Popen(
            [sys.executable, supervisor, "--parent-pgid", str(os.getpgid(0))],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=_get_mcp_stderr_log(),
            start_new_session=True, close_fds=True, text=True)
    except Exception:
        # Never let supervisor bookkeeping block a real MCP connection: graceful shutdown paths
        # still reap normally; only the ungraceful-exit safety net is lost.
        logger.debug("Could not start the MCP parent-death supervisor", exc_info=True)
        return None


def _prune_dead_supervised_pgids() -> set:
    """Forget supervised groups with no members left; return what went. Caller holds
    ``_death_supervisor_lock``. Signal 0 is a pure existence probe (cannot terminate anything).
    It narrows, but cannot close, the window where a dead group's pgid is recycled before we
    notice (residual-risk note in ``tools/mcp_death_supervisor.py``)."""
    killpg = getattr(os, "killpg", None)
    if killpg is None:  # windows-footgun: ok - POSIX-only, guarded
        return set()
    stale = set()
    for pgid in list(_supervised_pgids):
        try:
            killpg(pgid, 0)
        except ProcessLookupError:
            stale.add(pgid)
        except (PermissionError, OSError):
            # Exists but not ours to signal, or the probe failed: keep it — dropping coverage on
            # an ambiguous answer is the more expensive mistake.
            pass
    _supervised_pgids.difference_update(stale)
    return stale


def _update_death_supervisor(verb: str, pgids) -> None:
    """Register or unregister process groups (``verb`` is ``"register"``/``"unregister"``) with
    the shared supervisor. Failures are swallowed: losing the safety net must never fail a live
    MCP session."""
    if os.name != "posix":
        return
    wanted = {int(pgid) for pgid in pgids}
    if not wanted:
        return

    global _death_supervisor
    with _death_supervisor_lock:
        if verb == "register":
            _supervised_pgids.update(wanted)
        else:
            _supervised_pgids.difference_update(wanted)

        # A registration outlives the server only while some member survives (e.g. an orphaned
        # grandchild teardown failed to kill, deliberately kept registered). Once that group is
        # empty its pgid can be recycled by a stranger, so prune here too — the orphan sweep
        # unregisters what it reaps but is not guaranteed to run in a given process.
        stale = _prune_dead_supervised_pgids()

        proc = _death_supervisor
        if proc is None or proc.poll() is not None:
            if not _supervised_pgids:
                # Nothing left to cover: nothing to tell and nothing to respawn for. Keyed on
                # the SET, not the verb: after a broken-pipe write dropped the supervisor with
                # groups still registered, an unregister must still rebuild coverage for the
                # survivors.
                return
            # See #93517.
            proc = _spawn_death_supervisor()
            _death_supervisor = proc
            if proc is None:
                return
            # A fresh supervisor knows nothing: replay live coverage (already reflects this
            # call's mutation and the prune, so pruned groups never reach the replacement).
            payload = "".join(f"register {pgid}\n" for pgid in _supervised_pgids)
        else:
            payload = "".join(f"{verb} {pgid}\n" for pgid in wanted)
            payload += "".join(f"unregister {pgid}\n" for pgid in stale)

        try:
            proc.stdin.write(payload)
            proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            # It exited between poll() and write(). Drop it so the next call respawns and replays
            # from ``_supervised_pgids`` (the set, not the pipe, is the record of what needs reaping).
            _death_supervisor = None
            return

        if not _supervised_pgids:
            # Nothing left to reap: release the supervisor rather than keep a ~15 MB process and a
            # pipe resident for the life of a gateway. Closing our write end is the same EOF parent
            # death sends; with an empty set it exits. The next register respawns and replays.
            try:
                proc.stdin.close()
            except (BrokenPipeError, ValueError, OSError):
                pass
            # Reap it, or the exited supervisor stays a zombie until the next Popen in this process.
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - timeout or already gone; either way we drop it
                pass
            _death_supervisor = None


def _mcp_registry_scope() -> Optional[str]:
    """Registry scope for MCP registrations: a profile overlay under a multiplexer, else None."""
    from agent.secret_scope import is_multiplex_active
    if not is_multiplex_active():
        return None
    from tools.registry import registry
    return registry.current_scope_key()


def _server_registry_scope(name: str) -> Optional[str]:
    """Scope owning *name*'s tools: the one captured at adoption (teardown runs on the MCP
    loop without the discovering profile's context), else the current one."""
    if name in _server_scope_keys:
        return _server_scope_keys[name]
    return _mcp_registry_scope()


def _server_visible_in_scope(name: str, scope: Optional[str]) -> bool:
    """Whether a live server is visible from ``scope`` without changing its teardown owner."""
    if scope is None:
        return True
    return (_server_scope_keys.get(name) == scope
            or scope in _server_tool_scopes.get(name, ()))


# Cross-process discovery guard: advisory file lock so gateway + CLI + TUI don't all discover.
# See issue #62771.
_LOCK_UNAVAILABLE: Any = object()  # sentinel: locking broken/unavailable
_MCP_DISCOVERY_LOCK_PATH: Optional[str] = None  # resolved lazily

# Retry constants for the bounded wait when another process holds the lock.
_MCP_DISCOVERY_LOCK_MAX_RETRIES: int = 240
_MCP_DISCOVERY_LOCK_RETRY_DELAY_S: float = 0.5


class _LockCookie:
    """Holds a cross-process file lock; release() drops it.

    On Windows the underlying file handle MUST stay alive while the lock is
    held (portalocker keeps the kernel lock on the fd).  On POSIX the fcntl
    lockdown is similarly tied to the file-descriptor lifetime.  We keep the
    file object in ``_fh`` and close it on release.
    """

    def __init__(self, fh: Any) -> None:
        self._fh = fh

    def release(self) -> None:
        if self._fh is not None:
            try:
                fd = self._fh.fileno()
                if os.name == "posix":
                    import fcntl
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except Exception:
                        pass
                else:
                    import portalocker
                    try:
                        portalocker.unlock(self._fh)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None


def _acquire_lock_on_fh(fh: Any) -> bool:
    """Acquire a non-blocking exclusive lock on an open file handle.

    Uses ``fcntl.flock`` on POSIX and ``portalocker.lock`` on Windows.

    Returns ``True`` if the lock was acquired, ``False`` if another process
    holds it (non-blocking refusal).  Raises ``RuntimeError`` on unexpected
    errors so the caller can treat lock acquisition as unavailable.
    """
    fd = fh.fileno()
    if os.name == "posix":
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            raise
    else:
        import portalocker
        try:
            portalocker.lock(fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
            return True
        except portalocker.LockException:
            return False


def _try_acquire_mcp_discovery_lock() -> Any:
    """Try to acquire an exclusive cross-process lock for MCP discovery.

    Returns
    -------
    _LockCookie
        Lock acquired successfully.
    None
        Another process holds the lock (non-blocking refusal).
    _LOCK_UNAVAILABLE
        Locking mechanism is broken or unavailable -- caller should run
        discovery unguarded.
    """
    global _MCP_DISCOVERY_LOCK_PATH
    try:
        from hermes_constants import get_hermes_home
        if _MCP_DISCOVERY_LOCK_PATH is None:
            _MCP_DISCOVERY_LOCK_PATH = str(
                get_hermes_home() / ".mcp-discovery.lock"
            )
        lock_path = _MCP_DISCOVERY_LOCK_PATH
    except Exception:
        return _LOCK_UNAVAILABLE

    try:
        fh = open(lock_path, "w", encoding="utf-8")
    except Exception:
        return _LOCK_UNAVAILABLE

    try:
        acquired = _acquire_lock_on_fh(fh)
    except Exception:
        fh.close()
        return _LOCK_UNAVAILABLE

    if acquired:
        return _LockCookie(fh)
    else:
        fh.close()
        return None


# PIDs of stdio MCP server subprocesses.  Tracked so we can force-kill
# them on shutdown if the graceful cleanup (SDK context-manager teardown)
# fails or times out.  PIDs are added after connection and removed on
# normal server shutdown.
_stdio_pids: Dict[int, str] = {}  # pid -> server_name

# PIDs that survived their session context exit (SDK teardown failed to
# terminate them).  These are detected in _run_stdio's finally block and
# can be cleaned up asynchronously by _kill_orphaned_mcp_children().
# Separate from _stdio_pids so cleanup sweeps never race with active
# sessions (e.g. concurrent cron jobs or live user chats).
_orphan_stdio_pids: set = set()
_orphan_stdio_pid_servers: Dict[int, str] = {}

# Process-group IDs of stdio MCP subprocesses, captured at spawn time.
# The MCP SDK spawns stdio children with ``start_new_session=True`` so each
# direct child becomes its own session/pgroup leader (PGID == its own PID).
# Grandchildren spawned by that child (e.g. a wrapper MCP server that itself
# launches helper subprocesses like ``claude mcp serve``) inherit that PGID
# unless they call ``setsid`` themselves.  When the direct child exits, those
# grandchildren reparent to init/systemd-user but keep the original PGID, so
# ``killpg(pgid, sig)`` still reaches them.  Tracked separately from
# ``_stdio_pids`` so we retain the PGID even after the direct child has
# exited and been removed from the active map.  Empty on Windows
# (``os.getpgid`` is POSIX-only).
_stdio_pgids: Dict[int, int] = {}  # pid -> pgid


def _snapshot_child_pids() -> set:
    """Return a set of current child process PIDs.

    Uses /proc on Linux, falls back to psutil, then empty set.
    Used by _run_stdio to identify the subprocess spawned by stdio_client.
    """
    my_pid = os.getpid()

    # Linux: read from /proc
    try:
        children_path = f"/proc/{my_pid}/task/{my_pid}/children"
        with open(children_path, encoding="utf-8") as f:
            return {int(p) for p in f.read().split() if p.strip()}
    except (FileNotFoundError, OSError, ValueError):
        pass

    # Fallback: psutil
    try:
        import psutil
        return {c.pid for c in psutil.Process(my_pid).children()}
    except Exception:
        pass

    return set()


# Non-MCP gateway children that can race into the _snapshot_child_pids() delta
# during stdio MCP server spawn. LSP servers and slash_worker now use
# start_new_session=True too; this remains defense-in-depth for any future
# non-MCP child spawn that briefly appears in the MCP snapshot delta. Match
# argv markers instead of argv[0] because Python/Java children begin with the
# interpreter or binary path.
_NON_MCP_CHILD_CMDLINE_MARKERS: tuple[str, ...] = (
    "tui_gateway.slash_worker",
    "tui_gateway.entry",
    "-dorg.eclipse.equinox.launcher",  # jdtls (legacy arg style)
    "eclipse.jdt.ls",
    "org.eclipse.equinox.launcher_",
)


def _filter_mcp_children(pids: set) -> set:
    """Remove non-MCP children from a PID snapshot delta.

    _snapshot_child_pids() returns *all* direct children of the gateway. When
    a stdio MCP server spawns concurrently with a slash_worker or LSP server
    spawn, the delta ``_snapshot_child_pids() - pids_before`` can include
    PIDs that are NOT the MCP server. Tracking those PIDs in _stdio_pgids is
    catastrophic if a future child lacks start_new_session: its pgid can be the
    TUI parent's PID, so the shutdown sweep's killpg() kills the TUI itself.
    """
    if not pids:
        return pids
    try:
        import psutil
    except ImportError:
        # psutil unavailable — keep all PIDs (preserves prior behavior).
        return pids
    filtered: set = set()
    for pid in pids:
        try:
            argv = psutil.Process(pid).cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            # Process raced away or is a zombie — skip it; it cannot be the
            # MCP server we just spawned and is not safe to track.
            continue
        if any(
            marker in arg
            for arg in argv[1:]
            for marker in _NON_MCP_CHILD_CMDLINE_MARKERS
        ):
            continue
        filtered.add(pid)
    return filtered


def _mcp_loop_exception_handler(loop, context):
    """Suppress benign 'Event loop is closed' noise during shutdown.

    When the MCP event loop is stopped and closed, httpx/httpcore async
    transports may fire __del__ finalizers that call call_soon() on the
    dead loop.  asyncio catches that RuntimeError and routes it here.
    We silence it because the connection is being torn down anyway; all
    other exceptions are forwarded to the default handler.
    """
    exc = context.get("exception")
    if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
        return  # benign shutdown race — suppress
    loop.default_exception_handler(context)


def _ensure_mcp_loop():
    """Start the background event loop thread if not already running."""
    global _mcp_loop, _mcp_thread
    with _lock:
        if _mcp_loop is not None and _mcp_loop.is_running():
            return
        _mcp_loop = asyncio.new_event_loop()
        _mcp_loop.set_exception_handler(_mcp_loop_exception_handler)
        _mcp_thread = threading.Thread(
            target=_mcp_loop.run_forever,
            name="mcp-event-loop",
            daemon=True,
        )
        _mcp_thread.start()


def _wrap_with_home_override(coro: "Coroutine") -> "Coroutine":
    """Carry the caller's context-local HERMES_HOME override into ``coro``.

    Returns ``coro`` unchanged when no override is active. Otherwise wraps
    it so the override is set inside the coroutine's own (task-local)
    context on the MCP loop and reset when it completes — concurrent calls
    carrying different scopes don't interfere.
    """
    try:
        from hermes_constants import (
            get_hermes_home_override,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        home_override = get_hermes_home_override()
    except Exception:
        return coro
    if not home_override:
        return coro

    async def _scoped():
        token = set_hermes_home_override(home_override)
        try:
            return await coro
        finally:
            reset_hermes_home_override(token)

    return _scoped()


def _wrap_with_dashboard_oauth_flow(coro):
    """Propagate a dashboard OAuth flow onto the dedicated MCP loop task."""
    try:
        from tools.mcp_dashboard_oauth import (
            dashboard_oauth_flow,
            get_dashboard_oauth_flow,
        )

        flow = get_dashboard_oauth_flow()
    except Exception:
        return coro
    if flow is None:
        return coro

    async def _scoped():
        with dashboard_oauth_flow(flow):
            return await coro

    return _scoped()


def _run_on_mcp_loop(coro_or_factory, timeout: float = 30):
    """Schedule a coroutine on the MCP event loop and block until done.

    Accepts either a coroutine object or a zero-arg callable that returns one.
    Callers can pass a factory to avoid constructing coroutine objects when
    the MCP loop is unavailable (which would otherwise leak the coroutine
    frame and emit ``"coroutine was never awaited"`` warnings).

    Poll in short intervals so the calling agent thread can honor user
    interrupts while the MCP work is still running on the background loop.
    """
    from tools.interrupt import is_interrupted
    from agent.async_utils import safe_schedule_threadsafe

    with _lock:
        loop = _mcp_loop
    if loop is None or not loop.is_running():
        if asyncio.iscoroutine(coro_or_factory):
            coro_or_factory.close()
        raise RuntimeError("MCP event loop is not running")

    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory

    # Propagate the context-local HERMES_HOME override onto the MCP loop.
    # Tasks scheduled via run_coroutine_threadsafe are created INSIDE the
    # loop thread, so they copy the loop thread's context — not the
    # scheduling thread's. A per-request profile scope (the dashboard's
    # ?profile= endpoints, e.g. the MCP "Test server" probe) would silently
    # vanish here: OAuth token stores and any other get_hermes_home()
    # resolution inside the coroutine would read the process home instead
    # of the selected profile's. Re-establish the override inside the
    # task's own context (task-local — concurrent calls carrying different
    # scopes don't interfere). No-op when no override is active.
    coro = _wrap_with_home_override(coro)
    coro = _wrap_with_dashboard_oauth_flow(coro)

    future = safe_schedule_threadsafe(
        coro, loop,
        logger=logger,
        log_message="MCP scheduling failed",
    )
    if future is None:
        raise RuntimeError("MCP event loop unavailable (failed to schedule)")
    start_time = time.monotonic()
    deadline = None if timeout is None else start_time + timeout

    while True:
        if is_interrupted():
            future.cancel()
            raise InterruptedError("User sent a new message")

        wait_timeout = 0.1
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                elapsed = time.monotonic() - start_time
                raise TimeoutError(
                    f"MCP call timed out after {elapsed:.1f}s "
                    f"(configured timeout: {float(timeout):.1f}s)"
                )
            wait_timeout = min(wait_timeout, remaining)

        try:
            return future.result(timeout=wait_timeout)
        except concurrent.futures.TimeoutError:
            # On supported Python versions, concurrent.futures.TimeoutError
            # aliases the built-in TimeoutError, so result(timeout=...) also
            # raises it for a coroutine's own timeout.
            # Resolve a done future without a timeout to propagate its stored
            # outcome, including completion racing with this polling timeout.
            if future.done():
                return future.result()
            continue


def _interrupted_call_result() -> str:
    """Standardized JSON error for a user-interrupted MCP tool call."""
    return tool_error("MCP call interrupted: user sent a new message")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _interpolate_env_vars(value):
    """Recursively resolve ``${VAR}`` placeholders.

    Both ``${VAR}`` and Cursor-style ``${env:VAR}`` are accepted — the
    ``env:`` prefix is stripped so a doc copied from a Cursor / Claude MCP
    config resolves the same secret. Cursor's context variables are also
    supported (case-sensitive): ``${userHome}``, ``${workspaceFolder}``,
    ``${workspaceFolderBasename}``, ``${pathSeparator}`` and ``${/}`` — see
    :func:`_context_var_value` / :func:`_workspace_folder` for resolution.
    Env refs resolve from the active profile's secret scope when multiplexing
    is on (so an MCP server config's ``${API_KEY}`` picks up the routed
    profile's value, not the process-global ``os.environ`` which may hold
    another profile's), falling back to ``os.environ`` otherwise. Unset vars
    keep the literal placeholder, as before.
    """
    from agent.secret_scope import get_secret as _get_secret

    if isinstance(value, str):
        def _replace(m):
            ctx = _context_var_value(m.group(1).strip())
            if ctx is not None:
                return ctx
            name = _env_ref_name(m.group(1))
            return _get_secret(name, m.group(0)) or m.group(0)
        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


# (server_name, dotted key path) pairs already warned about — see
# _warn_hidden_whitespace(); config loads happen on every discovery pass.
_whitespace_warned: Set[Tuple[str, str]] = set()


def _warn_hidden_whitespace(server_name: str, config: dict) -> List[str]:
    """Warn about MCP config string values with hidden leading/trailing whitespace.

    A token pasted with a trailing newline or a URL copied with a leading
    space produces opaque auth/connect failures (the server rejects the
    credential, TLS/DNS fails on ``"example.com "``), and the whitespace is
    invisible when eyeballing config.yaml. Inspired by Claude Code v2.1.219,
    which added the same startup warning for its MCP config values.

    Advisory only — values are never mutated (whitespace could theoretically
    be intentional in an arg). Returns the list of dotted key paths flagged,
    for testability. Values themselves are never logged (they are often
    secrets); only the key path is named. Each (server, key path) is warned
    about once per process — ``_load_mcp_config()`` runs on every discovery/
    status call and repeating the warning would be noise.
    """
    flagged: List[str] = []

    def _walk(value: Any, path: str) -> None:
        if isinstance(value, str):
            if value != value.strip():
                flagged.append(path)
        elif isinstance(value, dict):
            for k, v in value.items():
                _walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _walk(v, f"{path}[{i}]")

    _walk(config, "")
    for key_path in flagged:
        dedupe_key = (server_name, key_path)
        if dedupe_key in _whitespace_warned:
            continue
        _whitespace_warned.add(dedupe_key)
        logger.warning(
            "MCP server '%s': config value '%s' has hidden leading or "
            "trailing whitespace — this often causes authentication or "
            "connection failures. Check for stray spaces/newlines in "
            "config.yaml (or the referenced env var).",
            server_name,
            key_path,
        )
    return flagged


def _filter_suspicious_mcp_servers(servers: Dict[str, dict]) -> Dict[str, dict]:
    """Drop exfiltration-shaped MCP configs before any stdio spawn path."""
    try:
        from hermes_cli.mcp_security import validate_mcp_server_entry as _validate_mcp_server_entry
    except Exception:
        _validate_mcp_server_entry: Callable[[str, dict[str, Any]], list[str]] | None = None

    if _validate_mcp_server_entry is None:
        return servers

    safe_servers = {}
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            safe_servers[name] = cfg
            continue
        issues = _validate_mcp_server_entry(name, cfg)
        if issues:
            logger.warning(
                "Skipping suspicious MCP server '%s': %s",
                name,
                "; ".join(issues),
            )
            continue
        safe_servers[name] = cfg
    return safe_servers


def _load_mcp_config() -> Dict[str, dict]:
    """Read ``mcp_servers`` from the Hermes config file.

    Returns a dict of ``{server_name: server_config}`` or empty dict.
    Server config can contain either ``command``/``args``/``env`` for stdio
    transport or ``url``/``headers`` for HTTP transport, plus optional
    ``timeout``, ``connect_timeout``, and ``auth`` overrides.

    ``${ENV_VAR}`` placeholders in string values are resolved from
    ``os.environ`` (which includes ``~/.hermes/.env`` loaded at startup).
    """
    try:
        from hermes_cli.config import load_config
        from utils import env_var_enabled as _env_enabled

        if _env_enabled("HERMES_SAFE_MODE"):
            return {}
        config = load_config()
        servers = config.get("mcp_servers")
        if not isinstance(servers, dict):
            servers = {}
        # Ensure .env vars are available for interpolation
        try:
            from hermes_cli.env_loader import load_hermes_dotenv
            load_hermes_dotenv()
        except Exception:
            pass
        safe_servers: Dict[str, dict] = {}
        for name, cfg in _filter_suspicious_mcp_servers(servers).items():
            interpolated = _interpolate_env_vars(cfg)
            if isinstance(interpolated, dict):
                _warn_hidden_whitespace(name, interpolated)
                safe_servers[name] = interpolated
        try:
            from hermes_cli.plugins import discover_plugins, get_plugin_manager

            discover_plugins()
            portable = get_plugin_manager().get_portable_mcp_servers()
            for name, cfg in _filter_suspicious_mcp_servers(portable).items():
                if name in safe_servers:
                    logger.warning(
                        "Portable MCP server '%s' conflicts with native config; skipping",
                        name,
                    )
                    continue
                safe_servers[name] = dict(cfg)
        except Exception:
            logger.debug("Failed to load portable MCP servers", exc_info=True)
        return safe_servers
    except Exception as exc:
        logger.debug("Failed to load MCP config: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Server connection helper
# ---------------------------------------------------------------------------

async def _connect_server(name: str, config: dict) -> MCPServerTask:
    """Create an MCPServerTask, start it, and return when ready.

    The server Task keeps the connection alive in the background.
    Call ``server.shutdown()`` (on the same event loop) to tear it down.

    Raises:
        ValueError: if required config keys are missing.
        ImportError: if HTTP transport is needed but not available.
        Exception: on connection or initialization failure.
    """
    server = MCPServerTask(name)
    claim = _connect_server_claim.get()
    claim_token = None
    if claim is not None:
        claim(server)
        # ``start()`` creates the long-lived run task by copying this context.
        # The ownership callback is only for this connection attempt; do not
        # retain its discovery closure for the server's lifetime.
        claim_token = _connect_server_claim.set(None)
    try:
        await server.start(config)
    except asyncio.CancelledError:
        # start() already cancels/reaps server._task on external cancellation
        # (see the comment there) -- awaiting a redundant shutdown() inside a
        # cancelled context would only risk swallowing the cancellation.
        raise
    except BaseException:
        # Discovery owns claimed tasks and decides whether a failed start is a
        # live recoverable park or a terminal failure. Standalone probes have
        # no revival owner, so they must reap their failed task locally.
        if claim is None:
            try:
                await server.shutdown()
            except Exception as shutdown_exc:  # noqa: BLE001 -- best-effort reap, don't mask the real error
                logger.debug(
                    "MCP server '%s' shutdown during orphan-reap failed: %s",
                    name, shutdown_exc,
                )
        raise
    finally:
        if claim_token is not None:
            _connect_server_claim.reset(claim_token)
    return server


# ---------------------------------------------------------------------------
# Handler / check-fn factories
# ---------------------------------------------------------------------------

def _request_lazy_reconnect(server_name: str, server: MCPServerTask) -> bool:
    """Wake a recycled stdio server and wait briefly for a fresh session."""
    if not server._is_recycled_stdio():
        return False

    with _lock:
        loop = _mcp_loop
    if loop is None or not loop.is_running():
        return False

    def _signal_reconnect() -> None:
        server._ready.clear()
        server._reconnect_event.set()

    loop.call_soon_threadsafe(_signal_reconnect)

    async def _await_ready() -> bool:
        deadline = time.monotonic() + _RECYCLED_RECONNECT_TIMEOUT
        while time.monotonic() < deadline:
            if server.session is not None and server._ready.is_set():
                return True
            await asyncio.sleep(0.05)
        return False

    try:
        return bool(_run_on_mcp_loop(_await_ready, timeout=_RECYCLED_RECONNECT_TIMEOUT))
    except Exception as exc:
        logger.warning(
            "MCP server '%s': lazy reconnect after stdio recycle failed: %s",
            server_name, exc,
        )
        return False


def _resolve_server_lazy(name: str, config: dict) -> bool:
    """True when this server defers spawn/connect until first tool use.

    Gated per-server by ``mcp_servers.<name>.lazy`` in config (default OFF),
    following the same per-server key pattern as ``idle_timeout_seconds``.
    Design from #56832 (Vansh5632).
    """
    return _parse_boolish(config.get("lazy", False), default=False)


def _ensure_lazy_server_connected(server_name: str) -> bool:
    """Connect a lazily-registered MCP server on demand (sync, blocks caller).

    Composes with the existing connect machinery: respects the per-server
    connect cooldown (#50394), the ``_server_connecting`` dedup set, and
    routes through ``_discover_and_register_server`` so parked/recycle/
    cooldown bookkeeping stays in one place. Returns True when a live
    session is available afterwards.
    """
    with _lock:
        server = _servers.get(server_name)
        if server is not None and server.session is not None:
            return True
        config = _lazy_server_configs.get(server_name)
        if not config:
            return False
        if _connect_cooldown_active(server_name):
            return False
        if server_name in _server_connecting:
            return False
        _server_connecting.add(server_name)
        _server_connect_errors.pop(server_name, None)

    logger.info("MCP server '%s': lazy start on first use", server_name)
    _ensure_mcp_loop()
    connect_timeout = config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)

    async def _connect():
        return await _discover_and_register_server(server_name, config)

    try:
        _run_on_mcp_loop(_connect, timeout=float(connect_timeout) + 30.0)
    except BaseException as exc:
        message = _format_connect_error(exc)
        with _lock:
            _server_connecting.discard(server_name)
            _server_connect_errors[server_name] = message
            _record_connect_failure(server_name)
        logger.warning(
            "Lazy MCP connect failed for '%s': %s", server_name, message,
        )
        return False

    with _lock:
        _server_connecting.discard(server_name)
        _clear_connect_failure(server_name)
        _lazy_server_configs.pop(server_name, None)
        stale_fingerprint = _lazy_server_fingerprints.pop(server_name, None)
        cached_names = _lazy_server_tool_names.pop(server_name, None) or []
        server = _servers.get(server_name)
        live_names = set(
            getattr(server, "_registered_tool_names", []) or []
        )
    # Stale-cache reconciliation: the cached manifest may advertise tools
    # the live server no longer serves. Deregister those phantoms so the
    # model stops seeing tools that can never succeed.
    phantom_names = [n for n in cached_names if n not in live_names]
    if phantom_names:
        from tools.registry import registry

        for tool_name in phantom_names:
            registry.deregister(tool_name)
            _forget_mcp_tool_server(tool_name)
        logger.info(
            "MCP server '%s': deregistered %d phantom cached tool(s) not "
            "served live (stale schema-cache fingerprint %s): %s",
            server_name, len(phantom_names), stale_fingerprint,
            ", ".join(phantom_names),
        )
    return server is not None and server.session is not None


def _get_connected_server_for_call(server_name: str) -> Optional[MCPServerTask]:
    """Return a connected server, lazily reconnecting recycled stdio state.

    Also the single first-use connect point for lazy (schema-cache
    registered) servers, so raw tool calls AND the resource/prompt utility
    handlers all trigger the deferred spawn (#56832).
    """
    with _lock:
        server = _servers.get(server_name)
        is_lazy = server_name in _lazy_server_configs
    if is_lazy and (server is None or server.session is None):
        _ensure_lazy_server_connected(server_name)
        with _lock:
            server = _servers.get(server_name)
        return server
    if server is not None and server.session is None and server._is_recycled_stdio():
        _request_lazy_reconnect(server_name, server)
        with _lock:
            server = _servers.get(server_name)

    # A keepalive/session-expiry reconnect deliberately clears ``_ready``
    # before the old transport is torn down. Treat that as an in-progress
    # reconnect even if ``session`` still temporarily references the stale
    # ClientSession object. Waiting here keeps callers out of that stale-session
    # window instead of generating a second ClosedResourceError.
    if server is not None:
        ready = getattr(server, "_ready", None)
        if ready is not None and hasattr(ready, "is_set") and not ready.is_set():
            old_session = getattr(server, "session", None)
            if _wait_for_server_session_ready(
                server,
                old_session=old_session,
                timeout=5.0,
            ):
                return server
            return None
    return server


def _mark_server_call_started(server: Any) -> None:
    """Record a user-visible MCP operation when the server supports it."""
    mark_tool_call = getattr(server, "mark_tool_call", None)
    if callable(mark_tool_call):
        mark_tool_call()


@asynccontextmanager
async def _track_inflight_rpc(server: Any, server_name: str, op: str):
    """Register the running RPC on the server so teardown can fail it fast.

    Every user-visible request family wraps its RPC in this context
    (#48069 salvage). If a deliberate reconnect/shutdown teardown cancels
    the task (``_fail_inflight_calls`` sets ``_reconnecting`` first), the
    cancel is converted into a clean retryable RuntimeError instead of a raw
    CancelledError; external cancels (caller timeout, user interrupt)
    propagate unchanged.
    """
    inflight = getattr(server, "_inflight_tasks", None)
    task = asyncio.current_task()
    if task is not None and inflight is not None:
        # Test doubles may pass a bare SimpleNamespace; tracking is then
        # simply skipped (fast-fail teardown is a production-connection
        # feature, not something a fake needs).
        inflight.add(task)
    try:
        yield
    except asyncio.CancelledError:
        if getattr(server, "_reconnecting", False):
            raise RuntimeError(
                f"MCP {op} on '{server_name}' was aborted by a reconnect "
                f"teardown; retry the request on the rebuilt session"
            ) from None
        raise
    finally:
        if task is not None and inflight is not None:
            inflight.discard(task)


def _ensure_healthy_or_recycle(server: Any, server_name: str) -> None:
    """Health-check a suspect connection before its next call (#85125 3b).

    Implements the SuspectableBackend cheap-mark/lazy-verify contract at the
    dispatch boundary: a connection latched as suspect by a race or an auth
    error is probed once; a failed probe recycles it so the call below hits
    the normal reconnect path. A HEALTHY connection is never recycled here.
    """
    if not getattr(server, "_suspect_reason", None):
        return
    with _lock:
        loop = _mcp_loop
    if loop is None or not loop.is_running():
        return  # no background loop — nothing to verify against
    try:
        healthy = bool(_run_on_mcp_loop(server.ensure_healthy, timeout=15.0))
    except Exception as exc:  # never let the probe break dispatch
        logger.debug(
            "MCP server '%s': suspect health check errored: %s",
            server_name, exc,
        )
        healthy = False
    if not healthy:
        _signal_reconnect(server)


def _make_tool_handler(server_name: str, tool_name: str, tool_timeout: float):
    """Return a sync handler that calls an MCP tool via the background loop.

    The handler conforms to the registry's dispatch interface:
    ``handler(args_dict, **kwargs) -> str``
    """

    def _handler(args: dict, **kwargs) -> str:
        # Trust-tier gate (security boundary): write-capable tools on
        # servers configured ``trust: untrusted`` must be approved by the
        # user before ANY transport work happens — including the lazy
        # first-use spawn below. A denied call never touches the server.
        gate_error = _trust_gate_check(server_name, tool_name)
        if gate_error is not None:
            return gate_error

        # Circuit breaker: if this server has failed too many times
        # consecutively, short-circuit with a clear message so the model
        # stops retrying and uses alternative approaches (#10447).
        #
        # Once the cooldown elapses, the breaker transitions to
        # half-open: we let the *next* call through as a probe. On
        # success the success-path below resets the breaker; on
        # failure the error paths below bump the count again, which
        # re-stamps the open-time via _bump_server_error (re-arming
        # the cooldown).
        if _server_error_counts.get(server_name, 0) >= _CIRCUIT_BREAKER_THRESHOLD:
            opened_at = _server_breaker_opened_at.get(server_name, 0.0)
            age = time.monotonic() - opened_at
            if age < _CIRCUIT_BREAKER_COOLDOWN_SEC:
                remaining = max(1, int(_CIRCUIT_BREAKER_COOLDOWN_SEC - age))
                return tool_error(
                    f"MCP server '{server_name}' is unreachable after "
                    f"{_server_error_counts[server_name]} consecutive "
                    f"failures. Auto-retry available in ~{remaining}s. "
                    f"Do NOT retry this tool yet — use alternative "
                    f"approaches or ask the user to check the MCP server."
                )
            # Cooldown elapsed → fall through as a half-open probe.

        server = _get_connected_server_for_call(server_name)
        if not server:
            _bump_server_error(server_name)
            return tool_error(f"MCP server '{server_name}' is not connected")

        if not server.session:
            # No live session. A reconnect may already be completing (the
            # transport swaps in a fresh session object asynchronously) —
            # wait briefly before treating this as a failure, so a
            # transient reconnect window doesn't burn a circuit-breaker
            # strike (#26892).
            if _wait_for_server_session_ready(
                server, timeout=min(5.0, float(tool_timeout or 5.0)),
            ):
                pass  # Fresh session arrived; proceed below.
            else:
                # Still down — the server task is reconnecting, or it has
                # exhausted its retry budget and parked (e.g. a dead stdio
                # subprocess). Probing here would write into a dead/absent
                # transport and re-arm the breaker forever (#16788). Instead,
                # ask the (always-present) server task to rebuild the
                # transport — which respawns a dead stdio subprocess — and
                # return a clean "reconnecting" error so the model backs off
                # without burning iterations. The breaker resets once the
                # fresh session initializes (_run_stdio/_run_http call
                # _reset_server_error).
                _bump_server_error(server_name)
                if _signal_reconnect(server):
                    return tool_error(
                        f"MCP server '{server_name}' transport is down; "
                        f"reconnect requested. Do NOT retry this tool "
                        f"immediately — give it a few seconds to come back."
                    )
                return tool_error(f"MCP server '{server_name}' is not connected")

        async def _call():
            _mark_server_call_started(server)
            async with server._rpc_lock, _track_inflight_rpc(
                server, server_name, f"tools/call {tool_name}"
            ):
                # Snapshot the agent's context so an elicitation callback
                # triggered during this call (fired on the MCP recv loop
                # task, which doesn't inherit our contextvars) can replay
                # it and detect the gateway platform / session for routing.
                server._pending_call_context = contextvars.copy_context()
                try:
                    # Fast-fail (#81995): a stdio subprocess that is already
                    # dead must not own this call slot — fail immediately
                    # instead of waiting out the full tool timeout on a
                    # transport nobody will ever answer.
                    _stdio_dead = getattr(server, "_stdio_children_dead", None)
                    # callable() + real-bool result: MagicMock attributes return
                    # truthy Mocks, which would spuriously trip the fast-fail.
                    if (
                        callable(_stdio_dead)
                        and isinstance(_stdio_dead_result := _stdio_dead(), bool)
                        and _stdio_dead_result
                    ):
                        # Dead children but stale server.session, so the
                        # transport-down path above never fired — signal the
                        # server task to respawn and return a clean
                        # reconnecting error. No explicit _bump_server_error:
                        # the error return flows through the handler's JSON
                        # parse, which already bumps once.
                        if _signal_reconnect(server):
                            return tool_error(
                                f"MCP server '{server_name}' stdio subprocess is "
                                f"dead and reconnect was requested. Do NOT retry "
                                f"immediately — give it a few seconds to respawn."
                            )
                        raise TimeoutError(
                            f"MCP stdio subprocess for '{server_name}' has "
                            f"exited; failing the call fast instead of "
                            f"waiting {float(tool_timeout):.0f}s"
                        )
                    _call_coro = server.session.call_tool(tool_name, arguments=args)
                    _watch_children = getattr(server, "_watch_stdio_children", None)
                    _watch_ok = (
                        _watch_children is not None
                        and inspect.isawaitable(_watch_children())
                        and asyncio.iscoroutine(_call_coro)
                    )
                    if not _watch_ok:
                        # Stubbed sessions (MagicMock in tests) return a
                        # non-awaitable, or there is no child-watcher to race
                        # against: plain await is exactly the pre-#81995
                        # semantics.
                        result = (
                            await _call_coro
                            if asyncio.iscoroutine(_call_coro)
                            else _call_coro
                        )
                    else:
                        # Fast-fail machinery (#81995): the RPC races a
                        # stdio-children watcher so a dead subprocess fails
                        # the call immediately instead of riding out the full
                        # tool timeout.
                        rpc_task = asyncio.ensure_future(_call_coro)
                        watch_task = asyncio.ensure_future(_watch_children())
                        try:
                            done, _pending = await asyncio.wait(
                                {rpc_task, watch_task},
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if watch_task in done and not rpc_task.done():
                                rpc_task.cancel()
                                # Same stale-session problem as the pre-call
                                # gate above: the subprocess died mid-call but
                                # nothing clears server.session, so without a
                                # reconnect signal the server would stay dead
                                # until the idle keepalive probe notices.
                                _signal_reconnect(server)
                                raise TimeoutError(
                                    f"MCP stdio subprocess for '{server_name}' "
                                    f"exited mid-call; failing the call fast "
                                    f"instead of waiting "
                                    f"{float(tool_timeout):.0f}s; reconnect "
                                    f"requested — give it a few seconds to "
                                    f"respawn before retrying"
                                )
                            result = await rpc_task
                        finally:
                            watch_task.cancel()
                            if not rpc_task.done():
                                rpc_task.cancel()
                            await asyncio.gather(
                                rpc_task, watch_task, return_exceptions=True
                            )
                finally:
                    server._pending_call_context = None
            # The RPC round-trip completed — the session is demonstrably
            # healthy at the transport level (even if the tool itself
            # returned isError). Clear the rapid-drop budget (#62212).
            _mark_proven = getattr(server, "_mark_session_proven", None)
            if _mark_proven is not None:
                _mark_proven()
            # MCP CallToolResult has .content (list of content blocks) and
            # .is_error (.isError before mcp 2.0)
            if mcp_field(result, "is_error", "isError", False):
                error_text = ""
                for block in (result.content or []):
                    if getattr(block, "text", None):
                        error_text += block.text
                        continue
                    # EmbeddedResource blocks inside error payloads carry
                    # their text under .resource.text — previously dropped,
                    # leaving a bare "MCP tool returned an error".
                    res_text = getattr(getattr(block, "resource", None), "text", None)
                    if res_text:
                        error_text += str(res_text)
                return tool_error(_sanitize_error(
                    _truncate_mcp_text_result(
                        error_text or "MCP tool returned an error"
                    )
                ))

            # Collect text from content blocks. MCP tool results can also
            # include ImageContent blocks (screenshot / Blockbench / Playwright
            # etc.); cache those via the gateway's image-cache helper so they
            # flow through Hermes' MEDIA: tag convention and out to messaging
            # adapters that render images natively. Without this, image blocks
            # were silently dropped and the agent got an empty response.
            #
            # Distilled from #17915 (c3115644151) and #10848 (gnanirahulnutakki),
            # both too stale to cherry-pick. #10848's approach (integrate with
            # Hermes' MEDIA tag + cache_image_from_bytes) was the cleaner of
            # the two — plugs into existing infrastructure.
            parts: List[str] = []
            for block in (result.content or []):
                if hasattr(block, "text") and block.text:
                    parts.append(strip_unicode_tags(block.text))
                    continue
                image_tag = _cache_mcp_image_block(block)
                if image_tag:
                    parts.append(image_tag)
                    continue
                audio_tag = _cache_mcp_audio_block(block)
                if audio_tag:
                    parts.append(audio_tag)
                    continue
                # ResourceLink / EmbeddedResource blocks (PDFs, archives,
                # office docs, ...). Previously these were silently dropped,
                # so document-oriented MCP tools appeared to return metadata
                # only (enterprise customer report, 2026-07).
                resource_text = _render_mcp_resource_block(block, server_name)
                if resource_text:
                    parts.append(resource_text)
                    continue
                # Benign empty renders (empty text blocks, empty text
                # resources, audio in a process without the gateway cache)
                # aren't data loss — log at debug. Warn only for genuinely
                # unrecognized block shapes.
                block_type = getattr(block, "type", None) or type(block).__name__
                if block_type in {"text", "resource", "audio", "image"}:
                    logger.debug(
                        "MCP %s: content block type %r rendered empty",
                        server_name, block_type,
                    )
                else:
                    logger.warning(
                        "MCP %s: dropping unsupported content block type %r",
                        server_name, block_type,
                    )
            text_result = "\n".join(parts) if parts else ""

            # Hard-cap pathological payloads before they propagate (#56059);
            # ordinary large results pass untouched to the spillover layer.
            text_result = _truncate_mcp_text_result(text_result)

            # Combine content + structuredContent when both are present.
            # MCP spec: content is model-oriented (text), structuredContent
            # is machine-oriented (JSON metadata).  For an AI agent, content
            # is the primary payload; structuredContent supplements it.
            #
            # Server-level `_meta` is also surfaced (ported from
            # MoonshotAI/kimi-code#2596): servers return namespaced metadata
            # there (validated contracts, browser-handoff payloads, ...) that
            # was previously invisible to the agent. Protocol-reserved keys
            # are dropped first (kimi-code#2600) — per the MCP spec's key-name
            # rules a prefix is reserved when a `modelcontextprotocol` or
            # `mcp` label is followed by at least one more label (e.g.
            # `modelcontextprotocol.io/...`, `tools.mcp.com/...`); those carry
            # host/protocol plumbing, not model-facing data. Unprefixed and
            # vendor-namespaced keys (`com.example.mcp/...`) pass through —
            # their semantics belong to the server.
            structured = mcp_field(result, "structured_content", "structuredContent")
            # Cap structuredContent too — a malicious server could flood
            # context via a multi-MB JSON payload (#56059). When the
            # serialized form exceeds the hard cap, replace it with the
            # truncated string (head + tail preserved) so it degrades
            # gracefully instead of flooding downstream.
            if structured is not None:
                try:
                    _structured_json = json.dumps(structured, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    _structured_json = None
                if _structured_json is not None and len(_structured_json) > _MCP_HARD_RESULT_CAP_CHARS:
                    structured = _truncate_mcp_text_result(_structured_json)
            meta = _strip_reserved_meta_keys(mcp_field(result, "meta", "meta"))
            if structured is not None or meta is not None:
                payload: Dict[str, Any] = {}
                if text_result:
                    payload["result"] = text_result
                if structured is not None:
                    if text_result:
                        payload["structuredContent"] = structured
                    else:
                        payload["result"] = structured
                if meta is not None:
                    payload["_meta"] = meta
                if "result" not in payload:
                    payload["result"] = text_result
                try:
                    return json.dumps(payload, ensure_ascii=False)
                except (TypeError, ValueError):
                    # Non-serializable metadata: drop the extras rather than
                    # failing the whole tool call.
                    return json.dumps({"result": text_result}, ensure_ascii=False)
            return json.dumps({"result": text_result}, ensure_ascii=False)

        def _call_once():
            return _run_on_mcp_loop(_call, timeout=tool_timeout)

        try:
            result = _call_once()
            # Check if the MCP tool itself returned an error
            try:
                parsed = json.loads(result)
                if "error" in parsed:
                    _bump_server_error(server_name)
                else:
                    _reset_server_error(server_name)  # success — reset
            except (json.JSONDecodeError, TypeError):
                _reset_server_error(server_name)  # non-JSON = success
            return result
        except InterruptedError:
            return _interrupted_call_result()
        except Exception as exc:
            # Auth-specific recovery path: consult the manager, signal
            # reconnect if viable, retry once. Returns None to fall
            # through for non-auth exceptions.
            recovered = _handle_auth_error_and_retry(
                server_name, exc, _call_once,
                f"tools/call {tool_name}",
            )
            if recovered is not None:
                return recovered

            # Transport session expiry (#13383): same reconnect flow
            # but skips OAuth recovery because the access token is
            # still valid — only the server-side session is stale.
            recovered = _handle_session_expired_and_retry(
                server_name, exc, _call_once,
                f"tools/call {tool_name}",
            )
            if recovered is not None:
                return recovered

            _bump_server_error(server_name)
            logger.error(
                "MCP tool %s/%s call failed: %s",
                server_name, tool_name, exc,
            )
            return tool_error(_sanitize_error(
                f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
            ))

    return _handler


def _make_list_resources_handler(server_name: str, tool_timeout: float):
    """Return a sync handler that lists resources from an MCP server."""

    def _handler(args: dict, **kwargs) -> str:
        server = _get_connected_server_for_call(server_name)
        if not server or not server.session:
            return tool_error(f"MCP server '{server_name}' is not connected")

        async def _call():
            _mark_server_call_started(server)
            async with server._rpc_lock:
                all_resources = await _paginate_full_list(
                    server.session.list_resources, "resources", server_name
                )
            resources = []
            for r in all_resources:
                entry = {}
                if hasattr(r, "uri"):
                    entry["uri"] = str(r.uri)
                if hasattr(r, "name"):
                    entry["name"] = r.name
                if hasattr(r, "description") and r.description:
                    entry["description"] = r.description
                # Key stays camelCase — this dict is the tool's own JSON
                # output shape, not an SDK model.
                _mime = mcp_field(r, "mime_type", "mimeType")
                if _mime:
                    entry["mimeType"] = _mime
                resources.append(entry)
            return json.dumps({"resources": resources}, ensure_ascii=False)

        def _call_once():
            return _run_on_mcp_loop(_call, timeout=tool_timeout)

        try:
            return _call_once()
        except InterruptedError:
            return _interrupted_call_result()
        except Exception as exc:
            recovered = _handle_auth_error_and_retry(
                server_name, exc, _call_once, "resources/list",
            )
            if recovered is not None:
                return recovered
            recovered = _handle_session_expired_and_retry(
                server_name, exc, _call_once, "resources/list",
            )
            if recovered is not None:
                return recovered
            logger.error(
                "MCP %s/list_resources failed: %s", server_name, exc,
            )
            return tool_error(_sanitize_error(
                f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
            ))

    return _handler


def _make_read_resource_handler(server_name: str, tool_timeout: float):
    """Return a sync handler that reads a resource by URI from an MCP server."""

    def _handler(args: dict, **kwargs) -> str:
        server = _get_connected_server_for_call(server_name)
        if not server or not server.session:
            return tool_error(f"MCP server '{server_name}' is not connected")

        uri = args.get("uri")
        if not uri:
            return tool_error("Missing required parameter 'uri'")

        async def _call():
            _mark_server_call_started(server)
            async with server._rpc_lock:
                result = await server.session.read_resource(uri)
            # read_resource returns ReadResourceResult with .contents list
            parts: List[str] = []
            contents = result.contents if hasattr(result, "contents") else []
            for block in contents:
                if getattr(block, "text", None) is not None:
                    parts.append(strip_unicode_tags(block.text))
                elif getattr(block, "blob", None) is not None:
                    # Materialize binary resource contents into the document
                    # cache instead of discarding them (same contract as
                    # EmbeddedResource blocks in tool results).
                    rendered = _render_mcp_resource_block(
                        SimpleNamespace(type="resource", resource=block),
                        server_name,
                    )
                    parts.append(rendered or f"[binary data, {len(block.blob)} bytes]")
            return json.dumps({"result": "\n".join(parts) if parts else ""}, ensure_ascii=False)

        def _call_once():
            return _run_on_mcp_loop(_call, timeout=tool_timeout)

        try:
            return _call_once()
        except InterruptedError:
            return _interrupted_call_result()
        except Exception as exc:
            recovered = _handle_auth_error_and_retry(
                server_name, exc, _call_once, "resources/read",
            )
            if recovered is not None:
                return recovered
            recovered = _handle_session_expired_and_retry(
                server_name, exc, _call_once, "resources/read",
            )
            if recovered is not None:
                return recovered
            logger.error(
                "MCP %s/read_resource failed: %s", server_name, exc,
            )
            return tool_error(_sanitize_error(
                f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
            ))

    return _handler


def _make_list_prompts_handler(server_name: str, tool_timeout: float):
    """Return a sync handler that lists prompts from an MCP server."""

    def _handler(args: dict, **kwargs) -> str:
        server = _get_connected_server_for_call(server_name)
        if not server or not server.session:
            return tool_error(f"MCP server '{server_name}' is not connected")

        async def _call():
            _mark_server_call_started(server)
            async with server._rpc_lock:
                all_prompts = await _paginate_full_list(
                    server.session.list_prompts, "prompts", server_name
                )
            prompts = []
            for p in all_prompts:
                entry = {}
                if hasattr(p, "name"):
                    entry["name"] = p.name
                if hasattr(p, "description") and p.description:
                    entry["description"] = p.description
                if hasattr(p, "arguments") and p.arguments:
                    entry["arguments"] = [
                        {
                            "name": a.name,
                            **({"description": a.description} if hasattr(a, "description") and a.description else {}),
                            **({"required": a.required} if hasattr(a, "required") else {}),
                        }
                        for a in p.arguments
                    ]
                prompts.append(entry)
            return json.dumps({"prompts": prompts}, ensure_ascii=False)

        def _call_once():
            return _run_on_mcp_loop(_call, timeout=tool_timeout)

        try:
            return _call_once()
        except InterruptedError:
            return _interrupted_call_result()
        except Exception as exc:
            recovered = _handle_auth_error_and_retry(
                server_name, exc, _call_once, "prompts/list",
            )
            if recovered is not None:
                return recovered
            recovered = _handle_session_expired_and_retry(
                server_name, exc, _call_once, "prompts/list",
            )
            if recovered is not None:
                return recovered
            logger.error(
                "MCP %s/list_prompts failed: %s", server_name, exc,
            )
            return tool_error(_sanitize_error(
                f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
            ))

    return _handler


def _make_get_prompt_handler(server_name: str, tool_timeout: float):
    """Return a sync handler that gets a prompt by name from an MCP server."""

    def _handler(args: dict, **kwargs) -> str:
        server = _get_connected_server_for_call(server_name)
        if not server or not server.session:
            return tool_error(f"MCP server '{server_name}' is not connected")

        name = args.get("name")
        if not name:
            return tool_error("Missing required parameter 'name'")
        arguments = args.get("arguments", {})

        async def _call():
            _mark_server_call_started(server)
            async with server._rpc_lock:
                result = await server.session.get_prompt(name, arguments=arguments)
            # GetPromptResult has .messages list
            messages = []
            for msg in (result.messages if hasattr(result, "messages") else []):
                entry = {}
                if hasattr(msg, "role"):
                    entry["role"] = msg.role
                if hasattr(msg, "content"):
                    content = msg.content
                    if hasattr(content, "text"):
                        entry["content"] = strip_unicode_tags(content.text)
                    elif isinstance(content, str):
                        entry["content"] = strip_unicode_tags(content)
                    else:
                        entry["content"] = strip_unicode_tags(str(content))
                messages.append(entry)
            resp = {"messages": messages}
            if hasattr(result, "description") and result.description:
                resp["description"] = result.description
            return json.dumps(resp, ensure_ascii=False)

        def _call_once():
            return _run_on_mcp_loop(_call, timeout=tool_timeout)

        try:
            return _call_once()
        except InterruptedError:
            return _interrupted_call_result()
        except Exception as exc:
            recovered = _handle_auth_error_and_retry(
                server_name, exc, _call_once, "prompts/get",
            )
            if recovered is not None:
                return recovered
            recovered = _handle_session_expired_and_retry(
                server_name, exc, _call_once, "prompts/get",
            )
            if recovered is not None:
                return recovered
            logger.error(
                "MCP %s/get_prompt failed: %s", server_name, exc,
            )
            return tool_error(_sanitize_error(
                f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
            ))

    return _handler


def _make_check_fn(server_name: str):
    """Return a check function that verifies the MCP connection is alive."""

    def _check() -> bool:
        with _lock:
            server = _servers.get(server_name)
            if server is not None and (
                server.session is not None or server._is_recycled_stdio()
            ):
                return True
            # Lazy (schema-cache registered) servers are available: the
            # first real call spawns/connects them (#56832).
            return server_name in _lazy_server_configs

    return _check


# ---------------------------------------------------------------------------
# Discovery & registration
# ---------------------------------------------------------------------------

def _normalize_mcp_input_schema(schema: dict | None) -> dict:
    """Normalize MCP input schemas for LLM tool-calling compatibility.

    MCP servers can emit plain JSON Schema with ``definitions`` /
    ``#/definitions/...`` references.  Kimi / Moonshot rejects that form and
    requires local refs to point into ``#/$defs/...`` instead.  Normalize the
    common draft-07 shape here so MCP tool schemas remain portable across
    OpenAI-compatible providers.

    Additional MCP-server robustness repairs applied recursively:

    * Missing or ``null`` ``type`` on an object-shaped node is coerced to
      ``"object"`` (some servers omit it).  See PR #4897.
    * When an ``object`` node lacks ``properties``, an empty ``properties``
      dict is added so ``required`` entries don't dangle.
    * ``required`` arrays are pruned to only names that exist in
      ``properties``; otherwise Google AI Studio / Gemini 400s with
      ``property is not defined``.  See PR #4651.
    * MCP/Pydantic optional fields commonly arrive as
      ``anyOf: [{...}, {"type": "null"}], default: null``.  Anthropic rejects
      nullable branches in tool input schemas, so nullable unions are collapsed
      to the non-null branch and optionality remains represented solely by the
      parent object's ``required`` list.

    All repairs are provider-agnostic and ideally produce a schema valid on
    OpenAI, Anthropic, Gemini, and Moonshot in one pass.
    """
    if not schema:
        return {"type": "object", "properties": {}}

    def _rewrite_local_refs(node):
        """Walk the schema, promoting legacy ``definitions`` to ``$defs``.

        The promotion is contextual: ``definitions`` is renamed only when it
        appears as a JSON Schema *meta-keyword* (sibling of ``properties`` /
        ``$ref`` at a schema node), never when it appears as the *name of a
        property* (i.e., as a key inside a ``properties`` dict).

        Without this gate, MCP servers that legitimately expose a tool
        parameter named ``definitions`` (e.g. a CI/pipelines tool that uses
        ``definitions`` for an array of pipeline-definition IDs) would have
        that user-facing property name silently rewritten to ``$defs``.
        Anthropic and OpenAI both reject ``$`` in property names
        (``^[a-zA-Z0-9_.-]{1,64}$``), so the whole tool array gets a 400 and
        every conversation breaks.

        The gate works by treating ``properties`` and ``patternProperties``
        specially during descent: we iterate the property-name -> schema map
        directly, leaving the property names verbatim, then recurse into each
        property's schema where ordinary JSON Schema semantics resume (so any
        legitimately-nested ``definitions`` meta-keyword inside a property's
        schema is still promoted).
        """
        if isinstance(node, dict):
            normalized = {}
            for key, value in node.items():
                if key in ("properties", "patternProperties") and isinstance(value, dict):
                    # Keys of this dict are user-facing property names, not
                    # meta-keywords. Preserve them verbatim; recurse only into
                    # each property's schema, where ``definitions`` again has
                    # its JSON Schema meaning.
                    normalized[key] = {
                        prop_name: _rewrite_local_refs(prop_schema)
                        for prop_name, prop_schema in value.items()
                    }
                else:
                    out_key = "$defs" if key == "definitions" else key
                    normalized[out_key] = _rewrite_local_refs(value)
            ref = normalized.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/definitions/"):
                normalized["$ref"] = "#/$defs/" + ref[len("#/definitions/"):]
            return normalized
        if isinstance(node, list):
            return [_rewrite_local_refs(item) for item in node]
        return node

    def _strip_nullable_union(node):
        """Collapse JSON Schema nullable unions to provider-safe non-null schemas.

        Delegates to ``tools.schema_sanitizer.strip_nullable_unions`` so MCP
        ingestion, the Anthropic guard, and the global sanitizer all share one
        implementation. Keeps the ``nullable: true`` hint so runtime argument
        coercion can still map a model-emitted ``"null"`` string to Python
        ``None`` for this optional field.
        """
        from tools.schema_sanitizer import strip_nullable_unions

        return strip_nullable_unions(node, keep_nullable_hint=True)

    def _collapse_const_unions(node):
        """Collapse anyOf/oneOf unions of same-typed consts to property enums.

        Delegates to ``tools.schema_sanitizer.collapse_const_unions``. Runs
        AFTER the nullable strip: single-non-null unions are already collapsed
        by then, and unions of several const branches plus a null branch are
        handled here (consts -> enum, null -> ``nullable: true`` hint).
        Ported from block/goose tool_schema_normalize.rs (Apache-2.0).
        """
        from tools.schema_sanitizer import collapse_const_unions

        return collapse_const_unions(node)

    def _repair_object_shape(node):
        """Recursively repair object-shaped nodes: fill type, prune required."""
        if isinstance(node, list):
            return [_repair_object_shape(item) for item in node]
        if not isinstance(node, dict):
            return node

        repaired = {k: _repair_object_shape(v) for k, v in node.items()}

        # Coerce missing / null type when the shape is clearly an object
        # (has properties or required but no type).
        if not repaired.get("type") and (
            "properties" in repaired or "required" in repaired
        ):
            repaired["type"] = "object"

        if repaired.get("type") == "object":
            # Ensure properties exists so required can reference it safely
            if "properties" not in repaired or not isinstance(
                repaired.get("properties"), dict
            ):
                repaired["properties"] = {} if "properties" not in repaired else repaired["properties"]
                if not isinstance(repaired.get("properties"), dict):
                    repaired["properties"] = {}

            # Prune required to only include names that exist in properties
            required = repaired.get("required")
            if isinstance(required, list):
                props = repaired.get("properties") or {}
                valid = [r for r in required if isinstance(r, str) and r in props]
                if len(valid) != len(required):
                    if valid:
                        repaired["required"] = valid
                    else:
                        repaired.pop("required", None)

        return repaired

    normalized = _rewrite_local_refs(schema)
    normalized = _strip_nullable_union(normalized)
    normalized = _collapse_const_unions(normalized)
    normalized = _repair_object_shape(normalized)

    # Ensure top-level is a well-formed object schema
    if not isinstance(normalized, dict):
        return {"type": "object", "properties": {}}
    if normalized.get("type") == "object" and "properties" not in normalized:
        normalized = {**normalized, "properties": {}}

    return normalized


def sanitize_mcp_name_component(value: str) -> str:
    """Return an MCP name component safe for tool and prefix generation.

    Preserves Hermes's historical behavior of converting hyphens to
    underscores, and also replaces any other character outside
    ``[A-Za-z0-9_]`` with ``_`` so generated tool names are compatible with
    provider validation rules.
    """
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))


# Native MCP tool-name prefix. Hermes uses the ``mcp__<server>__<tool>``
# convention shared by Claude Code, Codex, and OpenCode (anomalyco/opencode
# #33533). The double-underscore delimiter disambiguates the server/tool
# boundary even when either component contains underscores, and matches the
# naming models are trained on. It also aligns native registration with the
# Anthropic-OAuth wire form (``_MCP_TOOL_PREFIX`` in anthropic_adapter.py),
# removing the single->double rewrite that path previously had to perform.
MCP_TOOL_NAME_PREFIX = "mcp__"
_MCP_NAME_DELIM = "__"


def mcp_prefixed_tool_name(server_name: str, tool_name: str) -> str:
    """Build the registry/wire name for an MCP tool.

    Produces ``mcp__<sanitizedServer>__<sanitizedTool>``.
    """
    safe_server = sanitize_mcp_name_component(server_name)
    safe_tool = sanitize_mcp_name_component(tool_name)
    return f"{MCP_TOOL_NAME_PREFIX}{safe_server}{_MCP_NAME_DELIM}{safe_tool}"


def _convert_mcp_schema(server_name: str, mcp_tool) -> dict:
    """Convert an MCP tool listing to the Hermes registry schema format.

    Args:
        server_name: The logical server name for prefixing.
        mcp_tool:    An MCP ``Tool`` object with ``.name``, ``.description``,
                     and ``.input_schema`` (``.inputSchema`` before mcp 2.0).

    Returns:
        A dict suitable for ``registry.register(schema=...)``.
    """
    prefixed_name = mcp_prefixed_tool_name(server_name, mcp_tool.name)
    return {
        "name": prefixed_name,
        "description": strip_unicode_tags(
            mcp_tool.description or f"MCP tool {mcp_tool.name} from {server_name}"
        ),
        "parameters": _normalize_mcp_input_schema(
            mcp_field(mcp_tool, "input_schema", "inputSchema")
        ),
    }


def _build_utility_schemas(server_name: str) -> List[dict]:
    """Build schemas for the MCP utility tools (resources & prompts).

    Returns a list of (schema, handler_factory_name) tuples encoded as dicts
    with keys: schema, handler_key.
    """
    return [
        {
            "schema": {
                "name": mcp_prefixed_tool_name(server_name, "list_resources"),
                "description": f"List available resources from MCP server '{server_name}'",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            "handler_key": "list_resources",
        },
        {
            "schema": {
                "name": mcp_prefixed_tool_name(server_name, "read_resource"),
                "description": f"Read a resource by URI from MCP server '{server_name}'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uri": {
                            "type": "string",
                            "description": "URI of the resource to read",
                        },
                    },
                    "required": ["uri"],
                },
            },
            "handler_key": "read_resource",
        },
        {
            "schema": {
                "name": mcp_prefixed_tool_name(server_name, "list_prompts"),
                "description": f"List available prompts from MCP server '{server_name}'",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            "handler_key": "list_prompts",
        },
        {
            "schema": {
                "name": mcp_prefixed_tool_name(server_name, "get_prompt"),
                "description": f"Get a prompt by name from MCP server '{server_name}'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the prompt to retrieve",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Optional arguments to pass to the prompt",
                            "properties": {},
                            "additionalProperties": True,
                        },
                    },
                    "required": ["name"],
                },
            },
            "handler_key": "get_prompt",
        },
    ]


def _normalize_name_filter(value: Any, label: str) -> set[str]:
    """Normalize include/exclude config to a set of tool-name patterns.

    Entries may be exact tool names or fnmatch-style globs
    (``*_radar_*``, ``get_zones_*``). Matching happens in
    :func:`matches_name_filter`.
    """
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    logger.warning("MCP config %s must be a string or list of strings; ignoring %r", label, value)
    return set()


def matches_name_filter(tool_name: str, patterns: set[str]) -> bool:
    """True if ``tool_name`` matches any entry in ``patterns``.

    Exact names match literally; entries containing fnmatch metacharacters
    (``*``, ``?``, ``[``) match as case-sensitive globs — the same pattern
    semantics as ``approvals.deny``. Exact membership is checked first so
    large literal lists stay O(1).
    """
    if not patterns:
        return False
    if tool_name in patterns:
        return True
    return any(
        fnmatch.fnmatchcase(tool_name, p)
        for p in patterns
        if "*" in p or "?" in p or "[" in p
    )


def _parse_boolish(value: Any, default: bool = True) -> bool:
    """Parse a bool-like config value with safe fallback."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    logger.warning("MCP config expected a boolean-ish value, got %r; using default=%s", value, default)
    return default


def _get_lifecycle_seconds(config: dict, key: str) -> Optional[float]:
    """Return an optional positive lifecycle timeout from top-level/nested config."""
    raw = config.get(key)
    lifecycle = config.get("lifecycle")
    if raw is None and isinstance(lifecycle, dict):
        raw = lifecycle.get(key)
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        logger.warning("MCP config %s must be a number of seconds; ignoring %r", key, raw)
        return None
    if seconds == 0:
        return None
    if seconds < 0:
        logger.warning("MCP config %s must be positive; ignoring %r", key, raw)
        return None
    return seconds


_UTILITY_CAPABILITY_METHODS = {
    "list_resources": "list_resources",
    "read_resource": "read_resource",
    "list_prompts": "list_prompts",
    "get_prompt": "get_prompt",
}

_plugin_compat_prev_getattr = __getattr__


def __getattr__(name):  # PEP 562 — chained onto the module's own __getattr__
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        return _plugin_compat_prev_getattr(name)
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
