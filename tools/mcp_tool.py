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
# Remote clients MUST ping faster than the server's idle-session TTL (short-TTL servers need a
# smaller configured ``keepalive_interval``); stdio only opts in explicitly because local pipes
# have no remote session TTL. The floor stops a tiny interval busy-looping.
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
            try:
                import mcp.types as _types  # late: keeps the SDK import lazy
                _params_cls = getattr(_types, "PaginatedRequestParams", None)
                result = await (list_method(params=_params_cls(cursor=cursor)) if _params_cls is not None
                                else list_method(cursor=cursor))
            except TypeError:
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

    # Content types a real Streamable-HTTP endpoint may return on the initial POST/GET;
    # anything else on a 2xx means the URL is not an MCP endpoint.
    _MCP_CONTENT_TYPES = ("application/json", "text/event-stream")


# ---- Module-level state (every mutation under ``_lock``) ----

_servers: Dict[str, MCPServerTask] = {}
# Profile registry scope per live connection (None outside multiplex) so a multiplexed
# /reload-mcp tears down only its own profile's servers.
_server_scope_keys: Dict[str, Optional[str]] = {}
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


# Cross-process discovery guard: advisory file lock so gateway + CLI + TUI don't all discover.
# See issue #62771.
_LOCK_UNAVAILABLE: Any = object()  # sentinel: locking broken/unavailable
_MCP_DISCOVERY_LOCK_PATH: Optional[str] = None  # resolved lazily
# Bounded wait when another process holds the lock.
_MCP_DISCOVERY_LOCK_MAX_RETRIES, _MCP_DISCOVERY_LOCK_RETRY_DELAY_S = 240, 0.5


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Coroutine  # noqa: F401,E402
from types import SimpleNamespace  # noqa: F401,E402
from typing import Tuple  # noqa: F401,E402
from contextlib import asynccontextmanager  # noqa: F401,E402
import concurrent.futures  # noqa: F401,E402
from datetime import datetime  # noqa: F401,E402
import errno  # noqa: F401,E402
import fnmatch  # noqa: F401,E402
import json  # noqa: F401,E402
import math  # noqa: F401,E402
import random  # noqa: F401,E402
import re  # noqa: F401,E402
import shutil  # noqa: F401,E402
from urllib.parse import urlparse  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'InvalidMcpUrlError': ('tools.mcp_tool_errors', 'InvalidMcpUrlError'),
    'MCP_TOOL_NAME_PREFIX': ('tools.mcp_tool_schema', 'MCP_TOOL_NAME_PREFIX'),
    'NonMcpEndpointError': ('tools.mcp_tool_errors', 'NonMcpEndpointError'),
    'discover_mcp_tools': ('tools.mcp_tool_discovery', 'discover_mcp_tools'),
    'get_mcp_status': ('tools.mcp_tool_discovery', 'get_mcp_status'),
    'get_registered_mcp_server_names': ('tools.mcp_tool_discovery', 'get_registered_mcp_server_names'),
    'has_registered_mcp_tools': ('tools.mcp_tool_discovery', 'has_registered_mcp_tools'),
    'is_mcp_tool_parallel_safe': ('tools.mcp_tool_discovery', 'is_mcp_tool_parallel_safe'),
    'matches_name_filter': ('tools.mcp_tool_schema', 'matches_name_filter'),
    'mcp_prefixed_tool_name': ('tools.mcp_tool_schema', 'mcp_prefixed_tool_name'),
    'persist_agent_tool_names': ('tools.mcp_tool_agent', 'persist_agent_tool_names'),
    'probe_mcp_server_tools': ('tools.mcp_tool_discovery', 'probe_mcp_server_tools'),
    'reconnect_mcp_server': ('tools.mcp_tool_loop', 'reconnect_mcp_server'),
    'refresh_agent_mcp_tools': ('tools.mcp_tool_agent', 'refresh_agent_mcp_tools'),
    'register_mcp_servers': ('tools.mcp_tool_discovery', 'register_mcp_servers'),
    'reprobe_tool_availability': ('tools.mcp_tool_agent', 'reprobe_tool_availability'),
    'restore_agent_tool_prefix': ('tools.mcp_tool_agent', 'restore_agent_tool_prefix'),
    'sanitize_mcp_name_component': ('tools.mcp_tool_schema', 'sanitize_mcp_name_component'),
    'shutdown_mcp_servers': ('tools.mcp_tool_lifecycle', 'shutdown_mcp_servers'),
    'strip_unicode_tags': ('tools.ansi_strip', 'strip_unicode_tags'),
    'tool_error': ('tools.registry', 'tool_error'),
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
