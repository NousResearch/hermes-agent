#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Client Support

Connects to external MCP servers via stdio, HTTP/StreamableHTTP, or SSE
transport, discovers their tools, and registers them into the hermes-agent
tool registry so the agent can call them like any built-in tool.

Configuration is read from ~/.hermes/config.yaml under the ``mcp_servers`` key.
The ``mcp`` Python package is optional -- if not installed, this module is a
no-op and logs a debug message.

Example config::

    mcp_servers:
      filesystem:
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        env: {}
        timeout: 120         # per-tool-call timeout in seconds (default: 300)
        connect_timeout: 60  # initial connection timeout (default: 60)
        keepalive_interval: 10  # liveness ping cadence in seconds (default:
                                # 180). Set below the server's session TTL for
                                # servers that GC idle sessions quickly (e.g.
                                # Unreal Engine editor MCP, ~15s). Floored at 5s.
      github:
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-github"]
        env:
          GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
        supports_parallel_tool_calls: true  # tools from this server may run concurrently
      remote_api:
        url: "https://my-mcp-server.example.com/mcp"
        headers:
          Authorization: "Bearer sk-..."
        timeout: 180
        skip_preflight: true  # bypass the content-type probe for a valid
                              # Streamable HTTP endpoint that answers HEAD/GET
                              # with a non-MCP content type but serves real
                              # MCP over POST. Default: false.
      searxng:
        url: "http://localhost:8000/sse"
        transport: sse       # use SSE transport instead of Streamable HTTP
        timeout: 180
        connect_timeout: 10
        command: "npx"
        args: ["-y", "analysis-server"]
        sampling:                    # server-initiated LLM requests
          enabled: true              # default: true
          model: "gemini-3-flash"    # override model (optional)
          max_tokens_cap: 4096       # max tokens per request
          timeout: 30                # LLM call timeout (seconds)
          max_rpm: 10                # max requests per minute
          allowed_models: []         # model whitelist (empty = all)
          max_tool_rounds: 5         # tool loop limit (0 = disable)
          log_level: "info"          # audit verbosity

Features:
    - Stdio transport (command + args) and HTTP/StreamableHTTP transport (url)
    - SSE transport (transport: sse) for MCP servers using the SSE protocol
    - Automatic reconnection with exponential backoff (up to 5 retries)
    - Environment variable filtering for stdio subprocesses (security)
    - Credential stripping in error messages returned to the LLM
    - Configurable per-server timeouts for tool calls and connections
    - Thread-safe architecture with dedicated background event loop
    - Sampling support: MCP servers can request LLM completions via
      sampling/createMessage (text and tool-use responses)
    - Parallel tool call opt-in: per-server ``supports_parallel_tool_calls``
      flag allows concurrent execution of tools from the same server

Architecture:
    A dedicated background event loop (_mcp_loop) runs in a daemon thread.
    Each MCP server runs as a long-lived asyncio Task on this loop, keeping
    its transport context alive. Tool call coroutines are scheduled onto the
    loop via ``run_coroutine_threadsafe()``.

    On shutdown, each server Task is signalled to exit its ``async with``
    block, ensuring the anyio cancel-scope cleanup happens in the *same*
    Task that opened the connection (required by anyio).

Thread safety:
    _servers and _mcp_loop/_mcp_thread are accessed from both the MCP
    background thread and caller threads.  All mutations are protected by
    _lock so the code is safe regardless of GIL presence (e.g. Python 3.13+
    free-threading).
"""

import asyncio
import contextvars
import concurrent.futures
import inspect
import json
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
from typing import Callable
from datetime import datetime
from typing import Any, Coroutine, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Upper bound for the OSV malware preflight during stdio MCP startup. The
# check makes a blocking urllib HTTPS call whose own timeout can fail to
# interrupt a stalled SSL handshake, which froze the asyncio event loop and
# blew past the gateway's 15s startup budget (#29184). We run it off the loop
# AND bound it here; the check is fail-open, so a timeout lets startup proceed.
# Set just ABOVE osv_check._TIMEOUT (10s) so the inner socket timeout fires
# first in the normal case; this outer bound only bites when a stalled SSL
# handshake defeats the inner timeout (the #29184 failure mode).
_OSV_MALWARE_CHECK_TIMEOUT_S = 12.0


# ---------------------------------------------------------------------------
# Stdio subprocess stderr redirection
# ---------------------------------------------------------------------------
#
# The MCP SDK's ``stdio_client(server, errlog=sys.stderr)`` defaults the
# subprocess stderr stream to the parent process's real stderr, i.e. the
# user's TTY.  That means any MCP server we spawn at startup (FastMCP
# banners, slack-mcp-server JSON startup logs, etc.) writes directly onto
# the terminal while prompt_toolkit / Rich is rendering the TUI — which
# corrupts the display and can hang the session.
#
# Instead we redirect every stdio MCP subprocess's stderr into a shared
# per-profile log file (~/.hermes/logs/mcp-stderr.log), tagged with the
# server name so individual servers remain debuggable.
#
# Fallback is os.devnull if opening the log file fails for any reason.

_mcp_stderr_log_fh: Optional[Any] = None
_mcp_stderr_log_lock = threading.Lock()


def _get_mcp_stderr_log() -> Any:
    """Return a shared append-mode file handle for MCP subprocess stderr.

    Opened once per process and reused for every stdio server.  Must have a
    real OS-level file descriptor (``fileno()``) because asyncio's subprocess
    machinery wires the child's stderr directly to that fd.  Falls back to
    ``/dev/null`` if opening the log file fails.
    """
    global _mcp_stderr_log_fh
    with _mcp_stderr_log_lock:
        if _mcp_stderr_log_fh is not None:
            return _mcp_stderr_log_fh
        try:
            from hermes_constants import get_hermes_home
            log_dir = get_hermes_home() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "mcp-stderr.log"
            # Line-buffered so server output lands on disk promptly; errors=
            # "replace" tolerates garbled binary output from misbehaving
            # servers.
            fh = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
            # Sanity-check: confirm a real fd is available before we commit.
            fh.fileno()
            _mcp_stderr_log_fh = fh
        except Exception as exc:  # pragma: no cover — best-effort fallback
            logger.debug("Failed to open MCP stderr log, using devnull: %s", exc)
            try:
                _mcp_stderr_log_fh = open(os.devnull, "w", encoding="utf-8")
            except Exception:
                # Last resort: the real stderr.  Not ideal for TUI users but
                # it matches pre-fix behavior.
                _mcp_stderr_log_fh = sys.stderr
        return _mcp_stderr_log_fh


def _write_stderr_log_header(server_name: str) -> None:
    """Write a human-readable session marker before launching a server.

    Gives operators a way to find each server's output in the shared
    ``mcp-stderr.log`` file without needing per-line prefixes (which would
    require a pipe + reader thread and complicate shutdown).
    """
    fh = _get_mcp_stderr_log()
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fh.write(f"\n===== [{ts}] starting MCP server '{server_name}' =====\n")
        fh.flush()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Graceful import -- MCP SDK is an optional dependency
# ---------------------------------------------------------------------------

_MCP_AVAILABLE = False
_MCP_HTTP_AVAILABLE = False
_MCP_SAMPLING_TYPES = False
_MCP_NOTIFICATION_TYPES = False
_MCP_ELICITATION_TYPES = False
_MCP_MESSAGE_HANDLER_SUPPORTED = False
# Conservative fallback for SDK builds that don't export LATEST_PROTOCOL_VERSION.
# Streamable HTTP was introduced by 2025-03-26, so this remains valid for the
# HTTP transport path even on older-but-supported SDK versions.
LATEST_PROTOCOL_VERSION = "2025-03-26"
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _MCP_AVAILABLE = True
    try:
        from mcp.client.streamable_http import streamablehttp_client
        _MCP_HTTP_AVAILABLE = True
    except ImportError:
        _MCP_HTTP_AVAILABLE = False
    # Prefer the non-deprecated API (mcp >= 1.24.0); fall back to the
    # deprecated wrapper for older SDK versions.
    try:
        from mcp.client.streamable_http import streamable_http_client
        _MCP_NEW_HTTP = True
    except ImportError:
        _MCP_NEW_HTTP = False
    try:
        from mcp.types import LATEST_PROTOCOL_VERSION
    except ImportError:
        logger.debug("mcp.types.LATEST_PROTOCOL_VERSION not available -- using fallback protocol version")
    # SSE transport client (for MCP servers using SSE transport instead of Streamable HTTP)
    try:
        from mcp.client.sse import sse_client
    except ImportError:
        sse_client = None
        logger.debug("mcp.client.sse.sse_client not available -- SSE transport disabled")
    # Sampling types -- separated so older SDK versions don't break MCP support
    try:
        from mcp.types import (
            CreateMessageResult,
            CreateMessageResultWithTools,
            ErrorData,
            SamplingCapability,
            SamplingToolsCapability,
            TextContent,
            ToolUseContent,
        )
        _MCP_SAMPLING_TYPES = True
    except ImportError:
        logger.debug("MCP sampling types not available -- sampling disabled")
    # Elicitation types -- gated separately for the same reason as sampling.
    # Added in mcp Python SDK 1.11.0 (Jul 2025); servers use elicitation to
    # ask the client for structured input mid-tool-call (e.g. payment
    # authorization). Missing types just disable the feature; everything
    # else keeps working.
    try:
        from mcp.types import ElicitRequestParams, ElicitResult
        _MCP_ELICITATION_TYPES = True
    except ImportError:
        logger.debug("MCP elicitation types not available -- elicitation disabled")
    # Notification types for dynamic tool discovery (tools/list_changed)
    try:
        from mcp.types import (
            ServerNotification,
            ToolListChangedNotification,
            PromptListChangedNotification,
            ResourceListChangedNotification,
        )
        _MCP_NOTIFICATION_TYPES = True
    except ImportError:
        logger.debug("MCP notification types not available -- dynamic tool discovery disabled")
except ImportError:
    logger.debug("mcp package not installed -- MCP tool support disabled")


def _check_message_handler_support() -> bool:
    """Check if ClientSession accepts ``message_handler`` kwarg.

    Inspects the constructor signature for backward compatibility with older
    MCP SDK versions that don't support notification handlers.
    """
    if not _MCP_AVAILABLE:
        return False
    try:
        return "message_handler" in inspect.signature(ClientSession).parameters
    except (TypeError, ValueError):
        return False


_MCP_MESSAGE_HANDLER_SUPPORTED = _check_message_handler_support()
if _MCP_AVAILABLE and not _MCP_MESSAGE_HANDLER_SUPPORTED:
    logger.debug("MCP SDK does not support message_handler -- dynamic tool discovery disabled")


def _check_logging_callback_support() -> bool:
    """Check if ClientSession accepts the ``logging_callback`` kwarg.

    Mirrors ``_check_message_handler_support`` for backward compatibility
    with older MCP SDK versions.  Without a logging_callback, the SDK's
    default handler silently discards every ``notifications/message`` a
    server emits, so server-side diagnostics never reach Hermes' logs.
    """
    if not _MCP_AVAILABLE:
        return False
    try:
        return "logging_callback" in inspect.signature(ClientSession).parameters
    except (TypeError, ValueError):
        return False


_MCP_LOGGING_CALLBACK_SUPPORTED = _check_logging_callback_support()

# MCP logging levels (RFC 5424 syslog severities) -> Python logging levels.
# Port of anomalyco/opencode#34529's serverLog mapping.
_MCP_LOG_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.ERROR,
    "alert": logging.ERROR,
    "emergency": logging.ERROR,
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TOOL_TIMEOUT = 300      # seconds for tool calls
_DEFAULT_CONNECT_TIMEOUT = 60    # seconds for initial connection per server
_MAX_RECONNECT_RETRIES = 5
_MAX_INITIAL_CONNECT_RETRIES = 3 # retries for the very first connection attempt
_MAX_BACKOFF_SECONDS = 60
# While parked (reconnect budget exhausted, tools deregistered) the run task
# wakes on this cadence and attempts one revival probe. Without it a parked
# server is unrevivable: its tools are out of the registry, so no tool call
# can ever reach the circuit-breaker half-open probe or _signal_reconnect.
_PARKED_RETRY_INTERVAL = 300     # seconds between parked self-probes

# Keepalive cadence for HTTP/SSE sessions. The MCP spec lets a server expire
# idle sessions on any TTL it chooses (Streamable HTTP "Session Management"),
# so a client that wants a session to survive idle periods MUST refresh faster
# than that TTL. The default suits long LB/NAT idle windows (commonly
# 300-600s); servers with short session TTLs (e.g. Unreal Engine's editor MCP,
# ~15s) need a smaller ``keepalive_interval`` in their config or every idle
# tool call lands on a dead session and pays the full reconnect path. The floor
# stops a misconfigured tiny interval from busy-looping the keepalive.
_DEFAULT_KEEPALIVE_INTERVAL = 180  # seconds between liveness pings
_MIN_KEEPALIVE_INTERVAL = 5        # clamp floor for configured intervals

# Environment variables that are safe to pass to stdio subprocesses
_SAFE_ENV_KEYS = frozenset({
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR",
})

_SAFE_ENV_KEYS_CASE_INSENSITIVE = frozenset({
    # Windows process/location vars. These are needed by launcher-style tools
    # such as Docker Desktop's MCP plugin discovery, and do not carry secrets.
    "ALLUSERSPROFILE",
    "APPDATA",
    "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)",
    "COMMONPROGRAMW6432",
    "COMPUTERNAME",
    "COMSPEC",
    "HOMEDRIVE",
    "HOMEPATH",
    "LOCALAPPDATA",
    "NUMBER_OF_PROCESSORS",
    "OS",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "PUBLIC",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERDOMAIN",
    "USERNAME",
    "USERPROFILE",
    "WINDIR",
})

# Regex for credential patterns to strip from error messages
_CREDENTIAL_PATTERN = re.compile(
    r"(?:"
    r"ghp_[A-Za-z0-9_]{1,255}"           # GitHub PAT
    r"|sk-[A-Za-z0-9_]{1,255}"           # OpenAI-style key
    r"|Bearer\s+\S+"                      # Bearer token
    r"|token=[^\s&,;\"']{1,255}"         # token=...
    r"|key=[^\s&,;\"']{1,255}"           # key=...
    r"|API_KEY=[^\s&,;\"']{1,255}"       # API_KEY=...
    r"|password=[^\s&,;\"']{1,255}"      # password=...
    r"|secret=[^\s&,;\"']{1,255}"        # secret=...
    r")",
    re.IGNORECASE,
)

# Pre-compiled pattern for ${VAR_NAME} style env-var interpolation.
# Supports any non-} characters in the variable name (hyphens, dots, etc.)
# so providers like MY-VAR or my.var work correctly.
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _env_ref_name(ref: str) -> str:
    """Normalize a ``${...}`` reference body into an env-var name.

    Accepts Cursor-style ``${env:VAR}`` in addition to plain ``${VAR}`` by
    stripping a leading ``env:`` prefix. The result is the bare variable name
    to look up in the secret scope / ``os.environ``.
    """
    ref = ref.strip()
    if ref.startswith("env:"):
        ref = ref[len("env:"):].strip()
    return ref


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _build_safe_env(user_env: Optional[dict]) -> dict:
    """Build a filtered environment dict for stdio subprocesses.

    Only passes through safe baseline variables (PATH, HOME, etc.) and XDG_*
    variables from the current process environment, plus any variables
    explicitly specified by the user in the server config.

    This prevents accidentally leaking secrets like API keys, tokens, or
    credentials to MCP server subprocesses.
    """
    env = {}
    for key, value in os.environ.items():
        if (
            key in _SAFE_ENV_KEYS
            or key.upper() in _SAFE_ENV_KEYS_CASE_INSENSITIVE
            or key.startswith("XDG_")
        ):
            env[key] = value
    if user_env:
        env.update(user_env)
    return env


def _sanitize_error(text: str) -> str:
    """Strip credential-like patterns from error text before returning to LLM.

    Replaces tokens, keys, and other secrets with [REDACTED] to prevent
    accidental credential exposure in tool error responses.
    """
    return _CREDENTIAL_PATTERN.sub("[REDACTED]", text)


def _exc_str(exc: BaseException) -> str:
    """Return a non-empty human-readable string for *exc*.

    Some exception classes (e.g. ``anyio.ClosedResourceError``) are raised
    without a message argument, so ``str(exc)`` is ``""``.  This helper
    falls back to ``repr(exc)`` so that error messages shown to the user
    and logged to disk always carry *some* diagnostic information.
    """
    text = str(exc).strip()
    return text if text else repr(exc)


# JSON-RPC "method not found" — the error a server returns when it does not
# implement a requested method (e.g. a tool-capable server that never wired up
# the optional ``ping`` utility). Defined locally with a fallback so detection
# works even on SDK builds that don't export the constant.
try:
    from mcp.types import METHOD_NOT_FOUND as _JSONRPC_METHOD_NOT_FOUND
except Exception:  # pragma: no cover — older/newer SDK without the constant
    _JSONRPC_METHOD_NOT_FOUND = -32601


def _is_method_not_found_error(exc: BaseException) -> bool:
    """Return True if *exc* is a JSON-RPC ``method not found`` (-32601).

    ``ping`` is an *optional* MCP utility (spec: "optional ping mechanism").
    A server that doesn't implement it answers a ping with -32601 rather than
    an empty result. Structurally inspect ``McpError.error.code`` first, then
    fall back to a substring match so detection survives SDK version drift and
    servers that surface the condition as a plain message.

    The substring fallback matters when a server reports method-not-found
    without a structural ``-32601`` code (e.g. surfaced as a plain exception
    string). Besides the canonical "method not found", many JSON-RPC
    implementations phrase it as "Unknown method: <name>" — agentmemory's MCP
    server is one such case (#50028). Without matching that phrasing the
    ping→list_tools fallback never latches and the keepalive reconnect-loops.
    """
    # Structural: mcp.shared.exceptions.McpError carries ErrorData.code.
    err = getattr(exc, "error", None)
    code = getattr(err, "code", None)
    if code == _JSONRPC_METHOD_NOT_FOUND:
        return True
    msg = str(exc).lower()
    if not msg:
        return False
    return (
        str(_JSONRPC_METHOD_NOT_FOUND) in msg
        or "method not found" in msg
        or "unknown method" in msg
        or "not found: ping" in msg
    )


# ---------------------------------------------------------------------------
# MCP tool description content scanning
# ---------------------------------------------------------------------------

# Patterns that indicate potential prompt injection in MCP tool descriptions.
# These are WARNING-level — we log but don't block, since false positives
# would break legitimate MCP servers.
_MCP_INJECTION_PATTERNS = [
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
     "prompt override attempt ('ignore previous instructions')"),
    (re.compile(r"you\s+are\s+now\s+a", re.I),
     "identity override attempt ('you are now a...')"),
    (re.compile(r"your\s+new\s+(task|role|instructions?)\s+(is|are)", re.I),
     "task override attempt"),
    (re.compile(r"system\s*:\s*", re.I),
     "system prompt injection attempt"),
    (re.compile(r"<\s*(system|human|assistant)\s*>", re.I),
     "role tag injection attempt"),
    (re.compile(r"do\s+not\s+(tell|inform|mention|reveal)", re.I),
     "concealment instruction"),
    (re.compile(r"(curl|wget|fetch)\s+https?://", re.I),
     "network command in description"),
    (re.compile(r"base64\.(b64decode|decodebytes)", re.I),
     "base64 decode reference"),
    (re.compile(r"exec\s*\(|eval\s*\(", re.I),
     "code execution reference"),
    (re.compile(r"import\s+(subprocess|os|shutil|socket)", re.I),
     "dangerous import reference"),
]


def _scan_mcp_description(server_name: str, tool_name: str, description: str) -> List[str]:
    """Scan an MCP tool description for prompt injection patterns.

    Returns a list of finding strings (empty = clean).
    """
    findings = []
    if not description:
        return findings
    for pattern, reason in _MCP_INJECTION_PATTERNS:
        if pattern.search(description):
            findings.append(reason)
    if findings:
        logger.warning(
            "MCP server '%s' tool '%s': suspicious description content — %s. "
            "Description: %.200s",
            server_name, tool_name, "; ".join(findings),
            description,
        )
    return findings


def _prepend_path(env: dict, directory: str) -> dict:
    """Prepend *directory* to env PATH if it is not already present."""
    updated = dict(env or {})
    if not directory:
        return updated

    existing = updated.get("PATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if directory not in parts:
        parts = [directory, *parts]
    updated["PATH"] = os.pathsep.join(parts) if parts else directory
    return updated


def _resolve_stdio_command(command: str, env: dict) -> tuple[str, dict]:
    """Resolve a stdio MCP command against the exact subprocess environment.

    This primarily exists to make bare ``npx``/``npm``/``node`` commands work
    reliably even when MCP subprocesses run under a filtered PATH.
    """
    resolved_command = os.path.expanduser(str(command).strip())
    resolved_env = dict(env or {})

    if os.sep not in resolved_command:
        path_arg = resolved_env["PATH"] if "PATH" in resolved_env else None
        which_hit = shutil.which(resolved_command, path=path_arg)
        if which_hit:
            resolved_command = which_hit
        elif resolved_command in {"npx", "npm", "node"}:
            hermes_home = os.path.expanduser(
                os.getenv(
                    "HERMES_HOME", os.path.join(os.path.expanduser("~"), ".hermes")
                )
            )
            candidates = [
                os.path.join(hermes_home, "node", "bin", resolved_command),
                os.path.join(os.path.expanduser("~"), ".local", "bin", resolved_command),
                # /usr/local/bin is the canonical install location for Node on
                # Linux from-source builds, the upstream node:bookworm-slim
                # image (which the Hermes Docker image copies node + npm +
                # corepack from since #4977), and macOS Homebrew on Intel.
                # Without this candidate, any MCP server configured with an
                # env.PATH that omits /usr/local/bin (a common pattern when
                # users hand-author PATH for sandboxing) fails with ENOENT
                # at execvp, and a naive symlink workaround into the user's
                # PATH only fails one layer deeper because npx's shebang
                # re-execs /usr/bin/env node which needs the same directory.
                os.path.join(os.sep, "usr", "local", "bin", resolved_command),
            ]
            for candidate in candidates:
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    resolved_command = candidate
                    break

    command_dir = os.path.dirname(resolved_command)
    if command_dir:
        resolved_env = _prepend_path(resolved_env, command_dir)

    return resolved_command, resolved_env


# ---------------------------------------------------------------------------
# MCP ImageContent block → Hermes MEDIA tag
# ---------------------------------------------------------------------------


def _mcp_image_extension_for_mime_type(mime_type: str) -> str:
    """Return a reasonable file extension for an MCP image MIME type."""
    import mimetypes
    normalized = (mime_type or "").split(";", 1)[0].strip().lower()
    if normalized in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    return mimetypes.guess_extension(normalized) or ".png"


def _cache_mcp_image_block(block) -> str:
    """Cache an MCP ``ImageContent`` block to the shared image cache and
    return a ``MEDIA:<path>`` tag that Hermes gateways know how to render.

    Returns an empty string when *block* is not an image, when the base64
    payload is malformed, or when the cache helper rejects the bytes (e.g.
    non-image MIME masquerading as an image). Errors are logged, not raised:
    a single bad block shouldn't kill the tool result, and the caller will
    fall through to any text blocks that did parse.
    """
    import base64

    data = getattr(block, "data", None)
    mime_type = getattr(block, "mimeType", None)
    normalized_mime = str(mime_type or "").split(";", 1)[0].strip().lower()
    if data is None or not normalized_mime.startswith("image/"):
        return ""

    try:
        raw_bytes = base64.b64decode(data)
    except (TypeError, ValueError) as exc:
        logger.warning("MCP image block decode failed (%s): %s", normalized_mime, exc)
        return ""

    try:
        from gateway.platforms.base import cache_image_from_bytes

        image_path = cache_image_from_bytes(
            raw_bytes,
            ext=_mcp_image_extension_for_mime_type(normalized_mime),
        )
    except ImportError:
        # gateway.platforms.base not importable in this process (e.g. cron
        # without gateway deps). Fall back to silently dropping — callers
        # get any text blocks that did parse.
        logger.debug("MCP image caching skipped — gateway.platforms.base unavailable")
        return ""
    except Exception as exc:
        logger.warning("MCP image block cache failed: %s", exc)
        return ""

    return f"MEDIA:{image_path}"


# ---------------------------------------------------------------------------
# Remote MCP URL validation
# ---------------------------------------------------------------------------


class InvalidMcpUrlError(ValueError):
    """Raised when a remote MCP server's ``url`` cannot be parsed as http(s)://.

    Validated once at startup so we fail fast with a clear message instead of
    burning through the reconnect-backoff loop on every attempt.  (Ported from
    anomalyco/opencode#25019.)
    """


class NonMcpEndpointError(ConnectionError):
    """Raised when an HTTP MCP URL serves a non-MCP response.

    A genuine MCP Streamable-HTTP endpoint answers with ``application/json``
    or ``text/event-stream``.  Anything else on a 2xx response (typically
    ``text/html`` from a web-app root) means the configured ``url`` points at
    the wrong place.  This is non-retryable: every attempt returns the same
    page, so the reconnect-backoff loop is skipped and the server is reported
    failed immediately with an actionable message.

    Subclasses :class:`ConnectionError` so callers that only catch the broad
    class still treat it as a connection problem.
    """


def _validate_remote_mcp_url(server_name: str, url: Any) -> str:
    """Return the URL as a string if it's a valid http(s) remote MCP URL.

    Raises :class:`InvalidMcpUrlError` otherwise with a message naming the
    offending server, so users can spot the bad entry in their config.

    Accepts:
    - ``http://host`` / ``https://host`` with optional port, path, query
    - IPv4, IPv6 (bracketed), DNS hostnames

    Rejects:
    - Non-string values (``None``, dicts, ints)
    - Missing scheme (``example.com/mcp``)
    - Non-http(s) schemes (``file://``, ``ws://``, ``stdio:`` — stdio servers
      use the ``command`` key, not ``url``)
    - Empty host (``http://``, ``https:///path``)
    """
    if not isinstance(url, str):
        raise InvalidMcpUrlError(
            f"Invalid MCP URL for '{server_name}': expected a string, got "
            f"{type(url).__name__}"
        )
    stripped = url.strip()
    if not stripped:
        raise InvalidMcpUrlError(
            f"Invalid MCP URL for '{server_name}': empty url"
        )
    try:
        parsed = urlparse(stripped)
    except Exception as exc:  # urlparse is very permissive — belt and braces
        raise InvalidMcpUrlError(
            f"Invalid MCP URL for '{server_name}': {stripped!r} ({exc})"
        ) from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise InvalidMcpUrlError(
            f"Invalid MCP URL for '{server_name}': scheme must be http or "
            f"https, got {parsed.scheme!r} ({stripped!r})"
        )
    if not parsed.netloc:
        raise InvalidMcpUrlError(
            f"Invalid MCP URL for '{server_name}': missing host ({stripped!r})"
        )
    # ``urlparse`` accepts ``http://:8080`` (empty host, explicit port).
    # Reject that — we need a real host.
    if not parsed.hostname:
        raise InvalidMcpUrlError(
            f"Invalid MCP URL for '{server_name}': missing hostname "
            f"({stripped!r})"
        )
    return stripped


def _resolve_client_cert(server_name: str, config: dict):
    """Resolve the ``client_cert`` / ``client_key`` config for mTLS.

    Returns whatever ``httpx``'s ``cert=`` parameter accepts, or ``None`` when
    no client certificate is configured:

      - ``None`` if neither ``client_cert`` nor ``client_key`` is set.
      - A single absolute path string if ``client_cert`` is a string and
        ``client_key`` is unset (PEM file with cert + key combined).
      - A ``(cert_path, key_path)`` tuple when both are set, or when
        ``client_cert`` is a 2-element list/tuple.
      - A ``(cert_path, key_path, password)`` tuple when ``client_cert`` is
        a 3-element list/tuple — the third element is the key passphrase.

    User paths support ``~`` expansion. Missing files raise ``FileNotFoundError``
    with a server-scoped message so the failure surfaces as a clear setup
    error rather than an opaque TLS handshake error.
    """
    raw_cert = config.get("client_cert")
    raw_key = config.get("client_key")

    if raw_cert is None and raw_key is None:
        return None

    def _expand(path: Any, label: str) -> str:
        if not isinstance(path, str) or not path.strip():
            raise ValueError(
                f"MCP server '{server_name}': {label} must be a non-empty "
                f"string path (got {type(path).__name__})"
            )
        expanded = os.path.expanduser(path.strip())
        if not os.path.isfile(expanded):
            raise FileNotFoundError(
                f"MCP server '{server_name}': {label} not found at "
                f"{expanded!r}"
            )
        return expanded

    # Tuple/list form for client_cert — (cert, key) or (cert, key, password).
    if isinstance(raw_cert, (list, tuple)):
        if raw_key is not None:
            raise ValueError(
                f"MCP server '{server_name}': specify either client_cert as "
                f"a list [cert, key] OR client_cert + client_key, not both"
            )
        if len(raw_cert) == 2:
            cert_path = _expand(raw_cert[0], "client_cert[0]")
            key_path = _expand(raw_cert[1], "client_cert[1]")
            return (cert_path, key_path)
        if len(raw_cert) == 3:
            cert_path = _expand(raw_cert[0], "client_cert[0]")
            key_path = _expand(raw_cert[1], "client_cert[1]")
            password = raw_cert[2]
            if not isinstance(password, str):
                raise ValueError(
                    f"MCP server '{server_name}': client_cert[2] (key "
                    f"passphrase) must be a string"
                )
            return (cert_path, key_path, password)
        raise ValueError(
            f"MCP server '{server_name}': client_cert list form must have 2 "
            f"or 3 elements (got {len(raw_cert)})"
        )

    # String form for client_cert.
    cert_path = _expand(raw_cert, "client_cert")
    if raw_key is not None:
        key_path = _expand(raw_key, "client_key")
        return (cert_path, key_path)
    # Single combined PEM file (cert + key in one file).
    return cert_path


def _format_connect_error(exc: BaseException) -> str:
    """Render nested MCP connection errors into an actionable short message."""

    def _find_missing(current: BaseException) -> Optional[str]:
        nested = getattr(current, "exceptions", None)
        if nested:
            for child in nested:
                missing = _find_missing(child)
                if missing:
                    return missing
            return None
        if isinstance(current, FileNotFoundError):
            if getattr(current, "filename", None):
                return str(current.filename)
            match = re.search(r"No such file or directory: '([^']+)'", str(current))
            if match:
                return match.group(1)
        for attr in ("__cause__", "__context__"):
            nested_exc = getattr(current, attr, None)
            if isinstance(nested_exc, BaseException):
                missing = _find_missing(nested_exc)
                if missing:
                    return missing
        return None

    def _flatten_messages(current: BaseException) -> List[str]:
        nested = getattr(current, "exceptions", None)
        if nested:
            flattened: List[str] = []
            for child in nested:
                flattened.extend(_flatten_messages(child))
            return flattened
        messages = []
        text = str(current).strip()
        if text:
            messages.append(text)
        for attr in ("__cause__", "__context__"):
            nested_exc = getattr(current, attr, None)
            if isinstance(nested_exc, BaseException):
                messages.extend(_flatten_messages(nested_exc))
        return messages or [current.__class__.__name__]

    missing = _find_missing(exc)
    if missing:
        message = f"missing executable '{missing}'"
        if os.path.basename(missing) in {"npx", "npm", "node"}:
            message += (
                " (ensure Node.js is installed and PATH includes its bin directory, "
                "or set mcp_servers.<name>.command to an absolute path and include "
                "that directory in mcp_servers.<name>.env.PATH)"
            )
        return _sanitize_error(message)

    deduped: List[str] = []
    for item in _flatten_messages(exc):
        if item not in deduped:
            deduped.append(item)
    return _sanitize_error("; ".join(deduped[:3]))


# ---------------------------------------------------------------------------
# Sampling -- server-initiated LLM requests (MCP sampling/createMessage)
# ---------------------------------------------------------------------------

def _safe_numeric(value, default, coerce=int, minimum=1):
    """Coerce a config value to a numeric type, returning *default* on failure.

    Handles string values from YAML (e.g. ``"10"`` instead of ``10``),
    non-finite floats, and values below *minimum*.
    """
    try:
        result = coerce(value)
        if isinstance(result, float) and not math.isfinite(result):
            return default
        return max(result, minimum)
    except (TypeError, ValueError, OverflowError):
        return default


class SamplingHandler:
    """Handles sampling/createMessage requests for a single MCP server.

    Each MCPServerTask that has sampling enabled creates one SamplingHandler.
    The handler is callable and passed directly to ``ClientSession`` as
    the ``sampling_callback``.  All state (rate-limit timestamps, metrics,
    tool-loop counters) lives on the instance -- no module-level globals.

    The callback is async and runs on the MCP background event loop.  The
    sync LLM call is offloaded to a thread via ``asyncio.to_thread()`` so
    it doesn't block the event loop.
    """

    _STOP_REASON_MAP = {"stop": "endTurn", "length": "maxTokens", "tool_calls": "toolUse"}

    def __init__(self, server_name: str, config: dict):
        self.server_name = server_name
        self.max_rpm = _safe_numeric(config.get("max_rpm", 10), 10, int)
        self.timeout = _safe_numeric(config.get("timeout", 30), 30, float)
        self.max_tokens_cap = _safe_numeric(config.get("max_tokens_cap", 4096), 4096, int)
        self.max_tool_rounds = _safe_numeric(
            config.get("max_tool_rounds", 5), 5, int, minimum=0,
        )
        self.model_override = config.get("model")
        self.allowed_models = config.get("allowed_models", [])

        _log_levels = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}
        self.audit_level = _log_levels.get(
            str(config.get("log_level", "info")).lower(), logging.INFO,
        )

        # Per-instance state
        self._rate_timestamps: List[float] = []
        self._tool_loop_count = 0
        self.metrics = {"requests": 0, "errors": 0, "tokens_used": 0, "tool_use_count": 0}

    # -- Rate limiting -------------------------------------------------------

    def _check_rate_limit(self) -> bool:
        """Sliding-window rate limiter.  Returns True if request is allowed."""
        now = time.time()
        window = now - 60
        self._rate_timestamps[:] = [t for t in self._rate_timestamps if t > window]
        if len(self._rate_timestamps) >= self.max_rpm:
            return False
        self._rate_timestamps.append(now)
        return True

    # -- Model resolution ----------------------------------------------------

    def _resolve_model(self, preferences) -> Optional[str]:
        """Config override > server hint > None (use default)."""
        if self.model_override:
            return self.model_override
        if preferences and hasattr(preferences, "hints") and preferences.hints:
            for hint in preferences.hints:
                if hasattr(hint, "name") and hint.name:
                    return hint.name
        return None

    # -- Message conversion --------------------------------------------------

    @staticmethod
    def _extract_tool_result_text(block) -> str:
        """Extract text from a ToolResultContent block."""
        if not hasattr(block, "content") or block.content is None:
            return ""
        items = block.content if isinstance(block.content, list) else [block.content]
        return "\n".join(item.text for item in items if hasattr(item, "text"))

    def _convert_messages(self, params) -> List[dict]:
        """Convert MCP SamplingMessages to OpenAI format.

        Uses ``msg.content_as_list`` (SDK helper) so single-block and
        list-of-blocks are handled uniformly.  Dispatches per block type
        with ``isinstance`` on real SDK types when available, falling back
        to duck-typing via ``hasattr`` for compatibility.
        """
        messages: List[dict] = []
        for msg in params.messages:
            blocks = msg.content_as_list if hasattr(msg, "content_as_list") else (
                msg.content if isinstance(msg.content, list) else [msg.content]
            )

            # Separate blocks by kind
            tool_results = [b for b in blocks if hasattr(b, "toolUseId")]
            tool_uses = [b for b in blocks if hasattr(b, "name") and hasattr(b, "input") and not hasattr(b, "toolUseId")]
            content_blocks = [b for b in blocks if not hasattr(b, "toolUseId") and not (hasattr(b, "name") and hasattr(b, "input"))]

            # Emit tool result messages (role: tool)
            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr.toolUseId,
                    "content": self._extract_tool_result_text(tr),
                })

            # Emit assistant tool_calls message
            if tool_uses:
                tc_list = []
                for tu in tool_uses:
                    tc_list.append({
                        "id": getattr(tu, "id", f"call_{len(tc_list)}"),
                        "type": "function",
                        "function": {
                            "name": tu.name,
                            "arguments": json.dumps(tu.input, ensure_ascii=False) if isinstance(tu.input, dict) else str(tu.input),
                        },
                    })
                msg_dict: dict = {"role": msg.role, "tool_calls": tc_list}
                # Include any accompanying text
                text_parts = [b.text for b in content_blocks if hasattr(b, "text")]
                if text_parts:
                    msg_dict["content"] = "\n".join(text_parts)
                messages.append(msg_dict)
            elif content_blocks:
                # Pure text/image content
                if len(content_blocks) == 1 and hasattr(content_blocks[0], "text"):
                    messages.append({"role": msg.role, "content": content_blocks[0].text})
                else:
                    parts = []
                    for block in content_blocks:
                        if hasattr(block, "text"):
                            parts.append({"type": "text", "text": block.text})
                        elif hasattr(block, "data") and hasattr(block, "mimeType"):
                            parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
                            })
                        else:
                            logger.warning(
                                "Unsupported sampling content block type: %s (skipped)",
                                type(block).__name__,
                            )
                    if parts:
                        messages.append({"role": msg.role, "content": parts})

        return messages

    # -- Error helper --------------------------------------------------------

    @staticmethod
    def _error(message: str, code: int = -1):
        """Return ErrorData (MCP spec) or raise as fallback."""
        if _MCP_SAMPLING_TYPES:
            return ErrorData(code=code, message=message)
        raise Exception(message)

    # -- Response building ---------------------------------------------------

    def _build_tool_use_result(self, choice, response):
        """Build a CreateMessageResultWithTools from an LLM tool_calls response."""
        self.metrics["tool_use_count"] += 1

        # Tool loop governance
        if self.max_tool_rounds == 0:
            self._tool_loop_count = 0
            return self._error(
                f"Tool loops disabled for server '{self.server_name}' (max_tool_rounds=0)"
            )

        self._tool_loop_count += 1
        if self._tool_loop_count > self.max_tool_rounds:
            self._tool_loop_count = 0
            return self._error(
                f"Tool loop limit exceeded for server '{self.server_name}' "
                f"(max {self.max_tool_rounds} rounds)"
            )

        content_blocks = []
        for tc in choice.message.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(
                        "MCP server '%s': malformed tool_calls arguments "
                        "from LLM (wrapping as raw): %.100s",
                        self.server_name, args,
                    )
                    parsed = {"_raw": args}
            else:
                parsed = args if isinstance(args, dict) else {"_raw": str(args)}

            content_blocks.append(ToolUseContent(
                type="tool_use",
                id=tc.id,
                name=tc.function.name,
                input=parsed,
            ))

        logger.log(
            self.audit_level,
            "MCP server '%s' sampling response: model=%s, tokens=%s, tool_calls=%d",
            self.server_name, response.model,
            getattr(getattr(response, "usage", None), "total_tokens", "?"),
            len(content_blocks),
        )

        return CreateMessageResultWithTools(
            role="assistant",
            content=content_blocks,
            model=response.model,
            stopReason="toolUse",
        )

    def _build_text_result(self, choice, response):
        """Build a CreateMessageResult from a normal text response."""
        self._tool_loop_count = 0  # reset on text response
        response_text = choice.message.content or ""

        logger.log(
            self.audit_level,
            "MCP server '%s' sampling response: model=%s, tokens=%s",
            self.server_name, response.model,
            getattr(getattr(response, "usage", None), "total_tokens", "?"),
        )

        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=_sanitize_error(response_text)),
            model=response.model,
            stopReason=self._STOP_REASON_MAP.get(choice.finish_reason, "endTurn"),
        )

    # -- Session kwargs helper -----------------------------------------------

    def session_kwargs(self) -> dict:
        """Return kwargs to pass to ClientSession for sampling support."""
        return {
            "sampling_callback": self,
            "sampling_capabilities": SamplingCapability(
                tools=SamplingToolsCapability(),
            ),
        }

    # -- Main callback -------------------------------------------------------

    async def __call__(self, context, params):
        """Sampling callback invoked by the MCP SDK.

        Conforms to ``SamplingFnT`` protocol.  Returns
        ``CreateMessageResult``, ``CreateMessageResultWithTools``, or
        ``ErrorData``.
        """
        # Rate limit
        if not self._check_rate_limit():
            logger.warning(
                "MCP server '%s' sampling rate limit exceeded (%d/min)",
                self.server_name, self.max_rpm,
            )
            self.metrics["errors"] += 1
            return self._error(
                f"Sampling rate limit exceeded for server '{self.server_name}' "
                f"({self.max_rpm} requests/minute)"
            )

        # Resolve model
        model = self._resolve_model(getattr(params, "modelPreferences", None))

        # Get auxiliary LLM client via centralized router
        from agent.auxiliary_client import call_llm

        # Model whitelist check (we need to resolve model before calling)
        resolved_model = model or self.model_override or ""

        if self.allowed_models and resolved_model and resolved_model not in self.allowed_models:
            logger.warning(
                "MCP server '%s' requested model '%s' not in allowed_models",
                self.server_name, resolved_model,
            )
            self.metrics["errors"] += 1
            return self._error(
                f"Model '{resolved_model}' not allowed for server "
                f"'{self.server_name}'. Allowed: {', '.join(self.allowed_models)}"
            )

        # Convert messages
        messages = self._convert_messages(params)
        if hasattr(params, "systemPrompt") and params.systemPrompt:
            messages.insert(0, {"role": "system", "content": params.systemPrompt})

        # Build LLM call kwargs
        max_tokens = min(params.maxTokens, self.max_tokens_cap)
        call_temperature = None
        if hasattr(params, "temperature") and params.temperature is not None:
            call_temperature = params.temperature

        # Forward server-provided tools
        call_tools = None
        server_tools = getattr(params, "tools", None)
        if server_tools:
            call_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": getattr(t, "name", ""),
                        "description": getattr(t, "description", "") or "",
                        "parameters": _normalize_mcp_input_schema(
                            getattr(t, "inputSchema", None)
                        ),
                    },
                }
                for t in server_tools
            ]

        logger.log(
            self.audit_level,
            "MCP server '%s' sampling request: model=%s, max_tokens=%d, messages=%d",
            self.server_name, resolved_model, max_tokens, len(messages),
        )

        # Offload sync LLM call to thread (non-blocking)
        def _sync_call():
            return call_llm(
                task="mcp",
                model=resolved_model or None,
                messages=messages,
                temperature=call_temperature,
                max_tokens=max_tokens,
                tools=call_tools,
                timeout=self.timeout,
            )

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_sync_call), timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            self.metrics["errors"] += 1
            return self._error(
                f"Sampling LLM call timed out after {self.timeout}s "
                f"for server '{self.server_name}'"
            )
        except Exception as exc:
            self.metrics["errors"] += 1
            return self._error(
                f"Sampling LLM call failed: {_sanitize_error(_exc_str(exc))}"
            )

        # Guard against empty choices (content filtering, provider errors)
        if not getattr(response, "choices", None):
            self.metrics["errors"] += 1
            return self._error(
                f"LLM returned empty response (no choices) for server "
                f"'{self.server_name}'"
            )

        # Track metrics
        choice = response.choices[0]
        self.metrics["requests"] += 1
        total_tokens = getattr(getattr(response, "usage", None), "total_tokens", 0)
        if isinstance(total_tokens, int):
            self.metrics["tokens_used"] += total_tokens

        # Dispatch based on response type
        if (
            choice.finish_reason == "tool_calls"
            and hasattr(choice.message, "tool_calls")
            and choice.message.tool_calls
        ):
            return self._build_tool_use_result(choice, response)

        return self._build_text_result(choice, response)


# ---------------------------------------------------------------------------
# Elicitation handler
# ---------------------------------------------------------------------------

def _format_elicitation_schema_summary(schema: dict, server_name: str) -> str:
    """Render a JSON-schema-ish requested_schema to a human-readable field list.

    Elicitation schemas are restricted to a flat object with named top-level
    properties. We surface field names, types, and descriptions so the user
    can tell what the server is asking for before approving.
    """
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict) or not props:
        return f"Approval requested by MCP server '{server_name}'."

    lines = [f"Fields requested by MCP server '{server_name}':"]
    for field_name, field_spec in props.items():
        field_type = ""
        field_desc = ""
        if isinstance(field_spec, dict):
            field_type = str(field_spec.get("type", "") or "")
            field_desc = str(field_spec.get("description", "") or "")
        suffix = f" ({field_type})" if field_type else ""
        if field_desc:
            lines.append(f"  - {field_name}{suffix}: {field_desc}")
        else:
            lines.append(f"  - {field_name}{suffix}")
    return "\n".join(lines)


class ElicitationHandler:
    """Handles ``elicitation/create`` requests for a single MCP server.

    Each ``MCPServerTask`` that has elicitation enabled creates one handler.
    The handler is callable and passed directly to ``ClientSession`` as the
    ``elicitation_callback`` (added in mcp Python SDK 1.11.0).

    Elicitation lets a server ask the client to collect structured input from
    the user mid-tool-call (e.g. payment authorization, OAuth confirmation).
    Form-mode elicitations are routed through Hermes' existing approval
    system (``tools.approval.prompt_dangerous_approval``), which surfaces
    the prompt on whichever surface the active session uses -- CLI, TUI,
    Telegram, Slack, etc. URL-mode elicitations are declined as unsupported.

    Failure modes are fail-closed: any timeout, exception, or unexpected
    state returns ``decline``/``cancel`` rather than silently accepting.
    The server treats this as the user not approving.
    """

    # Outer cap for the approval await. ``prompt_dangerous_approval`` runs
    # its own input() timeout via the approval-config value; this is an
    # asyncio-side safety net so the MCP event loop never blocks
    # indefinitely if the inner timeout machinery is bypassed.
    _OUTER_TIMEOUT_GRACE_SECONDS = 5

    def __init__(self, server_name: str, config: dict, owner: Optional["MCPServerTask"] = None):
        self.server_name = server_name
        # Per-elicitation timeout. Default 5 min mirrors the gateway approval
        # default so users on async surfaces (Telegram, Slack) have time to
        # respond before the server gives up.
        self.timeout = _safe_numeric(config.get("timeout", 300), 300, float)
        # Back-reference to the MCPServerTask so we can read the agent's
        # captured contextvars snapshot at elicitation time. Optional so
        # the handler stays unit-testable in isolation.
        self.owner = owner
        self.metrics = {
            "requests": 0,
            "accepted": 0,
            "declined": 0,
            "errors": 0,
        }

    def session_kwargs(self) -> dict:
        """Return kwargs to pass to ClientSession for elicitation support."""
        return {"elicitation_callback": self}

    async def __call__(self, context, params):
        """Elicitation callback invoked by the MCP SDK.

        Conforms to ``ElicitationFnT`` protocol. Returns ``ElicitResult``
        or ``ErrorData``.
        """
        self.metrics["requests"] += 1

        # URL-mode elicitations point the user to an external URL for
        # sensitive out-of-band flows (OAuth, payment processing). Honouring
        # them requires opening a browser to that URL and waiting for the
        # server's notifications/elicitation/complete -- out of scope for
        # the initial implementation. Decline cleanly so the server does
        # not hang.
        mode = getattr(params, "mode", "form")
        if mode == "url":
            logger.info(
                "MCP server '%s' requested URL-mode elicitation; "
                "declining (URL-mode elicitation not implemented)",
                self.server_name,
            )
            self.metrics["declined"] += 1
            return ElicitResult(action="decline")

        message = getattr(params, "message", "") or (
            f"MCP server '{self.server_name}' is requesting your approval"
        )
        schema = getattr(params, "requested_schema", {}) or {}
        description = _format_elicitation_schema_summary(schema, self.server_name)

        logger.info(
            "MCP server '%s' elicitation request: %s",
            self.server_name, _sanitize_error(message)[:200],
        )

        # Lazy import: tools.approval is imported very early during process
        # bootstrap; matching the lazy pattern used by _fire_approval_hook
        # avoids any chance of import-order coupling.
        try:
            from tools.approval import request_elicitation_consent
        except Exception as exc:  # pragma: no cover -- defensive
            logger.error(
                "MCP server '%s' elicitation: approval system unavailable: %s",
                self.server_name, exc,
            )
            self.metrics["errors"] += 1
            return ElicitResult(action="decline")

        # Offload the sync consent flow to a worker thread. Running it
        # inline would freeze the MCP background event loop, blocking every
        # other RPC on this session. request_elicitation_consent() routes
        # itself to the right surface (gateway notify_cb for Telegram /
        # Slack / etc., prompt_dangerous_approval for CLI / TUI) and
        # normalizes the answer to one of accept / decline / cancel.
        #
        # The recv-loop task that fires this callback does NOT inherit
        # the agent's contextvars (HERMES_SESSION_PLATFORM etc.). When
        # the MCP tool wrapper captured the agent's context onto
        # owner._pending_call_context we replay it here via
        # contextvars.Context.run so the gateway-platform detection in
        # request_elicitation_consent picks up the right session.
        captured = getattr(self.owner, "_pending_call_context", None) if self.owner else None

        def _invoke_consent() -> str:
            if captured is None:
                return request_elicitation_consent(
                    message,
                    description,
                    timeout_seconds=int(self.timeout),
                    surface=f"mcp-elicitation/{self.server_name}",
                )
            # Context.run can only execute a context once — copy to allow
            # multiple elicitations within a single tool call.
            return captured.copy().run(
                request_elicitation_consent,
                message,
                description,
                timeout_seconds=int(self.timeout),
                surface=f"mcp-elicitation/{self.server_name}",
            )

        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(_invoke_consent),
                timeout=self.timeout + self._OUTER_TIMEOUT_GRACE_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "MCP server '%s' elicitation timed out after %ds",
                self.server_name, int(self.timeout),
            )
            self.metrics["errors"] += 1
            return ElicitResult(action="cancel")
        except Exception as exc:
            logger.error(
                "MCP server '%s' elicitation failed: %s",
                self.server_name, exc, exc_info=True,
            )
            self.metrics["errors"] += 1
            return ElicitResult(action="decline")

        if answer == "accept":
            self.metrics["accepted"] += 1
            return ElicitResult(action="accept", content={})
        if answer == "cancel":
            self.metrics["errors"] += 1
            return ElicitResult(action="cancel")
        self.metrics["declined"] += 1
        return ElicitResult(action="decline")


# ---------------------------------------------------------------------------
# Server task -- each MCP server lives in one long-lived asyncio Task
# ---------------------------------------------------------------------------

class MCPServerTask:
    """Manages a single MCP server connection in a dedicated asyncio Task.

    The entire connection lifecycle (connect, discover, serve, disconnect)
    runs inside one asyncio Task so that anyio cancel-scopes created by
    the transport client are entered and exited in the same Task context.

    Supports both stdio and HTTP/StreamableHTTP transports.
    """

    __slots__ = (
        "name", "session", "tool_timeout",
        "_task", "_ready", "_shutdown_event", "_reconnect_event",
        "_tools", "_error", "_config",
        "_sampling", "_elicitation",
        "_registered_tool_names", "_auth_type", "_refresh_lock",
        "_rpc_lock", "_pending_refresh_tasks",
        "_pending_call_context",
        "initialize_result", "_ping_unsupported",
        "_reconnect_retries",
    )

    def __init__(self, name: str):
        self.name = name
        self.session: Optional[Any] = None
        self.tool_timeout: float = _DEFAULT_TOOL_TIMEOUT
        self._task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        # Set by tool handlers on auth failure after manager.handle_401()
        # confirms recovery is viable. When set, _run_http / _run_stdio
        # exit their async-with blocks cleanly (no exception), and the
        # outer run() loop re-enters the transport so the MCP session is
        # rebuilt with fresh credentials.
        self._reconnect_event = asyncio.Event()
        self._tools: list = []
        self._error: Optional[Exception] = None
        self._config: dict = {}
        self._sampling: Optional[SamplingHandler] = None
        self._elicitation: Optional[ElicitationHandler] = None
        self._registered_tool_names: list[str] = []
        self._reconnect_retries: int = 0
        self._auth_type: str = ""
        self._refresh_lock = asyncio.Lock()
        # MCP stdio sessions are a single JSON-RPC stream. Some servers emit
        # list_changed notifications during startup; if the notification
        # handler calls list_tools while a normal tool call is in flight, the
        # stream can wedge and the user-visible tool call times out. Serialize
        # client-initiated RPCs per server. The lock is also applied to HTTP
        # transports for conservative per-server ordering.
        self._rpc_lock = asyncio.Lock()
        self._pending_refresh_tasks: set[asyncio.Task] = set()
        # contextvars snapshot of the agent task that's currently in
        # session.call_tool(). The MCP recv loop dispatches incoming
        # elicitation/create requests on a SEPARATE asyncio task whose
        # context doesn't inherit HERMES_SESSION_PLATFORM, so the
        # elicitation handler has no way to detect the gateway session
        # that triggered the call. Capturing the agent's context here
        # and replaying it inside the elicitation callback restores
        # gateway-platform attribution and routes the approval prompt
        # to the right surface (Telegram, Slack, etc.).
        self._pending_call_context: Optional[contextvars.Context] = None
        # Captures the ``InitializeResult`` returned by
        # ``await session.initialize()`` so downstream code can inspect the
        # server's real advertised capabilities (``.capabilities.resources``,
        # ``.capabilities.prompts``) instead of assuming every ``ClientSession``
        # method attribute corresponds to a supported server method. See #18051.
        self.initialize_result: Optional[Any] = None
        # Set True the first time a keepalive ``ping`` returns JSON-RPC
        # -32601 (method not found): the server is tool-capable but doesn't
        # implement the optional ``ping`` utility. Subsequent keepalives fall
        # back to ``list_tools`` (the pre-ping probe) so we neither spam pings
        # nor reconnect-loop. Reset on each fresh transport connection.
        self._ping_unsupported: bool = False

    def _is_http(self) -> bool:
        """Check if this server uses HTTP transport."""
        return "url" in self._config

    def _advertises_tools(self) -> bool:
        """Whether the server advertises the ``tools`` capability.

        Per the MCP spec, ``InitializeResult.capabilities.tools`` is non-None
        iff the server implements the ``tools/*`` request family. Prompt-only
        or resource-only servers omit it, and calling ``tools/list`` against
        them raises ``McpError(-32601 Method not found)`` — which previously
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
                    match message.root:
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
            # would raise McpError(-32601).
            return

        async with self._refresh_lock:
            # Capture old tool names for change diff
            old_tool_names = set(self._registered_tool_names)

            # 1. Fetch current tool list from server
            async with self._rpc_lock:
                tools_result = await self.session.list_tools()
            new_mcp_tools = tools_result.tools if hasattr(tools_result, "tools") else []

            # 2. Re-register with fresh tool list. Avoid nuke-and-repave for
            # all names: live agent turns may already have tool-call IDs
            # pointing at existing handler functions. Replacing entries
            # in-place is enough for unchanged names and avoids transient
            # "tool not connected" / stale-handler races during startup
            # notifications. Tools absent from the fresh list are no longer
            # callable, so remove only those stale registry entries first.
            stale_tool_names = old_tool_names - {
                mcp_prefixed_tool_name(self.name, tool.name)
                for tool in new_mcp_tools
            }
            for tool_name in stale_tool_names:
                registry.deregister(tool_name)
                _forget_mcp_tool_server(tool_name)

            # 3. Re-register with fresh tool list
            self._tools = new_mcp_tools
            self._registered_tool_names = _register_server_tools(
                self.name, self, self._config
            )

            # 5. Log what changed (user-visible notification)
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
        if not self._ping_unsupported:
            try:
                await asyncio.wait_for(self.session.send_ping(), timeout=30.0)
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
        await asyncio.wait_for(self.session.list_tools(), timeout=30.0)

    async def _wait_for_lifecycle_event(self) -> str:
        """Block until either _shutdown_event or _reconnect_event fires.

        Returns:
            "shutdown"  if the server should exit the run loop entirely.
            "reconnect" if the server should tear down the current MCP
                        session and re-enter the transport (fresh OAuth
                        tokens, new session ID, etc.). The reconnect event
                        is cleared before return so the next cycle starts
                        with a fresh signal.

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
                done, _pending = await asyncio.wait(
                    {shutdown_task, reconnect_task},
                    timeout=keepalive_interval,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if done:
                    break

                # Timeout — no lifecycle event fired.  Probe the connection
                # to detect stale/expired sessions. Prefer ``ping`` (MCP base
                # protocol liveness): it works uniformly and stays a few bytes
                # regardless of tool count, unlike ``list_tools`` (~1 MB on an
                # 830-tool server). ``ping`` is an OPTIONAL utility, so a
                # tool-capable server that doesn't implement it answers -32601;
                # in that case fall back to the pre-ping ``list_tools`` probe
                # for the rest of this connection rather than reconnect-looping.
                if self.session:
                    try:
                        await self._keepalive_probe()
                    except Exception as exc:
                        logger.warning(
                            "MCP server '%s' keepalive failed, "
                            "triggering reconnect: %s",
                            self.name, exc,
                        )
                        self._reconnect_event.set()
                        break
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
        if not _MCP_AVAILABLE:
            raise ImportError(
                f"MCP server '{self.name}' requires the 'mcp' Python SDK, but "
                "it is not installed. Install with:\n"
                "  pip install 'hermes-agent[mcp]'\n"
                "or (full install):\n"
                "  pip install 'hermes-agent[all]'"
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

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=safe_env if safe_env else None,
        )

        sampling_kwargs = self._sampling.session_kwargs() if self._sampling else {}
        if self._elicitation:
            sampling_kwargs.update(self._elicitation.session_kwargs())
        if _MCP_NOTIFICATION_TYPES and _MCP_MESSAGE_HANDLER_SUPPORTED:
            sampling_kwargs["message_handler"] = self._make_message_handler()
        if _MCP_LOGGING_CALLBACK_SUPPORTED:
            sampling_kwargs["logging_callback"] = self._make_logging_callback()

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
                async with ClientSession(
                    read_stream, write_stream, **sampling_kwargs
                ) as session:
                    self.initialize_result = await session.initialize()
                    self.session = session
                    await self._discover_tools()
                    self._ready.set()
                    # Session is live again: clear any breaker state from a
                    # prior outage so the first call after recovery isn't
                    # gated on a stale consecutive-failure count (#16788).
                    _reset_server_error(self.name)
                    # This session is live: reset the reconnect retry counter
                    # so transient prior failures do not accumulate toward
                    # permanent parking (#57604).
                    self._reconnect_retries = 0
                    # stdio transport does not use OAuth, but we still honor
                    # _reconnect_event (e.g. future manual /mcp refresh) for
                    # consistency with _run_http.
                    await self._wait_for_lifecycle_event()
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
                        else:
                            # Nothing left to reap — drop the pgid entry so
                            # PID-reuse can't surface stale pgroup state later.
                            _stdio_pgids.pop(pid, None)

    # Content types a real MCP Streamable-HTTP endpoint may return on the
    # initial POST/GET. Anything else on a 2xx response means the URL is not
    # an MCP endpoint.
    _MCP_CONTENT_TYPES = ("application/json", "text/event-stream")

    async def _preflight_content_type(
        self,
        url: str,
        *,
        headers: Optional[dict] = None,
        ssl_verify: bool = True,
        client_cert=None,
        timeout: float = 5.0,
    ) -> None:
        """Probe *url* for an MCP-shaped response before the SDK connects.

        A misconfigured ``mcp_servers.<name>.url`` pointed at a plain web app
        returns HTML (or some other non-MCP body). The MCP SDK then sits on
        the connection for the full ``connect_timeout`` (default 60 s) before
        surfacing an opaque ``CancelledError``. A cheap, short-timeout probe
        here catches that in ≤ ``timeout`` seconds and raises
        :class:`NonMcpEndpointError` with an actionable message.

        Detection is allow-list based: a 2xx response is rejected only when it
        carries a definite content type that is NOT one an MCP endpoint uses
        (``application/json`` / ``text/event-stream``).  When HEAD/GET returns
        a non-MCP content type (e.g. ``text/html``), a lightweight JSON-RPC
        ``initialize`` POST is attempted before giving up — some servers
        (e.g. DocuSeal) serve a web UI on GET but speak Streamable HTTP only
        via POST.

        A missing or empty content type, non-2xx status, or any
        network/transport error passes through silently — the probe is
        strictly best-effort, and the real handshake remains the source of
        truth for everything except the unambiguous "this is a web page,
        not MCP" case.

        Runs on its own httpx client OUTSIDE the SDK's anyio task group, so the
        raised error propagates as itself rather than being wrapped in an
        ``ExceptionGroup`` (which is what defeats hooks installed inside the
        SDK transport).
        """
        try:
            import httpx as _httpx
        except ImportError:
            return  # No httpx → skip probe; SDK import would have failed first.

        client_kwargs: dict = {
            "verify": ssl_verify,
            "follow_redirects": True,
            "timeout": _httpx.Timeout(timeout),
        }
        if client_cert is not None:
            client_kwargs["cert"] = client_cert

        probe_headers = dict(headers) if headers else {}
        try:
            async with _httpx.AsyncClient(**client_kwargs) as client:
                # HEAD is cheapest; fall back to GET if the server doesn't
                # implement it (405 Method Not Allowed / 501 Not Implemented).
                resp = await client.head(url, headers=probe_headers)
                if resp.status_code in (405, 501):
                    resp = await client.get(url, headers=probe_headers)

                # Some MCP servers (e.g. DocuSeal) serve their web UI on
                # HEAD/GET but speak Streamable HTTP only via POST.  Before
                # rejecting the endpoint, try a lightweight JSON-RPC POST
                # probe so we don't false-positive on POST-only servers.
                ct = (
                    resp.headers.get("content-type", "")
                    .split(";")[0]
                    .strip()
                    .lower()
                )
                if (
                    ct
                    and ct not in self._MCP_CONTENT_TYPES
                    and 200 <= resp.status_code < 300
                ):
                    post_resp = await client.post(
                        url,
                        headers={
                            **probe_headers,
                            "Content-Type": "application/json",
                            "Accept": "application/json, text/event-stream",
                        },
                        content=(
                            '{"jsonrpc":"2.0","id":"_probe",'
                            '"method":"initialize",'
                            '"params":{"protocolVersion":"2025-03-26",'
                            '"capabilities":{},'
                            '"clientInfo":{"name":"hermes-probe",'
                            '"version":"0.1"}}}'
                        ),
                    )
                    if 200 <= post_resp.status_code < 300:
                        post_ct = (
                            post_resp.headers.get("content-type", "")
                            .split(";")[0]
                            .strip()
                            .lower()
                        )
                        if post_ct in self._MCP_CONTENT_TYPES:
                            resp = post_resp
        except _httpx.HTTPError:
            return  # DNS/connect/timeout/transport error — let the SDK try.

        # Only judge successful responses. A 4xx/5xx may be an auth challenge
        # or a transient error the real handshake handles correctly.
        if not (200 <= resp.status_code < 300):
            return

        ct_base = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        if not ct_base:
            return  # No content type advertised — don't second-guess the SDK.
        if ct_base in self._MCP_CONTENT_TYPES:
            return  # Looks like a real MCP endpoint.

        raise NonMcpEndpointError(
            f"MCP server '{self.name}' at {url} returned Content-Type "
            f"'{ct_base}', not an MCP response (expected one of: "
            f"{', '.join(self._MCP_CONTENT_TYPES)}). The URL most likely "
            "points at a web page rather than an MCP endpoint — check it "
            "resolves to a Streamable HTTP / SSE endpoint "
            "(e.g. https://host/mcp, not https://host/)."
        )

    async def _run_http(self, config: dict):
        """Run the server using HTTP/StreamableHTTP transport."""
        if not _MCP_HTTP_AVAILABLE:
            raise ImportError(
                f"MCP server '{self.name}' requires HTTP transport but "
                "mcp.client.streamable_http is not available. "
                "Upgrade the mcp package to get HTTP support."
            )

        url = config["url"]
        headers = dict(config.get("headers") or {})
        # Some MCP servers require MCP-Protocol-Version on the initial
        # initialize request and reject session-less POSTs otherwise.
        # Seed it as a client-level default, but treat user overrides as
        # case-insensitive so conventional casing is preserved.
        if not any(key.lower() == "mcp-protocol-version" for key in headers):
            headers["mcp-protocol-version"] = LATEST_PROTOCOL_VERSION
        connect_timeout = config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
        ssl_verify = config.get("ssl_verify", True)
        client_cert = _resolve_client_cert(self.name, config)

        # OAuth 2.1 PKCE: route through the central MCPOAuthManager so the
        # same provider instance is reused across reconnects, pre-flow
        # disk-watch is active, and config-time CLI code paths share state.
        # If OAuth setup fails (e.g. non-interactive env without cached
        # tokens), re-raise so this server is reported as failed without
        # blocking other MCP servers from connecting.
        _oauth_auth = None
        if self._auth_type == "oauth":
            try:
                from tools.mcp_oauth_manager import get_manager
                _oauth_auth = get_manager().get_or_build_provider(
                    self.name, url, config.get("oauth"),
                )
            except Exception as exc:
                logger.warning("MCP OAuth setup failed for '%s': %s", self.name, exc)
                raise

        sampling_kwargs = self._sampling.session_kwargs() if self._sampling else {}
        if self._elicitation:
            sampling_kwargs.update(self._elicitation.session_kwargs())
        if _MCP_NOTIFICATION_TYPES and _MCP_MESSAGE_HANDLER_SUPPORTED:
            sampling_kwargs["message_handler"] = self._make_message_handler()
        if _MCP_LOGGING_CALLBACK_SUPPORTED:
            sampling_kwargs["logging_callback"] = self._make_logging_callback()

        # SSE transport (for MCP servers that implement the SSE transport protocol
        # rather than Streamable HTTP). Configure with ``transport: sse`` in the
        # mcp_servers entry in config.yaml.
        if config.get("transport") == "sse":
            if sse_client is None:
                raise ImportError(
                    f"MCP server '{self.name}' requires SSE transport but "
                    "mcp.client.sse.sse_client is not available. "
                    "Upgrade the mcp package to get SSE support."
                )
            # sse_read_timeout governs how long sse_client will wait between
            # events on the SSE stream. Using the tool_timeout (default 60s)
            # here is wrong: SSE servers commonly hold the stream idle for
            # minutes between events, so a 60s read timeout drops the
            # connection after the first slow stretch. 300s matches the
            # Streamable HTTP code path's httpx read timeout below. Original
            # observation from @amiller in PR #5981 (Router Teamwork,
            # Supermemory on Cloudflare Workers idle-disconnect at ~60s).
            _sse_kwargs: dict = {
                "url": url,
                "headers": headers or None,
                "timeout": float(connect_timeout),
                "sse_read_timeout": 300.0,
            }
            if _oauth_auth is not None:
                # Pass OAuth auth through to sse_client so SSE MCP servers
                # behind OAuth 2.1 PKCE work. Previously built but never
                # forwarded — SSE OAuth would silently fail with 401s.
                _sse_kwargs["auth"] = _oauth_auth
            if client_cert is not None or ssl_verify is not True:
                # SSE transport doesn't expose verify/cert as kwargs, so route
                # them through an httpx_client_factory that wraps the SDK's
                # defaults (follow_redirects=True) and adds our TLS settings.
                # The SDK calls the factory with (headers, auth, timeout); we
                # forward all of those and layer verify/cert on top.
                import httpx as _httpx_mod

                _cert_for_factory = client_cert
                _verify_for_factory = ssl_verify

                def _mcp_http_client_factory(
                    headers=None, timeout=None, auth=None,
                ):
                    kwargs: dict = {
                        "follow_redirects": True,
                        "verify": _verify_for_factory,
                    }
                    if timeout is not None:
                        kwargs["timeout"] = timeout
                    else:
                        kwargs["timeout"] = _httpx_mod.Timeout(30.0, read=300.0)
                    if headers is not None:
                        kwargs["headers"] = headers
                    if auth is not None:
                        kwargs["auth"] = auth
                    if _cert_for_factory is not None:
                        kwargs["cert"] = _cert_for_factory
                    return _httpx_mod.AsyncClient(**kwargs)

                _sse_kwargs["httpx_client_factory"] = _mcp_http_client_factory
            async with sse_client(**_sse_kwargs) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream, write_stream, **sampling_kwargs
                ) as session:
                    self.initialize_result = await session.initialize()
                    self.session = session
                    await self._discover_tools()
                    self._ready.set()
                    # Session is live again: clear any breaker state from a
                    # prior outage so the first call after recovery isn't
                    # gated on a stale consecutive-failure count (#16788).
                    _reset_server_error(self.name)
                    self._reconnect_retries = 0
                    reason = await self._wait_for_lifecycle_event()
                    if reason == "reconnect":
                        logger.info(
                            "MCP server '%s': reconnect requested — "
                            "tearing down SSE session", self.name,
                        )
            return

        if _MCP_NEW_HTTP:
            # New API (mcp >= 1.24.0): build an explicit httpx.AsyncClient
            # matching the SDK's own create_mcp_http_client defaults.
            import httpx

            _original_url = httpx.URL(url)

            async def _strip_auth_on_cross_origin_redirect(response):
                """Strip Authorization headers when redirected to a different origin."""
                if response.is_redirect and response.next_request:
                    target = response.next_request.url
                    if (target.scheme, target.host, target.port) != (
                        _original_url.scheme, _original_url.host, _original_url.port,
                    ):
                        response.next_request.headers.pop("authorization", None)
                        response.next_request.headers.pop("Authorization", None)

            client_kwargs: dict = {
                "follow_redirects": True,
                "timeout": httpx.Timeout(float(connect_timeout), read=300.0),
                "verify": ssl_verify,
                "event_hooks": {"response": [_strip_auth_on_cross_origin_redirect]},
            }
            if headers:
                client_kwargs["headers"] = headers
            if _oauth_auth is not None:
                client_kwargs["auth"] = _oauth_auth
            if client_cert is not None:
                client_kwargs["cert"] = client_cert

            # Caller owns the client lifecycle — the SDK skips cleanup when
            # http_client is provided, so we wrap in async-with.
            async with httpx.AsyncClient(**client_kwargs) as http_client:
                async with streamable_http_client(url, http_client=http_client) as (
                    read_stream, write_stream, _get_session_id,
                ):
                    async with ClientSession(read_stream, write_stream, **sampling_kwargs) as session:
                        self.initialize_result = await session.initialize()
                        self.session = session
                        await self._discover_tools()
                        self._ready.set()
                        # Session is live again: clear any breaker state from
                        # a prior outage so the first call after recovery
                        # isn't gated on a stale failure count (#16788).
                        _reset_server_error(self.name)
                        self._reconnect_retries = 0
                        reason = await self._wait_for_lifecycle_event()
                        if reason == "reconnect":
                            logger.info(
                                "MCP server '%s': reconnect requested — "
                                "tearing down HTTP session", self.name,
                            )
        else:
            # Deprecated API (mcp < 1.24.0): manages httpx client internally.
            _http_kwargs: dict = {
                "headers": headers,
                "timeout": float(connect_timeout),
                "verify": ssl_verify,
            }
            if _oauth_auth is not None:
                _http_kwargs["auth"] = _oauth_auth
            async with streamablehttp_client(url, **_http_kwargs) as (
                read_stream, write_stream, _get_session_id,
            ):
                async with ClientSession(read_stream, write_stream, **sampling_kwargs) as session:
                    self.initialize_result = await session.initialize()
                    self.session = session
                    await self._discover_tools()
                    self._ready.set()
                    # Session is live again: clear any breaker state from a
                    # prior outage so the first call after recovery isn't
                    # gated on a stale consecutive-failure count (#16788).
                    _reset_server_error(self.name)
                    self._reconnect_retries = 0
                    reason = await self._wait_for_lifecycle_event()
                    if reason == "reconnect":
                        logger.info(
                            "MCP server '%s': reconnect requested — "
                            "tearing down legacy HTTP session", self.name,
                        )

    async def _discover_tools(self):
        """Discover tools from the connected session.

        Capability-gated: prompt-only / resource-only MCP servers don't
        implement ``tools/list``, and calling it raises ``McpError(-32601)``,
        which previously aborted the connection — those servers could never
        stay connected for their prompts/resources. Skip the call when the
        server doesn't advertise the ``tools`` capability.
        (Ported from anomalyco/opencode#31271.)
        """
        # Fresh transport connection → re-probe with the cheap ``ping`` path.
        # Clears any latch from a prior connection in case the server gained
        # ping support across the reconnect.
        self._ping_unsupported = False
        if self.session is None:
            return
        if not self._advertises_tools():
            logger.info(
                "MCP server '%s': does not advertise 'tools' capability — "
                "skipping tools/list (prompts/resources remain available)",
                self.name,
            )
            self._tools = []
            self._register_discovered_tools_if_needed()
            return
        async with self._rpc_lock:
            tools_result = await self.session.list_tools()
        self._tools = (
            tools_result.tools
            if hasattr(tools_result, "tools")
            else []
        )
        self._register_discovered_tools_if_needed()

    def _register_discovered_tools_if_needed(self) -> None:
        """Re-register tools after a post-ready reconnect if needed.

        Initial registration is performed by ``_discover_and_register_server``
        after ``start()`` completes. During a later reconnect, however,
        ``_ready`` remains set; if outage handling previously deregistered
        stale tools (parking calls ``_deregister_tools``), a successful
        revival must publish the freshly discovered tools again — otherwise
        the transport comes back alive with zero registered tools.
        """
        if not self._ready.is_set() or self._registered_tool_names:
            return
        self._registered_tool_names = _register_server_tools(
            self.name, self, self._config
        )

    async def run(self, config: dict):
        """Long-lived coroutine: connect, discover tools, wait, disconnect.

        Includes automatic reconnection with exponential backoff if the
        connection drops unexpectedly (unless shutdown was requested).
        """
        self._config = config
        self.tool_timeout = config.get("timeout", _DEFAULT_TOOL_TIMEOUT)
        self._auth_type = (config.get("auth") or "").lower().strip()

        # Set up sampling handler if enabled and SDK types are available
        sampling_config = config.get("sampling", {})
        if sampling_config.get("enabled", True) and _MCP_SAMPLING_TYPES:
            self._sampling = SamplingHandler(self.name, sampling_config)
        else:
            self._sampling = None

        # Set up elicitation handler if enabled and SDK types are available.
        # Servers use elicitation/create to ask the client for structured
        # input mid-tool-call (e.g. payment authorization). The handler
        # routes those requests through Hermes' approval system.
        elicitation_config = config.get("elicitation", {})
        if elicitation_config.get("enabled", True) and _MCP_ELICITATION_TYPES:
            self._elicitation = ElicitationHandler(self.name, elicitation_config, owner=self)
        else:
            self._elicitation = None

        # Validate: warn if both url and command are present
        if "url" in config and "command" in config:
            logger.warning(
                "MCP server '%s' has both 'url' and 'command' in config. "
                "Using HTTP transport ('url'). Remove 'command' to silence "
                "this warning.",
                self.name,
            )

        # Validate remote URL once, up front.  Raising here (rather than
        # letting it blow up inside the SDK's httpx layer on every retry)
        # means a typo in config.yaml fails fast with a clear error — and
        # critically, no reconnect-backoff burn.  (Ported from
        # anomalyco/opencode#25019.)
        if self._is_http():
            try:
                _validate_remote_mcp_url(self.name, config.get("url"))
            except InvalidMcpUrlError as exc:
                logger.warning("%s", exc)
                self._error = exc
                self._ready.set()
                return

            # Pre-flight content-type probe (Streamable HTTP only; SSE is
            # exercised by its own client and legitimately serves
            # text/event-stream). A URL pointed at a web-app root returns
            # HTML, which makes the SDK hang for the full connect_timeout
            # before surfacing an opaque CancelledError. Probing here — once,
            # outside the SDK task group — fails fast and non-retryably with
            # an actionable message, mirroring the URL-validation path above.
            # Skip the probe when _ready is already set (reconnect after a
            # prior successful connect) — the endpoint was validated once,
            # re-probing is a redundant round-trip. Also skip for OAuth servers:
            # without a cached token the endpoint returns HTML or 401, which
            # would incorrectly block the OAuth flow before it can run.
            if config.get("transport") != "sse" and not config.get("skip_preflight") and not self._ready.is_set() and self._auth_type != "oauth":
                try:
                    _probe_headers = dict(config.get("headers") or {})
                    await self._preflight_content_type(
                        config["url"],
                        headers=_probe_headers,
                        ssl_verify=config.get("ssl_verify", True),
                        client_cert=_resolve_client_cert(self.name, config),
                    )
                except NonMcpEndpointError as exc:
                    logger.warning("%s", exc)
                    self._error = exc
                    self._ready.set()
                    return

        self._reconnect_retries = 0
        initial_retries = 0
        backoff = 1.0

        while True:
            try:
                if self._is_http():
                    await self._run_http(config)
                else:
                    await self._run_stdio(config)
                # Transport returned cleanly. Two cases:
                #  - _shutdown_event was set: exit the run loop entirely.
                #  - _reconnect_event was set (auth recovery): loop back and
                #    rebuild the MCP session with fresh credentials. Do NOT
                #    touch the retry counters — this is not a failure.
                if self._shutdown_event.is_set():
                    break
                logger.info(
                    "MCP server '%s': reconnecting (OAuth recovery or "
                    "manual refresh)",
                    self.name,
                )
                # A clean transport return only happens after a session was
                # successfully established and then asked to rebuild (auth
                # recovery / manual refresh / breaker-driven reconnect). That
                # is proof the server is reachable, so clear the consecutive-
                # failure budget — otherwise transient drops accumulated over
                # a long-lived session would eventually exhaust it and
                # permanently kill an otherwise-healthy server.
                self._reconnect_retries = 0
                backoff = 1.0
                # Reset the session reference and readiness; _run_http/_run_stdio
                # will repopulate both on successful re-entry.  Leaving
                # _ready set here lets handler-side recovery mistake the stale
                # pre-reconnect session for a fresh one and retry too early.
                self._ready.clear()
                self.session = None
                continue
            except asyncio.CancelledError:
                # Task was cancelled (shutdown, gateway restart, explicit
                # task.cancel()). Don't treat this as a connection failure —
                # CancelledError inherits from BaseException (not Exception)
                # in Python 3.11+, so the broad ``except Exception`` below
                # would NOT catch it; we'd silently exit the reconnect loop
                # and the MCP server would stay dead until Hermes is fully
                # restarted. Re-raise so the task's cancellation propagates
                # correctly to asyncio's task machinery and ``shutdown()``'s
                # ``await self._task`` completes. See #9930.
                self.session = None
                raise
            except Exception as exc:
                self.session = None

                # If this is the first connection attempt, retry with backoff
                # before giving up. A transient DNS/network blip at startup
                # should not permanently kill the server.
                # (Ported from Kilo Code's MCP resilience fix.)
                if not self._ready.is_set():
                    if _is_auth_error(exc):
                        logger.warning(
                            "MCP server '%s' failed initial OAuth authentication, "
                            "not retrying automatically: %s",
                            self.name, exc,
                        )
                        self._error = exc
                        self._ready.set()
                        return

                    initial_retries += 1
                    if initial_retries > _MAX_INITIAL_CONNECT_RETRIES:
                        logger.warning(
                            "MCP server '%s' failed initial connection after "
                            "%d attempts, parking until a reconnect is requested: %s",
                            self.name, _MAX_INITIAL_CONNECT_RETRIES, exc,
                        )
                        self._error = exc
                        self._ready.set()
                        self._deregister_tools()
                        self._reconnect_event.clear()
                        parked = await self._wait_for_reconnect_or_shutdown(
                            timeout=_PARKED_RETRY_INTERVAL
                        )
                        if parked == "shutdown":
                            return
                        logger.info(
                            "MCP server '%s': attempting revival after initial "
                            "connection failures (self-probe or explicit "
                            "reconnect request); rebuilding transport.",
                            self.name,
                        )
                        initial_retries = 0
                        self._reconnect_retries = 0
                        backoff = 1.0
                        self._error = None
                        self._ready.clear()
                        continue

                    logger.warning(
                        "MCP server '%s' initial connection failed "
                        "(attempt %d/%d), retrying in %.0fs: %s",
                        self.name, initial_retries,
                        _MAX_INITIAL_CONNECT_RETRIES, backoff, exc,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

                    # Check if shutdown was requested during the sleep
                    if self._shutdown_event.is_set():
                        self._error = exc
                        self._ready.set()
                        return
                    continue

                # If shutdown was requested, don't reconnect
                if self._shutdown_event.is_set():
                    logger.debug(
                        "MCP server '%s' disconnected during shutdown: %s",
                        self.name, exc,
                    )
                    return

                self._reconnect_retries += 1
                if self._reconnect_retries > _MAX_RECONNECT_RETRIES:
                    logger.warning(
                        "MCP server '%s' failed after %d reconnection attempts, "
                        "parking; will self-probe every %ds until it recovers: %s",
                        self.name, _MAX_RECONNECT_RETRIES,
                        _PARKED_RETRY_INTERVAL, exc,
                    )
                    # Do NOT return — exiting the task orphans the server:
                    # nothing would ever listen for _reconnect_event again
                    # and the server would be permanently wedged for the
                    # life of the process (#16788). Instead, drop the phantom
                    # tools from the registry and park. Because parking
                    # deregisters the tools, no tool call can reach the
                    # circuit-breaker half-open probe or _signal_reconnect —
                    # so the park is a TIMED wait: every _PARKED_RETRY_INTERVAL
                    # we wake and attempt one reconnect ourselves (#57129).
                    # An explicit _reconnect_event.set() (OAuth recovery,
                    # manual /mcp refresh) still wakes us immediately.
                    self._deregister_tools()
                    self._reconnect_event.clear()
                    parked = await self._wait_for_reconnect_or_shutdown(
                        timeout=_PARKED_RETRY_INTERVAL
                    )
                    if parked == "shutdown":
                        return
                    logger.info(
                        "MCP server '%s': attempting revival from parked state "
                        "(self-probe or explicit reconnect request); "
                        "rebuilding transport.",
                        self.name,
                    )
                    # One probe attempt per wake: budget of 1 so a still-dead
                    # server parks again for another interval instead of
                    # burning 5 rapid retries each cycle.
                    self._reconnect_retries = _MAX_RECONNECT_RETRIES
                    backoff = 1.0
                    continue

                logger.warning(
                    "MCP server '%s' connection lost (attempt %d/%d), "
                    "reconnecting in %.0fs: %s",
                    self.name, self._reconnect_retries, _MAX_RECONNECT_RETRIES,
                    backoff, exc,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

                # Check again after sleeping
                if self._shutdown_event.is_set():
                    return
            finally:
                self.session = None

    async def start(self, config: dict):
        """Create the background Task and wait until ready (or failed)."""
        self._task = asyncio.ensure_future(self.run(config))
        await self._ready.wait()
        if self._error:
            raise self._error

    async def shutdown(self):
        """Signal the Task to exit and wait for clean resource teardown."""
        self._shutdown_event.set()
        # Defensive: if _wait_for_lifecycle_event is blocking, we need ANY
        # event to unblock it. _shutdown_event alone is sufficient (the
        # helper checks shutdown first), but setting reconnect too ensures
        # there's no race where the helper misses the shutdown flag after
        # returning "reconnect".
        self._reconnect_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP server '%s' shutdown timed out, cancelling task",
                    self.name,
                )
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
        """Drop this server's tools from the global registry (idempotent).

        Pulls the server's tool schemas out of the registry so the agent
        stops advertising them to the model. Called on shutdown AND when the
        reconnect budget is exhausted, so a dead server never leaves phantom
        tool definitions bloating the prompt cache and producing "not
        connected" errors on every turn.
        """
        from tools.registry import registry

        for tool_name in list(getattr(self, "_registered_tool_names", [])):
            registry.deregister(tool_name)
            _forget_mcp_tool_server(tool_name)
        self._registered_tool_names = []


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_servers: Dict[str, MCPServerTask] = {}
_server_connecting: set[str] = set()
_server_connect_errors: Dict[str, str] = {}

# Circuit breaker: consecutive error counts per server.  After
# _CIRCUIT_BREAKER_THRESHOLD consecutive failures, the handler returns
# a "server unreachable" message that tells the model to stop retrying,
# preventing the 90-iteration burn loop described in #10447.
#
# State machine:
#   closed    — error count below threshold; all calls go through.
#   open      — threshold reached; calls short-circuit until the
#               cooldown elapses.
#   half-open — cooldown elapsed; the next call is a probe that
#               actually hits the session. Probe success → closed.
#               Probe failure → reopens (cooldown re-armed).
#
# ``_server_breaker_opened_at`` records the monotonic timestamp when
# the breaker most recently transitioned into the open state. Use the
# ``_bump_server_error`` / ``_reset_server_error`` helpers to mutate
# this state — they keep the count and timestamp in sync.
_server_error_counts: Dict[str, int] = {}
_server_breaker_opened_at: Dict[str, float] = {}
_CIRCUIT_BREAKER_THRESHOLD = 3
_CIRCUIT_BREAKER_COOLDOWN_SEC = 60.0


def _bump_server_error(server_name: str) -> None:
    """Increment the consecutive-failure count for ``server_name``.

    When the count crosses :data:`_CIRCUIT_BREAKER_THRESHOLD`, stamp the
    breaker-open timestamp so the cooldown clock starts (or re-starts,
    for probe failures in the half-open state).
    """
    n = _server_error_counts.get(server_name, 0) + 1
    _server_error_counts[server_name] = n
    if n >= _CIRCUIT_BREAKER_THRESHOLD:
        _server_breaker_opened_at[server_name] = time.monotonic()


def _reset_server_error(server_name: str) -> None:
    """Fully close the breaker for ``server_name``.

    Clears both the failure count and the breaker-open timestamp. Call
    this on any unambiguous success signal (successful tool call,
    successful reconnect, manual /mcp refresh).
    """
    _server_error_counts[server_name] = 0
    _server_breaker_opened_at.pop(server_name, None)


def _signal_reconnect(server: Any) -> bool:
    """Ask a server task to rebuild its transport, thread-safely.

    The tool handlers run on caller threads, while the server task and its
    ``_reconnect_event`` live on the background MCP loop. Setting an
    asyncio.Event from another thread must go through
    ``loop.call_soon_threadsafe``; only fall back to a direct ``.set()``
    when the loop isn't running (e.g. unit tests that drive the handler
    synchronously).

    Returns True if a reconnect signal was delivered, False if the server
    has no reconnect machinery (nothing to revive).
    """
    event = getattr(server, "_reconnect_event", None)
    if event is None:
        return False
    loop = _mcp_loop
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(event.set)
    else:
        event.set()
    return True


def _wait_for_server_session_ready(
    srv: "MCPServerTask",
    *,
    old_session: Any = None,
    timeout: float = 15.0,
) -> bool:
    """Wait for an MCP server to expose a usable session.

    Tool handlers run in normal worker threads while the MCP transport lives on
    the module's background asyncio loop. During a reconnect there is a short
    window where ``srv.session`` is ``None`` (or still points at the stale
    session until the lifecycle coroutine has left the transport context). A
    handler that blindly retries in that window can burn circuit-breaker strikes
    and return ``not connected`` even though the reconnect is already in
    progress.

    When ``old_session`` is supplied, require the observed session object to be
    different so callers do not mistake the pre-reconnect, stale session for a
    fresh one.
    """
    # Iteration-bounded rather than deadline-bounded: several tests (and the
    # circuit-breaker cooldown logic) monkeypatch time.monotonic to a frozen
    # clock, which would make a monotonic-deadline loop spin forever.
    poll_interval = 0.25
    iterations = max(1, int(max(float(timeout), 0.0) / poll_interval))
    for i in range(iterations):
        session = getattr(srv, "session", None)
        ready = getattr(srv, "_ready", None)
        is_ready = True
        if ready is not None and hasattr(ready, "is_set"):
            try:
                is_ready = bool(ready.is_set())
            except Exception:
                is_ready = True
        if session is not None and session is not old_session and is_ready:
            return True
        if i < iterations - 1:
            time.sleep(poll_interval)
    return False


def _signal_reconnect_and_wait(
    server_name: str,
    srv: "MCPServerTask",
    *,
    op_description: str,
    timeout: float = 15.0,
) -> bool:
    """Ask a live MCP server task to rebuild its transport session.

    The important detail is clearing ``_ready`` on the MCP event loop before
    setting ``_reconnect_event``. Older code left ``_ready`` set across
    reconnects, so the caller's readiness poll could return immediately and
    retry against the same dead HTTP/stream session. That was observed as
    repeated ``Session terminated`` / ``not connected`` / circuit-breaker
    failures in long-lived gateway sessions even though a fresh CLI process
    could connect successfully.
    """
    loop = _mcp_loop
    if loop is None or not loop.is_running():
        return False

    old_session = getattr(srv, "session", None)

    def _request_reconnect() -> None:
        ready = getattr(srv, "_ready", None)
        if ready is not None and hasattr(ready, "clear"):
            ready.clear()
        reconnect_event = getattr(srv, "_reconnect_event", None)
        if reconnect_event is not None and hasattr(reconnect_event, "set"):
            reconnect_event.set()

    logger.info(
        "MCP server '%s': %s requesting transport reconnect",
        server_name, op_description,
    )
    loop.call_soon_threadsafe(_request_reconnect)
    return _wait_for_server_session_ready(
        srv,
        old_session=old_session,
        timeout=timeout,
    )

# ---------------------------------------------------------------------------
# Auth-failure detection helpers (Task 6 of MCP OAuth consolidation)
# ---------------------------------------------------------------------------

# Cached tuple of auth-related exception types. Lazy so this module
# imports cleanly when the MCP SDK OAuth module is missing.
_AUTH_ERROR_TYPES: tuple = ()


def _get_auth_error_types() -> tuple:
    """Return a tuple of exception types that indicate MCP OAuth failure.

    Cached after first call. Includes:
      - ``mcp.client.auth.OAuthFlowError`` / ``OAuthTokenError`` — raised by
        the SDK's auth flow when discovery, refresh, or full re-auth fails.
      - ``mcp.client.auth.UnauthorizedError`` (older MCP SDKs) — kept as an
        optional import for forward/backward compatibility.
      - ``tools.mcp_oauth.OAuthNonInteractiveError`` — raised by our callback
        handler when no user is present to complete a browser flow.
      - ``httpx.HTTPStatusError`` — caller must additionally check
        ``status_code == 401`` via :func:`_is_auth_error`.
    """
    global _AUTH_ERROR_TYPES
    if _AUTH_ERROR_TYPES:
        return _AUTH_ERROR_TYPES
    types: list = []
    try:
        from mcp.client.auth import OAuthFlowError, OAuthTokenError
        types.extend([OAuthFlowError, OAuthTokenError])
    except ImportError:
        pass
    try:
        # Older MCP SDK variants exported this
        from mcp.client.auth import UnauthorizedError  # type: ignore
        types.append(UnauthorizedError)
    except ImportError:
        pass
    try:
        from tools.mcp_oauth import OAuthNonInteractiveError
        types.append(OAuthNonInteractiveError)
    except ImportError:
        pass
    try:
        import httpx
        types.append(httpx.HTTPStatusError)
    except ImportError:
        pass
    _AUTH_ERROR_TYPES = tuple(types)
    return _AUTH_ERROR_TYPES


def _is_auth_error(exc: BaseException) -> bool:
    """Return True if ``exc`` indicates an MCP OAuth failure.

    ``httpx.HTTPStatusError`` is only treated as auth-related when the
    response status code is 401. Other HTTP errors fall through to the
    generic error path in the tool handlers.
    """
    types = _get_auth_error_types()
    if not types or not isinstance(exc, types):
        return False
    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError):
            return getattr(exc.response, "status_code", None) == 401
    except ImportError:
        pass
    return True


def _handle_auth_error_and_retry(
    server_name: str,
    exc: BaseException,
    retry_call,
    op_description: str,
):
    """Attempt auth recovery and one retry; return None to fall through.

    Called by the 5 MCP tool handlers when ``session.<op>()`` raises an
    auth-related exception. Workflow:

      1. Ask :class:`tools.mcp_oauth_manager.MCPOAuthManager.handle_401` if
         recovery is viable (i.e., disk has fresh tokens, or the SDK can
         refresh in-place).
      2. If yes, set the server's ``_reconnect_event`` so the server task
         tears down the current MCP session and rebuilds it with fresh
         credentials. Wait briefly for ``_ready`` to re-fire.
      3. Retry the operation once. Return the retry result if it produced
         a non-error JSON payload. Otherwise return the ``needs_reauth``
         error dict so the model stops hallucinating manual refresh.
      4. Return None if ``exc`` is not an auth error, signalling the
         caller to use the generic error path.

    Args:
        server_name: Name of the MCP server that raised.
        exc: The exception from the failed tool call.
        retry_call: Zero-arg callable that re-runs the tool call, returning
            the same JSON string format as the handler.
        op_description: Human-readable name of the operation (for logs).

    Returns:
        A JSON string if auth recovery was attempted, or None to fall
        through to the caller's generic error path.
    """
    if not _is_auth_error(exc):
        return None

    from tools.mcp_oauth_manager import get_manager
    manager = get_manager()

    async def _recover():
        return await manager.handle_401(server_name, None)

    try:
        recovered = _run_on_mcp_loop(_recover, timeout=10)
    except Exception as rec_exc:
        logger.warning(
            "MCP OAuth '%s': recovery attempt failed: %s",
            server_name, rec_exc,
        )
        recovered = False

    if recovered:
        with _lock:
            srv = _servers.get(server_name)
        reconnected = False
        if srv is not None and hasattr(srv, "_reconnect_event"):
            reconnected = _signal_reconnect_and_wait(
                server_name,
                srv,
                op_description=f"{op_description} after OAuth recovery",
                timeout=15,
            )

        # A successful OAuth recovery + transport reconnect is independent
        # evidence that the server is viable again, so close the circuit
        # breaker here — not only on retry success. Without this, a reconnect
        # followed by a failing retry would leave the breaker pinned above
        # threshold forever. The post-reset retry still goes through
        # _bump_server_error on failure, so a genuinely broken server will
        # re-trip the breaker as normal.
        if reconnected:
            _reset_server_error(server_name)

        try:
            result = retry_call()
            try:
                parsed = json.loads(result)
                if "error" not in parsed:
                    _reset_server_error(server_name)
                    return result
            except (json.JSONDecodeError, TypeError):
                _reset_server_error(server_name)
                return result
        except Exception as retry_exc:
            logger.warning(
                "MCP %s/%s retry after auth recovery failed: %s",
                server_name, op_description, retry_exc,
            )

    # No recovery available, or retry also failed: surface a structured
    # needs_reauth error. Bumps the circuit breaker so the model stops
    # retrying the tool.
    _bump_server_error(server_name)
    return json.dumps({
        "error": (
            f"MCP server '{server_name}' requires re-authentication. "
            f"Run `hermes mcp login {server_name}` (or delete the tokens "
            f"file under ~/.hermes/mcp-tokens/ and restart). Do NOT retry "
            f"this tool — ask the user to re-authenticate."
        ),
        "needs_reauth": True,
        "server": server_name,
    }, ensure_ascii=False)


# Substrings (lower-cased match) that indicate the MCP server rejected
# the request because its server-side transport session expired /
# was garbage-collected.  The caller's OAuth token is still valid —
# only the transport-layer session state needs rebuilding.  See #13383.
_SESSION_EXPIRED_MARKERS: tuple = (
    "invalid or expired session",
    "expired session",
    "session expired",
    "session not found",
    "unknown session",
    "session terminated",
    "closedresourceerror",
    "closed resource",
    "transport is closed",
    "connection closed",
    "broken pipe",
    "end of file",
)


def _is_session_expired_error(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like an MCP transport session expiry.

    Streamable HTTP MCP servers may garbage-collect server-side session
    state while the OAuth token remains valid — idle TTL, server
    restart, horizontal-scaling pod rotation, etc.  The SDK surfaces
    this as a JSON-RPC error whose message contains phrases like
    ``"Invalid or expired session"``.  This class of failure is
    distinct from :func:`_is_auth_error`: re-running the OAuth refresh
    flow would be pointless because the access token is fine.  What's
    needed is a transport reconnect — tear down and rebuild the
    ``streamablehttp_client`` + ``ClientSession`` pair, which is
    exactly what ``MCPServerTask._reconnect_event`` triggers.
    """
    if isinstance(exc, InterruptedError):
        return False
    # Exception messages vary across SDK versions + server
    # implementations, so match on a small allow-list of stable
    # substrings rather than exception type.  Kept narrow to avoid
    # false positives on unrelated server errors.
    msg = str(exc).lower()
    if not msg:
        return False
    return any(marker in msg for marker in _SESSION_EXPIRED_MARKERS)


def _handle_session_expired_and_retry(
    server_name: str,
    exc: BaseException,
    retry_call,
    op_description: str,
):
    """Trigger a transport reconnect and retry once on session expiry.

    Unlike :func:`_handle_auth_error_and_retry`, this does **not** call
    the OAuth manager's ``handle_401`` — the access token is still
    valid, only the server-side session state is stale.  Setting
    ``_reconnect_event`` causes the server task's lifecycle loop to
    tear down the current ``streamablehttp_client`` + ``ClientSession``
    and rebuild them, reusing the existing OAuth provider instance.
    See #13383.

    Args:
        server_name: Name of the MCP server that raised.
        exc: The exception from the failed call.
        retry_call: Zero-arg callable that re-runs the operation,
            returning the same JSON string format as the handler.
        op_description: Human-readable name of the operation (logs).

    Returns:
        A JSON string if reconnect + retry was attempted and produced
        a response, or ``None`` to fall through to the caller's
        generic error path (not a session-expired error, no server
        record, reconnect didn't ready in time, or retry also failed).
    """
    if not _is_session_expired_error(exc):
        return None

    with _lock:
        srv = _servers.get(server_name)
    if srv is None or not hasattr(srv, "_reconnect_event"):
        return None

    loop = _mcp_loop
    if loop is None or not loop.is_running():
        return None

    logger.info(
        "MCP server '%s': %s failed with session-expired error (%s); "
        "signalling transport reconnect and retrying once.",
        server_name, op_description, exc,
    )

    # Trigger the same reconnect mechanism the OAuth recovery path
    # uses, then wait briefly for the new session to come back ready.
    if not _signal_reconnect_and_wait(
        server_name,
        srv,
        op_description=op_description,
        timeout=15,
    ):
        logger.warning(
            "MCP server '%s': reconnect did not ready within 15s after "
            "session-expired error; falling through to error response.",
            server_name,
        )
        return None

    try:
        result = retry_call()
        try:
            parsed = json.loads(result)
            if "error" not in parsed:
                _server_error_counts[server_name] = 0
                return result
        except (json.JSONDecodeError, TypeError):
            _server_error_counts[server_name] = 0
            return result
    except Exception as retry_exc:
        logger.warning(
            "MCP %s/%s retry after session reconnect failed: %s",
            server_name, op_description, retry_exc,
        )
    return None


# Sanitized server names whose ``supports_parallel_tool_calls`` config is True.
# Populated during ``register_mcp_servers()`` and queried by
# ``is_mcp_tool_parallel_safe()`` for the parallel-execution check in run_agent.
_parallel_safe_servers: set = set()

# Exact MCP tool-name provenance. MCP tool names are formatted as
# ``mcp_{sanitized_server}_{sanitized_tool}``, which is ambiguous when server
# names contain underscores (``mcp_a_b_tool`` could be server ``a`` + tool
# ``b_tool`` or server ``a_b`` + tool ``tool``). Keep the server component
# captured at registration time so parallel safety never relies on prefix
# guessing.
_mcp_tool_server_names: Dict[str, str] = {}

# Dedicated event loop running in a background daemon thread.
_mcp_loop: Optional[asyncio.AbstractEventLoop] = None
_mcp_thread: Optional[threading.Thread] = None

# Protects _mcp_loop, _mcp_thread, _servers, MCP connection status maps,
# _parallel_safe_servers, _mcp_tool_server_names, and _stdio_pids.
_lock = threading.Lock()

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
            continue


def _interrupted_call_result() -> str:
    """Standardized JSON error for a user-interrupted MCP tool call."""
    return json.dumps({
        "error": "MCP call interrupted: user sent a new message"
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _interpolate_env_vars(value):
    """Recursively resolve ``${VAR}`` placeholders.

    Both ``${VAR}`` and Cursor-style ``${env:VAR}`` are accepted — the
    ``env:`` prefix is stripped so a doc copied from a Cursor / Claude MCP
    config resolves the same secret. Resolves from the active profile's secret
    scope when multiplexing is on (so an MCP server config's ``${API_KEY}``
    picks up the routed profile's value, not the process-global ``os.environ``
    which may hold another profile's), falling back to ``os.environ``
    otherwise. Unset vars keep the literal placeholder, as before.
    """
    from agent.secret_scope import get_secret as _get_secret

    if isinstance(value, str):
        def _replace(m):
            name = _env_ref_name(m.group(1))
            return _get_secret(name, m.group(0)) or m.group(0)
        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


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
        if not servers or not isinstance(servers, dict):
            return {}
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
                safe_servers[name] = interpolated
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
    await server.start(config)
    return server


# ---------------------------------------------------------------------------
# Handler / check-fn factories
# ---------------------------------------------------------------------------

def _make_tool_handler(server_name: str, tool_name: str, tool_timeout: float):
    """Return a sync handler that calls an MCP tool via the background loop.

    The handler conforms to the registry's dispatch interface:
    ``handler(args_dict, **kwargs) -> str``
    """

    def _handler(args: dict, **kwargs) -> str:
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
                return json.dumps({
                    "error": (
                        f"MCP server '{server_name}' is unreachable after "
                        f"{_server_error_counts[server_name]} consecutive "
                        f"failures. Auto-retry available in ~{remaining}s. "
                        f"Do NOT retry this tool yet — use alternative "
                        f"approaches or ask the user to check the MCP server."
                    )
                }, ensure_ascii=False)
            # Cooldown elapsed → fall through as a half-open probe.

        with _lock:
            server = _servers.get(server_name)
        if not server:
            _bump_server_error(server_name)
            return json.dumps({
                "error": f"MCP server '{server_name}' is not connected"
            }, ensure_ascii=False)

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
                    return json.dumps({
                        "error": (
                            f"MCP server '{server_name}' transport is down; "
                            f"reconnect requested. Do NOT retry this tool "
                            f"immediately — give it a few seconds to come back."
                        )
                    }, ensure_ascii=False)
                return json.dumps({
                    "error": f"MCP server '{server_name}' is not connected"
                }, ensure_ascii=False)

        async def _call():
            async with server._rpc_lock:
                # Snapshot the agent's context so an elicitation callback
                # triggered during this call (fired on the MCP recv loop
                # task, which doesn't inherit our contextvars) can replay
                # it and detect the gateway platform / session for routing.
                server._pending_call_context = contextvars.copy_context()
                try:
                    result = await server.session.call_tool(tool_name, arguments=args)
                finally:
                    server._pending_call_context = None
            # MCP CallToolResult has .content (list of content blocks) and .isError
            if result.isError:
                error_text = ""
                for block in (result.content or []):
                    if hasattr(block, "text"):
                        error_text += block.text
                return json.dumps({
                    "error": _sanitize_error(
                        error_text or "MCP tool returned an error"
                    )
                }, ensure_ascii=False)

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
                    parts.append(block.text)
                    continue
                image_tag = _cache_mcp_image_block(block)
                if image_tag:
                    parts.append(image_tag)
            text_result = "\n".join(parts) if parts else ""

            # Combine content + structuredContent when both are present.
            # MCP spec: content is model-oriented (text), structuredContent
            # is machine-oriented (JSON metadata).  For an AI agent, content
            # is the primary payload; structuredContent supplements it.
            structured = getattr(result, "structuredContent", None)
            if structured is not None:
                if text_result:
                    return json.dumps({
                        "result": text_result,
                        "structuredContent": structured,
                    }, ensure_ascii=False)
                return json.dumps({"result": structured}, ensure_ascii=False)
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
            return json.dumps({
                "error": _sanitize_error(
                    f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
                )
            }, ensure_ascii=False)

    return _handler


def _make_list_resources_handler(server_name: str, tool_timeout: float):
    """Return a sync handler that lists resources from an MCP server."""

    def _handler(args: dict, **kwargs) -> str:
        with _lock:
            server = _servers.get(server_name)
        if not server or not server.session:
            return json.dumps({
                "error": f"MCP server '{server_name}' is not connected"
            }, ensure_ascii=False)

        async def _call():
            async with server._rpc_lock:
                result = await server.session.list_resources()
            resources = []
            for r in (result.resources if hasattr(result, "resources") else []):
                entry = {}
                if hasattr(r, "uri"):
                    entry["uri"] = str(r.uri)
                if hasattr(r, "name"):
                    entry["name"] = r.name
                if hasattr(r, "description") and r.description:
                    entry["description"] = r.description
                if hasattr(r, "mimeType") and r.mimeType:
                    entry["mimeType"] = r.mimeType
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
            return json.dumps({
                "error": _sanitize_error(
                    f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
                )
            }, ensure_ascii=False)

    return _handler


def _make_read_resource_handler(server_name: str, tool_timeout: float):
    """Return a sync handler that reads a resource by URI from an MCP server."""

    def _handler(args: dict, **kwargs) -> str:
        from tools.registry import tool_error

        with _lock:
            server = _servers.get(server_name)
        if not server or not server.session:
            return json.dumps({
                "error": f"MCP server '{server_name}' is not connected"
            }, ensure_ascii=False)

        uri = args.get("uri")
        if not uri:
            return tool_error("Missing required parameter 'uri'")

        async def _call():
            async with server._rpc_lock:
                result = await server.session.read_resource(uri)
            # read_resource returns ReadResourceResult with .contents list
            parts: List[str] = []
            contents = result.contents if hasattr(result, "contents") else []
            for block in contents:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif hasattr(block, "blob"):
                    parts.append(f"[binary data, {len(block.blob)} bytes]")
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
            return json.dumps({
                "error": _sanitize_error(
                    f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
                )
            }, ensure_ascii=False)

    return _handler


def _make_list_prompts_handler(server_name: str, tool_timeout: float):
    """Return a sync handler that lists prompts from an MCP server."""

    def _handler(args: dict, **kwargs) -> str:
        with _lock:
            server = _servers.get(server_name)
        if not server or not server.session:
            return json.dumps({
                "error": f"MCP server '{server_name}' is not connected"
            }, ensure_ascii=False)

        async def _call():
            async with server._rpc_lock:
                result = await server.session.list_prompts()
            prompts = []
            for p in (result.prompts if hasattr(result, "prompts") else []):
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
            return json.dumps({
                "error": _sanitize_error(
                    f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
                )
            }, ensure_ascii=False)

    return _handler


def _make_get_prompt_handler(server_name: str, tool_timeout: float):
    """Return a sync handler that gets a prompt by name from an MCP server."""

    def _handler(args: dict, **kwargs) -> str:
        from tools.registry import tool_error

        with _lock:
            server = _servers.get(server_name)
        if not server or not server.session:
            return json.dumps({
                "error": f"MCP server '{server_name}' is not connected"
            }, ensure_ascii=False)

        name = args.get("name")
        if not name:
            return tool_error("Missing required parameter 'name'")
        arguments = args.get("arguments", {})

        async def _call():
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
                        entry["content"] = content.text
                    elif isinstance(content, str):
                        entry["content"] = content
                    else:
                        entry["content"] = str(content)
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
            return json.dumps({
                "error": _sanitize_error(
                    f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
                )
            }, ensure_ascii=False)

    return _handler


def _make_check_fn(server_name: str):
    """Return a check function that verifies the MCP connection is alive."""

    def _check() -> bool:
        with _lock:
            server = _servers.get(server_name)
        return server is not None and server.session is not None

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
                     and ``.inputSchema``.

    Returns:
        A dict suitable for ``registry.register(schema=...)``.
    """
    prefixed_name = mcp_prefixed_tool_name(server_name, mcp_tool.name)
    return {
        "name": prefixed_name,
        "description": mcp_tool.description or f"MCP tool {mcp_tool.name} from {server_name}",
        "parameters": _normalize_mcp_input_schema(getattr(mcp_tool, "inputSchema", None)),
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
    """Normalize include/exclude config to a set of tool names."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    logger.warning("MCP config %s must be a string or list of strings; ignoring %r", label, value)
    return set()


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


_UTILITY_CAPABILITY_METHODS = {
    "list_resources": "list_resources",
    "read_resource": "read_resource",
    "list_prompts": "list_prompts",
    "get_prompt": "get_prompt",
}

# Maps each utility handler to the MCP capability key that must be non-None
# on the server's ``initialize`` response for the handler to be registered.
# Source of truth: MCP spec — capabilities.resources / capabilities.prompts
# are present on the response only when the server actually implements
# those request families. Without this gate, tools-only servers (e.g.
# Context7 @upstash/context7-mcp, which advertises only ``tools``) had
# all four utility stubs registered and every model call to them came
# back with JSON-RPC ``-32601 Method not found``, which made the model
# conclude the server was broken even when the real tools worked. See
# #18051.
_UTILITY_CAPABILITY_ATTRS = {
    "list_resources": "resources",
    "read_resource": "resources",
    "list_prompts": "prompts",
    "get_prompt": "prompts",
}


def _track_mcp_tool_server(tool_name: str, server_name: str) -> None:
    """Remember the exact MCP server that registered *tool_name*."""
    safe_server_name = sanitize_mcp_name_component(server_name)
    with _lock:
        _mcp_tool_server_names[tool_name] = safe_server_name


def _forget_mcp_tool_server(tool_name: str) -> None:
    """Forget MCP server provenance for a deregistered tool."""
    with _lock:
        _mcp_tool_server_names.pop(tool_name, None)


def _select_utility_schemas(server_name: str, server: MCPServerTask, config: dict) -> List[dict]:
    """Select utility schemas based on config and server capabilities."""
    tools_filter = config.get("tools") or {}
    resources_enabled = _parse_boolish(tools_filter.get("resources"), default=True)
    prompts_enabled = _parse_boolish(tools_filter.get("prompts"), default=True)

    # ``initialize_result.capabilities`` is the source of truth: its sub-objects
    # (``resources``, ``prompts``) are non-None iff the server advertises that
    # request family. ``hasattr(server.session, ...)`` was the old gate but
    # ClientSession always has the four method attributes defined on the class,
    # so it never filtered anything.
    advertised_caps = None
    init_result = getattr(server, "initialize_result", None)
    if init_result is not None:
        advertised_caps = getattr(init_result, "capabilities", None)

    selected: List[dict] = []
    for entry in _build_utility_schemas(server_name):
        handler_key = entry["handler_key"]
        if handler_key in {"list_resources", "read_resource"} and not resources_enabled:
            logger.debug("MCP server '%s': skipping utility '%s' (resources disabled)", server_name, handler_key)
            continue
        if handler_key in {"list_prompts", "get_prompt"} and not prompts_enabled:
            logger.debug("MCP server '%s': skipping utility '%s' (prompts disabled)", server_name, handler_key)
            continue

        # Preferred gate: check the server's advertised capabilities. Skip
        # if the capability is explicitly not advertised.
        if advertised_caps is not None:
            cap_attr = _UTILITY_CAPABILITY_ATTRS[handler_key]
            if getattr(advertised_caps, cap_attr, None) is None:
                logger.debug(
                    "MCP server '%s': skipping utility '%s' "
                    "(server does not advertise '%s' capability)",
                    server_name,
                    handler_key,
                    cap_attr,
                )
                continue
        else:
            # Legacy fallback for test fixtures or older code paths where
            # initialize_result wasn't captured. Preserves the old behavior
            # of registering every stub in that case rather than regressing
            # any server that was working before this fix.
            required_method = _UTILITY_CAPABILITY_METHODS[handler_key]
            if not hasattr(server.session, required_method):
                logger.debug(
                    "MCP server '%s': skipping utility '%s' (session lacks %s)",
                    server_name,
                    handler_key,
                    required_method,
                )
                continue
        selected.append(entry)
    return selected


def _existing_tool_names() -> List[str]:
    """Return tool names for all currently connected servers."""
    names: List[str] = []
    for _sname, server in _servers.items():
        if hasattr(server, "_registered_tool_names"):
            names.extend(server._registered_tool_names)
            continue
        for mcp_tool in server._tools:
            schema = _convert_mcp_schema(server.name, mcp_tool)
            names.append(schema["name"])
    return names


def _register_server_tools(name: str, server: MCPServerTask, config: dict) -> List[str]:
    """Register tools from an already-connected server into the registry.

    Handles include/exclude filtering and utility tools. Toolset resolution
    for ``mcp-{server}`` and raw server-name aliases is derived from the live
    registry, rather than mutating ``toolsets.TOOLSETS`` at runtime.

    Used by both initial discovery and dynamic refresh (list_changed).

    Returns:
        List of registered prefixed tool names.
    """
    from tools.registry import registry

    registered_names: List[str] = []
    toolset_name = f"mcp-{name}"

    # Selective tool loading: honour include/exclude lists from config.
    # Rules (matching issue #690 spec):
    #   tools.include — whitelist: only these tool names are registered
    #   tools.exclude — blacklist: all tools EXCEPT these are registered
    #   include takes precedence over exclude
    #   Neither set → register all tools (backward-compatible default)
    tools_filter = config.get("tools") or {}
    include_set = _normalize_name_filter(tools_filter.get("include"), f"mcp_servers.{name}.tools.include")
    exclude_set = _normalize_name_filter(tools_filter.get("exclude"), f"mcp_servers.{name}.tools.exclude")

    def _should_register(tool_name: str) -> bool:
        if include_set:
            return tool_name in include_set
        if exclude_set:
            return tool_name not in exclude_set
        return True

    for mcp_tool in server._tools:
        if not _should_register(mcp_tool.name):
            logger.debug("MCP server '%s': skipping tool '%s' (filtered by config)", name, mcp_tool.name)
            continue

        # Scan tool description for prompt injection patterns
        _scan_mcp_description(name, mcp_tool.name, mcp_tool.description or "")

        schema = _convert_mcp_schema(name, mcp_tool)
        tool_name_prefixed = schema["name"]

        # Guard against collisions with built-in (non-MCP) tools.
        existing_toolset = registry.get_toolset_for_tool(tool_name_prefixed)
        if existing_toolset and not existing_toolset.startswith("mcp-"):
            logger.warning(
                "MCP server '%s': tool '%s' (→ '%s') collides with built-in "
                "tool in toolset '%s' — skipping to preserve built-in",
                name, mcp_tool.name, tool_name_prefixed, existing_toolset,
            )
            continue

        registry.register(
            name=tool_name_prefixed,
            toolset=toolset_name,
            schema=schema,
            handler=_make_tool_handler(name, mcp_tool.name, server.tool_timeout),
            check_fn=_make_check_fn(name),
            is_async=False,
            description=schema["description"],
        )
        _track_mcp_tool_server(tool_name_prefixed, name)
        registered_names.append(tool_name_prefixed)

    # Register MCP Resources & Prompts utility tools, filtered by config and
    # only when the server actually supports the corresponding capability.
    _handler_factories = {
        "list_resources": _make_list_resources_handler,
        "read_resource": _make_read_resource_handler,
        "list_prompts": _make_list_prompts_handler,
        "get_prompt": _make_get_prompt_handler,
    }
    check_fn = _make_check_fn(name)
    for entry in _select_utility_schemas(name, server, config):
        schema = entry["schema"]
        handler_key = entry["handler_key"]
        handler = _handler_factories[handler_key](name, server.tool_timeout)
        util_name = schema["name"]

        # Same collision guard for utility tools.
        existing_toolset = registry.get_toolset_for_tool(util_name)
        if existing_toolset and not existing_toolset.startswith("mcp-"):
            logger.warning(
                "MCP server '%s': utility tool '%s' collides with built-in "
                "tool in toolset '%s' — skipping to preserve built-in",
                name, util_name, existing_toolset,
            )
            continue

        registry.register(
            name=util_name,
            toolset=toolset_name,
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            is_async=False,
            description=schema["description"],
        )
        _track_mcp_tool_server(util_name, name)
        registered_names.append(util_name)

    if registered_names:
        registry.register_toolset_alias(name, toolset_name)

    return registered_names


async def _discover_and_register_server(name: str, config: dict) -> List[str]:
    """Connect to a single MCP server, discover tools, and register them.

    Returns list of registered tool names.
    """
    connect_timeout = config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
    server = await asyncio.wait_for(
        _connect_server(name, config),
        timeout=connect_timeout,
    )
    with _lock:
        _server_connecting.discard(name)
        _server_connect_errors.pop(name, None)
        _servers[name] = server

    registered_names = _register_server_tools(name, server, config)
    server._registered_tool_names = list(registered_names)

    transport_type = "HTTP" if "url" in config else "stdio"
    logger.info(
        "MCP server '%s' (%s): registered %d tool(s): %s",
        name, transport_type, len(registered_names),
        ", ".join(registered_names),
    )
    return registered_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_mcp_servers(servers: Dict[str, dict]) -> List[str]:
    """Connect to explicit MCP servers and register their tools.

    Idempotent for already-connected server names. Servers with
    ``enabled: false`` are skipped without disconnecting existing sessions.

    Args:
        servers: Mapping of ``{server_name: server_config}``.

    Returns:
        List of all currently registered MCP tool names.
    """
    if not _MCP_AVAILABLE:
        logger.debug("MCP SDK not available -- skipping explicit MCP registration")
        return []

    servers = _filter_suspicious_mcp_servers(servers)
    if not servers:
        logger.debug("No explicit MCP servers provided")
        return []

    # Only attempt servers that aren't already connected and are enabled
    # (enabled: false skips the server entirely without removing its config)
    with _lock:
        new_servers = {
            k: v
            for k, v in servers.items()
            if k not in _servers and _parse_boolish(v.get("enabled", True), default=True)
        }
        # Cached entries with no live session are parked or mid-reconnect.
        # Their tools are deregistered, so nothing else can reach
        # _signal_reconnect — without this nudge a new session silently
        # waits up to _PARKED_RETRY_INTERVAL for the next self-probe
        # (#50170). Wake them now so their tools come back promptly.
        stale_cached = [
            _servers[k]
            for k in servers
            if k in _servers and getattr(_servers[k], "session", None) is None
        ]
        _server_connecting.update(new_servers)
        for srv_name in new_servers:
            _server_connect_errors.pop(srv_name, None)
        # Track which servers opt-in to parallel tool calls (idempotent).
        for srv_name, srv_cfg in servers.items():
            if _parse_boolish(srv_cfg.get("supports_parallel_tool_calls", False), default=False):
                _parallel_safe_servers.add(sanitize_mcp_name_component(srv_name))
            else:
                _parallel_safe_servers.discard(sanitize_mcp_name_component(srv_name))

    for srv in stale_cached:
        _signal_reconnect(srv)

    if not new_servers:
        return _existing_tool_names()

    # Start the background event loop for MCP connections
    _ensure_mcp_loop()

    async def _discover_one(name: str, cfg: dict) -> List[str]:
        """Connect to a single server and return its registered tool names."""
        return await _discover_and_register_server(name, cfg)

    async def _discover_all():
        server_names = list(new_servers.keys())
        # Connect to all servers in PARALLEL
        results = await asyncio.gather(
            *(_discover_one(name, cfg) for name, cfg in new_servers.items()),
            return_exceptions=True,
        )
        for name, result in zip(server_names, results):
            if isinstance(result, BaseException):
                command = new_servers.get(name, {}).get("command")
                message = _format_connect_error(result)
                with _lock:
                    _server_connecting.discard(name)
                    _server_connect_errors[name] = message
                logger.warning(
                    "Failed to connect to MCP server '%s'%s: %s",
                    name,
                    f" (command={command})" if command else "",
                    message,
                )
            else:
                with _lock:
                    _server_connecting.discard(name)
                    _server_connect_errors.pop(name, None)

    # Per-server timeouts are handled inside _discover_and_register_server.
    # The outer timeout is generous: 120s total for parallel discovery.
    #
    # Temporarily clear the interrupt flag on the current thread so that MCP
    # discovery is never cancelled by a stale interrupt from a prior agent
    # session (executor threads get reused and may carry old interrupt state).
    from tools.interrupt import is_interrupted as _is_interrupted, set_interrupt as _set_interrupt
    _was_interrupted = _is_interrupted()
    if _was_interrupted:
        _set_interrupt(False)
    try:
        _run_on_mcp_loop(_discover_all, timeout=120)
    finally:
        if _was_interrupted:
            _set_interrupt(True)

    # Log a summary so ACP callers get visibility into what was registered.
    with _lock:
        connected = [n for n in new_servers if n in _servers]
        new_tool_count = sum(
            len(getattr(_servers[n], "_registered_tool_names", []))
            for n in connected
        )
    failed = len(new_servers) - len(connected)
    if new_tool_count or failed:
        summary = f"MCP: registered {new_tool_count} tool(s) from {len(connected)} server(s)"
        if failed:
            summary += f" ({failed} failed)"
        logger.info(summary)

    return _existing_tool_names()


def discover_mcp_tools() -> List[str]:
    """Entry point: load config, connect to MCP servers, register tools.

    Called from ``model_tools`` after ``discover_builtin_tools()``. Safe to call even when
    the ``mcp`` package is not installed (returns empty list).

    Idempotent for already-connected servers. If some servers failed on a
    previous call, only the missing ones are retried.

    Returns:
        List of all registered MCP tool names.
    """
    if not _MCP_AVAILABLE:
        logger.debug("MCP SDK not available -- skipping MCP tool discovery")
        return []

    servers = _load_mcp_config()
    if not servers:
        logger.debug("No MCP servers configured")
        return []

    with _lock:
        new_server_names = [
            name
            for name, cfg in servers.items()
            if name not in _servers and _parse_boolish(cfg.get("enabled", True), default=True)
        ]

    tool_names = register_mcp_servers(servers)
    if not new_server_names:
        return tool_names

    with _lock:
        connected_server_names = [name for name in new_server_names if name in _servers]
        new_tool_count = sum(
            len(getattr(_servers[name], "_registered_tool_names", []))
            for name in connected_server_names
        )

    failed_count = len(new_server_names) - len(connected_server_names)
    if new_tool_count or failed_count:
        summary = f"  MCP: {new_tool_count} tool(s) from {len(connected_server_names)} server(s)"
        if failed_count:
            summary += f" ({failed_count} failed)"
        logger.info(summary)

    return tool_names


def is_mcp_tool_parallel_safe(tool_name: str) -> bool:
    """Check if an MCP tool belongs to a server that supports parallel tool calls.

    MCP tool names follow the pattern ``mcp__{server}__{tool}``, but that
    string shape is ambiguous when server names contain underscores. Use the
    exact server provenance captured at registration time rather than prefix
    matching, then check whether that server's config includes
    ``supports_parallel_tool_calls: true``.

    Returns False for non-MCP tools or tools from servers without the flag.
    """
    if not tool_name.startswith(MCP_TOOL_NAME_PREFIX):
        return False
    with _lock:
        server_name = _mcp_tool_server_names.get(tool_name)
        return bool(server_name and server_name in _parallel_safe_servers)


def get_mcp_status() -> List[dict]:
    """Return status of all configured MCP servers for banner display.

    Returns a list of dicts with keys: name, transport, tools, connected,
    disabled, and status. Includes connected servers, disabled servers,
    in-flight connection attempts, recorded failures, and servers that are
    configured but have not been started in this process yet.
    """
    result: List[dict] = []

    # Get configured servers from config
    configured = _load_mcp_config()
    if not configured:
        return result

    with _lock:
        active_servers = dict(_servers)
        connecting = set(_server_connecting)
        connect_errors = dict(_server_connect_errors)

    for name, cfg in configured.items():
        transport = cfg.get("transport", "http") if "url" in cfg else "stdio"
        enabled = _parse_boolish(cfg.get("enabled", True), default=True)
        server = active_servers.get(name)
        if server and server.session is not None:
            entry = {
                "name": name,
                "transport": transport,
                "tools": len(server._registered_tool_names) if hasattr(server, "_registered_tool_names") else len(server._tools),
                "connected": True,
                "disabled": False,
                "status": "connected",
            }
            if server._sampling:
                entry["sampling"] = dict(server._sampling.metrics)
            result.append(entry)
        elif not enabled:
            # A server with enabled: false is intentionally not connected — it is
            # disabled, not failed. Surface that distinction so consumers (banner,
            # TUI) can render "disabled" rather than an alarming "failed".
            result.append({
                "name": name,
                "transport": transport,
                "tools": 0,
                "connected": False,
                "disabled": True,
                "status": "disabled",
            })
        elif name in connecting:
            result.append({
                "name": name,
                "transport": transport,
                "tools": 0,
                "connected": False,
                "disabled": False,
                "status": "connecting",
            })
        elif name in connect_errors:
            result.append({
                "name": name,
                "transport": transport,
                "tools": 0,
                "connected": False,
                "disabled": False,
                "status": "failed",
                "error": connect_errors[name],
            })
        else:
            result.append({
                "name": name,
                "transport": transport,
                "tools": 0,
                "connected": False,
                "disabled": False,
                "status": "configured",
            })

    return result


def probe_mcp_server_tools() -> Dict[str, List[tuple]]:
    """Temporarily connect to configured MCP servers and list their tools.

    Designed for ``hermes tools`` interactive configuration — connects to each
    enabled server, grabs tool names and descriptions, then disconnects.
    Does NOT register tools in the Hermes registry.

    Returns:
        Dict mapping server name to list of (tool_name, description) tuples.
        Servers that fail to connect are omitted from the result.
    """
    if not _MCP_AVAILABLE:
        return {}

    servers_config = _load_mcp_config()
    if not servers_config:
        return {}

    enabled = {
        k: v for k, v in servers_config.items()
        if _parse_boolish(v.get("enabled", True), default=True)
    }
    if not enabled:
        return {}

    _ensure_mcp_loop()

    result: Dict[str, List[tuple]] = {}
    probed_servers: List[MCPServerTask] = []

    async def _probe_all():
        names = list(enabled.keys())
        coros = []
        for name, cfg in enabled.items():
            ct = cfg.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
            coros.append(asyncio.wait_for(_connect_server(name, cfg), timeout=ct))

        outcomes = await asyncio.gather(*coros, return_exceptions=True)

        for name, outcome in zip(names, outcomes):
            if isinstance(outcome, Exception):
                logger.debug("Probe: failed to connect to '%s': %s", name, outcome)
                continue
            probed_servers.append(outcome)
            tools = []
            for t in outcome._tools:
                desc = getattr(t, "description", "") or ""
                tools.append((t.name, desc))
            result[name] = tools

        # Shut down all probed connections
        await asyncio.gather(
            *(s.shutdown() for s in probed_servers),
            return_exceptions=True,
        )

    try:
        _run_on_mcp_loop(_probe_all, timeout=120)
    except Exception as exc:
        logger.debug("MCP probe failed: %s", exc)
    finally:
        _stop_mcp_loop_if_idle()

    return result


# Serializes in-place mutation of an agent's tool snapshot.  The reload RPC,
# the gateway reload, and the late-binding refresh thread all swap
# ``agent.tools`` / ``agent.valid_tool_names`` after the agent was built; the
# agent's run loop reads those during tool iteration, so a concurrent write
# mid-read could otherwise expose a half-updated list.
_agent_tools_lock = threading.Lock()


def has_registered_mcp_tools() -> bool:
    """True if any MCP server has actually registered tools into the registry.

    Cheap — checks the global MCP-tool→server name map under ``_lock``, no
    registry walk.  Used by the per-turn refresh hook so a session with no MCP
    tools (the common case, and also a connected-but-zero-tool/prompt-only
    server) skips the ``get_tool_definitions`` rebuild entirely.  Checks
    registered TOOLS, not connected servers, so a server that registers no tools
    doesn't keep the hook firing every turn.
    """
    with _lock:
        return bool(_mcp_tool_server_names)


def refresh_agent_mcp_tools(
    agent,
    *,
    enabled_override=None,
    disabled_override=None,
    quiet_mode: bool = True,
) -> set:
    """Re-derive an already-built agent's tool snapshot from the live registry.

    The agent snapshots ``agent.tools`` once at build time and never re-reads
    the registry (see ``run_agent`` / ``agent_init``).  When MCP servers connect
    *after* that snapshot — a slow HTTP/OAuth server that misses the bounded
    startup wait, or a ``/reload-mcp`` — their tools are invisible until the
    snapshot is rebuilt.  This is the single shared rebuild used by every such
    caller (the TUI ``reload.mcp`` RPC, the gateway reload, the late-binding
    refresh thread, and the per-turn between-turns refresh) so they can't drift
    apart again.

    The rebuild respects the agent's own ``enabled_toolsets`` /
    ``disabled_toolsets`` (the same filtering it was built with) and diffs by
    tool **name** (not count — a count compare misses an equal-size add/remove
    swap).

    Crucially it is **additive-preserving**: ``get_tool_definitions`` returns
    only the registry-derived tools, but ``agent_init`` appends two further
    families directly onto ``agent.tools`` *after* that — external
    memory-provider tools (mem0/honcho/…) and context-engine tools
    (``lcm_*``).  A naive ``agent.tools = get_tool_definitions(...)`` would
    silently DELETE those.  So after rebuilding the registry set we re-run the
    same post-build injectors ``agent_init`` used, reconstructing the full
    surface.  The new ``(tools, valid_tool_names)`` pair is published together
    under ``_agent_tools_lock`` so a concurrent reader never sees a
    cross-attribute half-swap.

    Returns the set of newly-added tool names (empty when nothing changed), so
    callers can decide whether to notify the user / re-emit session info.  The
    caller owns the prompt-cache contract: this helper does NOT check turn state,
    because each caller has a different policy (``/reload-mcp`` rebuilds after
    explicit user consent; the late-binding and between-turns paths only rebuild
    at a turn boundary, before that turn's ``tools=`` prefix is assembled).
    """
    from model_tools import get_tool_definitions
    from tools.registry import registry

    # Explicit reloads (/reload-mcp) pass freshly-resolved toolsets so a server
    # the user just ENABLED in config is picked up; the agent's stored selection
    # is then updated to match. The automatic paths (between-turns, late-binding)
    # pass nothing and reuse the agent's build-time selection unchanged.
    if enabled_override is not None or disabled_override is not None:
        enabled = enabled_override if enabled_override is not None else getattr(agent, "enabled_toolsets", None)
        disabled = disabled_override if disabled_override is not None else getattr(agent, "disabled_toolsets", None)
        agent.enabled_toolsets = enabled
        agent.disabled_toolsets = disabled
    else:
        enabled = getattr(agent, "enabled_toolsets", None)
        disabled = getattr(agent, "disabled_toolsets", None)

    # Capture the registry generation this rebuild is derived from BEFORE the
    # (potentially slow) get_tool_definitions call. Used at publish time to
    # reject a stale write: if two callers race (e.g. the late-refresh daemon
    # and the between-turns prologue around turn 1), a slower caller that
    # computed an OLDER set must not clobber a newer set another caller already
    # published. ``registry._generation`` bumps on every (de)register.
    snapshot_generation = registry._generation

    # Registry-derived tools (built-ins + MCP), filtered to the agent's toolsets.
    # Computed OUTSIDE the lock (get_tool_definitions can be slow); the diff and
    # publish below happen together in ONE critical section so two concurrent
    # callers can't torn-publish or compute overlapping ``added`` sets.
    new_defs = list(
        get_tool_definitions(
            enabled_toolsets=enabled,
            disabled_toolsets=disabled,
            quiet_mode=quiet_mode,
        )
        or []
    )
    new_names = {t["function"]["name"] for t in new_defs}

    # Re-append the post-build injected families that get_tool_definitions does
    # NOT reproduce, so a refresh never strips them (memory-provider + context-
    # engine tools). Staged entirely on LOCALS — the live ``agent.tools`` /
    # ``valid_tool_names`` / ``_context_engine_tool_names`` are never touched
    # until the single atomic publish below, so a concurrent reader
    # (``build_api_kwargs``) can't see a partial rebuild or a cross-attribute
    # half-swap. ``staged_engine_names`` are the context-engine routing names
    # this rebuild actually appended (matching agent_init's dedup-aware add).
    staged_engine_names = _reinject_post_build_tools(agent, new_defs, new_names)

    # Single atomic read-diff-publish so the returned ``added`` is consistent
    # with what was actually published, even under concurrent callers, and a
    # stale (older-generation) rebuild can't overwrite a newer published one.
    with _agent_tools_lock:
        # Defensive: the published generation should be an int, but tolerate an
        # agent that never set it (or set a non-int, e.g. a test mock) rather
        # than throwing TypeError on the comparison and silently failing the
        # whole refresh.
        published_gen_raw = getattr(agent, "_tool_snapshot_generation", -1)
        published_gen = published_gen_raw if isinstance(published_gen_raw, int) else -1
        if snapshot_generation < published_gen:
            # A newer snapshot already won; our set is stale — drop it.
            return set()
        current = {
            t["function"]["name"]
            for t in (getattr(agent, "tools", None) or [])
        }
        if new_names == current:
            # No change → leave the live snapshot untouched (no churn), but
            # record the generation so an in-flight older caller can't clobber.
            agent._tool_snapshot_generation = max(published_gen, snapshot_generation)
            return set()
        agent.tools = new_defs
        agent.valid_tool_names = new_names
        # Publish context-engine routing names atomically with the snapshot.
        engine_names = getattr(agent, "_context_engine_tool_names", None)
        if isinstance(engine_names, set):
            engine_names.clear()
            engine_names.update(staged_engine_names)
        agent._tool_snapshot_generation = max(published_gen, snapshot_generation)
        return new_names - current


def _reinject_post_build_tools(agent, tools_list: list, name_set: set) -> set:
    """Append memory-provider and context-engine tools onto staged locals.

    Mirrors the post-``get_tool_definitions`` injection in ``agent_init`` so a
    snapshot rebuild reconstructs the FULL tool surface, not just the
    registry-derived subset. Operates ONLY on the caller's staged ``tools_list``
    / ``name_set`` (never the live agent attributes) so the rebuild stays atomic.
    Idempotent (skips names already present) and fail-soft.

    Returns the set of context-engine routing names actually appended by THIS
    rebuild — matching ``agent_init``'s dedup behavior (a name already provided
    by a registry/plugin tool is NOT claimed for context-engine routing). The
    caller publishes this into ``agent._context_engine_tool_names`` atomically
    with the snapshot.
    """
    def _add(schema: dict) -> bool:
        name = schema.get("name", "")
        if not name or name in name_set:
            return False
        tools_list.append({"type": "function", "function": schema})
        name_set.add(name)
        return True

    # Memory-provider tools (mem0/honcho/byterover/supermemory/…).
    try:
        memory_manager = getattr(agent, "_memory_manager", None)
        get_mem_schemas = getattr(memory_manager, "get_all_tool_schemas", None) if memory_manager else None
        if callable(get_mem_schemas):
            # Honor the same enablement gate inject_memory_provider_tools uses.
            from agent.memory_manager import memory_provider_tools_enabled
            if "memory" in name_set or memory_provider_tools_enabled(getattr(agent, "enabled_toolsets", None)):
                for schema in get_mem_schemas():
                    if isinstance(schema, dict):
                        _add(schema)
    except Exception:
        logger.debug("Memory-provider tool re-injection skipped", exc_info=True)

    # Context-engine tools (lcm_grep/lcm_describe/…) — the `context_engine`
    # toolset is intentionally empty, so these only exist via this append.
    # Honor the same enabled_toolsets gate agent_init uses (#5544): without it a
    # restricted-toolset platform (e.g. platform_toolsets: telegram: []) would
    # re-leak lcm_* tools the build deliberately excluded, and pay the local-
    # model latency penalty.
    staged_engine_names: set = set()
    try:
        enabled = getattr(agent, "enabled_toolsets", None)
        context_engine_allowed = enabled is None or "context_engine" in enabled
        compressor = getattr(agent, "context_compressor", None)
        get_schemas = getattr(compressor, "get_tool_schemas", None) if compressor else None
        if context_engine_allowed and callable(get_schemas):
            for schema in get_schemas():
                if not isinstance(schema, dict):
                    continue
                name = schema.get("name", "")
                # Only claim the routing name when WE appended the schema, so a
                # name already owned by a registry/plugin tool keeps its own
                # dispatch (matches agent_init.py's `continue`-before-claim).
                if _add(schema) and name:
                    staged_engine_names.add(name)
    except Exception:
        logger.debug("Context-engine tool re-injection skipped", exc_info=True)

    return staged_engine_names


def shutdown_mcp_servers():
    """Close all MCP server connections and stop the background loop.

    Each server Task is signalled to exit its ``async with`` block so that
    the anyio cancel-scope cleanup happens in the same Task that opened it.
    All servers are shut down in parallel via ``asyncio.gather``.
    """
    with _lock:
        servers_snapshot = list(_servers.values())

    # Fast path: nothing to shut down.
    if not servers_snapshot:
        _stop_mcp_loop()
        return

    async def _shutdown():
        results = await asyncio.gather(
            *(server.shutdown() for server in servers_snapshot),
            return_exceptions=True,
        )
        for server, result in zip(servers_snapshot, results):
            if isinstance(result, Exception):
                logger.debug(
                    "Error closing MCP server '%s': %s", server.name, result,
                )
        with _lock:
            _servers.clear()

    with _lock:
        loop = _mcp_loop
    if loop is not None and loop.is_running():
        from agent.async_utils import safe_schedule_threadsafe
        future = safe_schedule_threadsafe(
            _shutdown(), loop,
            logger=logger,
            log_message="MCP shutdown: failed to schedule",
        )
        if future is not None:
            try:
                future.result(timeout=15)
            except BaseException as exc:
                logger.debug("Error during MCP shutdown: %s", exc)

    _stop_mcp_loop()


def _kill_orphaned_mcp_children(include_active: bool = False) -> None:
    """Best-effort graceful shutdown of stdio MCP subprocesses to reap orphans.

    Orphans are PIDs that survived their session context exit (SDK teardown
    did not terminate the process — common on Linux when stdio children escape
    the parent cgroup on cancellation). By default only entries in
    ``_orphan_stdio_pids`` are reaped so concurrent cron jobs and live user
    sessions are not disrupted.

    Sends SIGTERM, waits 2 seconds, then escalates to SIGKILL for any
    survivors, avoiding shared-resource collisions when multiple hermes
    processes run on the same host (each has its own ``_stdio_pids`` dict).

    On POSIX, signals are sent via ``os.killpg`` to the spawn-time pgid when
    one is tracked, so reparented grandchildren in the same process group
    (e.g. ``claude mcp serve`` spawned by a stdio MCP wrapper that exited
    first) are reaped alongside the direct child.  Falls back to ``os.kill``
    on Windows and when no pgid is recorded.

    With ``include_active=True`` also kills every PID in ``_stdio_pids`` —
    used only at final shutdown, after the MCP event loop has stopped and no
    sessions can still be in flight.
    """
    import signal as _signal

    with _lock:
        pids: Dict[int, str] = {}
        for opid in _orphan_stdio_pids:
            pids[opid] = "orphan"
        _orphan_stdio_pids.clear()
        if include_active:
            pids.update(dict(_stdio_pids))
            _stdio_pids.clear()
        # Snapshot pgids for the pids we're about to kill, then drop the
        # entries so a future spawn can't collide with stale state.
        pgids: Dict[int, int] = {pid: _stdio_pgids[pid] for pid in pids if pid in _stdio_pgids}
        for pid in pgids:
            _stdio_pgids.pop(pid, None)

    # Fast path: no tracked stdio PIDs to reap. Skip the SIGTERM/sleep/SIGKILL
    # dance entirely — otherwise every MCP-free shutdown pays a 2s sleep tax.
    if not pids:
        return

    # Pre-compute the gateway's own pgid so _send_signal can avoid killing it.
    try:
        _my_pgid = os.getpgrp()
    except (AttributeError, OSError):
        _my_pgid = None  # Windows or restricted environment

    def _send_signal(pid: int, sig: int, server_name: str) -> None:
        """SIGTERM/SIGKILL via pgroup on POSIX, fall back to pid signal."""
        pgid = pgids.get(pid)
        killpg = getattr(os, "killpg", None)
        if pgid is not None and killpg is not None:
            if _my_pgid is not None and pgid == _my_pgid:
                # The MCP child shares the gateway's own process group.
                # Using killpg would deliver the signal to the gateway as
                # well, crashing it (see #47134).  Fall through to the
                # per-pid kill() path instead. Warn because per-pid kill
                # cannot reach grandchildren in this shared group — if the
                # direct child has already exited, they may leak (inherent:
                # group-killing them would also kill the gateway).
                logger.warning(
                    "MCP server '%s' pgid %d matches gateway pgid; skipping "
                    "killpg to avoid self-kill and using per-pid kill — any "
                    "grandchildren in this group may not be reaped",
                    server_name, pgid,
                )
            else:
                try:
                    killpg(pgid, sig)
                    return
                except (ProcessLookupError, PermissionError, OSError) as exc:
                    # Pgroup gone (all members exited) or refused — fall back to
                    # the per-pid path so we still try the direct child if alive.
                    logger.debug(
                        "killpg(%d, %d) failed for MCP server '%s': %s; falling back to kill(pid)",
                        pgid, sig, server_name, exc,
                    )
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    # Phase 1: SIGTERM (graceful)
    for pid, server_name in pids.items():
        _send_signal(pid, _signal.SIGTERM, server_name)
        logger.debug("Sent SIGTERM to orphaned MCP process %d (%s)", pid, server_name)

    # Phase 2: Wait for graceful exit
    time.sleep(2)

    # Phase 3: SIGKILL any survivors
    _sigkill = getattr(_signal, "SIGKILL", _signal.SIGTERM)
    # ``os.kill(pid, 0)`` is NOT a no-op on Windows. Use the cross-platform
    # existence check before escalating to SIGKILL.
    from gateway.status import _pid_exists
    for pid, server_name in pids.items():
        if not _pid_exists(pid):
            continue  # Good — exited after SIGTERM
        _send_signal(pid, _sigkill, server_name)
        logger.warning(
            "Force-killed MCP process %d (%s) after SIGTERM timeout",
            pid, server_name,
        )


def _stop_mcp_loop_if_idle() -> bool:
    """Stop the MCP loop only when no registered server still owns it.

    Probe paths create temporary MCPServerTask instances that are not placed in
    ``_servers``.  They should clean up an otherwise-idle loop, but must not
    tear down the process-global loop when live agent tools are registered on
    it.  Otherwise a dashboard/CLI probe can make later MCP tool calls fail
    with ``MCP event loop is not running``.
    """
    return _stop_mcp_loop(only_if_idle=True)


def _stop_mcp_loop(*, only_if_idle: bool = False) -> bool:
    """Stop the background event loop and join its thread."""
    global _mcp_loop, _mcp_thread
    with _lock:
        if only_if_idle and (_servers or _server_connecting):
            logger.debug("Leaving MCP event loop running; active servers are registered or connecting")
            return False
        loop = _mcp_loop
        thread = _mcp_thread
        _mcp_loop = None
        _mcp_thread = None
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        try:
            loop.close()
        except Exception:
            pass
        # After closing the loop, any stdio subprocesses that survived the
        # graceful shutdown are now orphaned — include active PIDs too
        # since the loop is gone and no session can still be in flight.
        _kill_orphaned_mcp_children(include_active=True)
    return True
