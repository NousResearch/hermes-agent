#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Client Support

Connects to external MCP servers via stdio or HTTP/StreamableHTTP transport,
discovers their tools, and registers them into the hermes-agent tool registry
so the agent can call them like any built-in tool.

Configuration is read from ~/.hermes/config.yaml under the ``mcp_servers`` key.
The ``mcp`` Python package is optional -- if not installed, this module is a
no-op and logs a debug message.

Example config::

    mcp_servers:
      filesystem:
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        env: {}
        timeout: 120         # per-tool-call timeout in seconds (default: 120)
        connect_timeout: 60  # initial connection timeout (default: 60)
      github:
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-github"]
        env:
          GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
      remote_api:
        url: "https://my-mcp-server.example.com/mcp"
        headers:
          Authorization: "Bearer sk-..."
        timeout: 180
      analysis:
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
    - Automatic reconnection with exponential backoff (up to 5 retries)
    - Environment variable filtering for stdio subprocesses (security)
    - Credential stripping in error messages returned to the LLM
    - Configurable per-server timeouts for tool calls and connections
    - Thread-safe architecture with dedicated background event loop
    - Sampling support: MCP servers can request LLM completions via
      sampling/createMessage (text and tool-use responses)

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
import json
import logging
import math
import os
import re
import shutil
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful import -- MCP SDK is an optional dependency
# ---------------------------------------------------------------------------

_MCP_AVAILABLE = False
_MCP_HTTP_AVAILABLE = False
_MCP_SAMPLING_TYPES = False
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _MCP_AVAILABLE = True
    try:
        from mcp.client.streamable_http import streamablehttp_client
        _MCP_HTTP_AVAILABLE = True
    except ImportError:
        _MCP_HTTP_AVAILABLE = False
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
except ImportError:
    logger.debug("mcp package not installed -- MCP tool support disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TOOL_TIMEOUT = 120      # seconds for tool calls
_DEFAULT_CONNECT_TIMEOUT = 60    # seconds for initial connection per server
_MAX_RECONNECT_RETRIES = 5
_MAX_BACKOFF_SECONDS = 60

# Environment variables that are safe to pass to stdio subprocesses
_SAFE_ENV_KEYS = frozenset({
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR",
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
        if key in _SAFE_ENV_KEYS or key.startswith("XDG_"):
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
            ]
            for candidate in candidates:
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    resolved_command = candidate
                    break

    command_dir = os.path.dirname(resolved_command)
    if command_dir:
        resolved_env = _prepend_path(resolved_env, command_dir)

    return resolved_command, resolved_env


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
                            "arguments": json.dumps(tu.input) if isinstance(tu.input, dict) else str(tu.input),
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
                        "parameters": getattr(t, "inputSchema", {}) or {},
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
                f"Sampling LLM call failed: {_sanitize_error(str(exc))}"
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
        "_task", "_ready", "_shutdown_event", "_tools", "_error", "_config",
        "_sampling", "_registered_tool_names",
    )

    def __init__(self, name: str):
        self.name = name
        self.session: Optional[Any] = None
        self.tool_timeout: float = _DEFAULT_TOOL_TIMEOUT
        self._task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._tools: list = []
        self._error: Optional[Exception] = None
        self._config: dict = {}
        self._sampling: Optional[SamplingHandler] = None
        self._registered_tool_names: list[str] = []

    def _is_http(self) -> bool:
        """Check if this server uses HTTP transport."""
        return "url" in self._config

    async def _run_stdio(self, config: dict):
        """Run the server using stdio transport."""
        command = config.get("command")
        args = config.get("args", [])
        user_env = config.get("env")

        if not command:
            raise ValueError(
                f"MCP server '{self.name}' has no 'command' in config"
            )

        safe_env = _build_safe_env(user_env)
        command, safe_env = _resolve_stdio_command(command, safe_env)
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=safe_env if safe_env else None,
        )

        sampling_kwargs = self._sampling.session_kwargs() if self._sampling else {}
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream, **sampling_kwargs) as session:
                await session.initialize()
                self.session = session
                await self._discover_tools()
                self._ready.set()
                await self._shutdown_event.wait()

    async def _run_http(self, config: dict):
        """Run the server using HTTP/StreamableHTTP transport."""
        if not _MCP_HTTP_AVAILABLE:
            raise ImportError(
                f"MCP server '{self.name}' requires HTTP transport but "
                "mcp.client.streamable_http is not available. "
                "Upgrade the mcp package to get HTTP support."
            )

        url = config["url"]
        headers = config.get("headers")
        connect_timeout = config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)

        sampling_kwargs = self._sampling.session_kwargs() if self._sampling else {}
        async with streamablehttp_client(
            url,
            headers=headers,
            timeout=float(connect_timeout),
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(read_stream, write_stream, **sampling_kwargs) as session:
                await session.initialize()
                self.session = session
                await self._discover_tools()
                self._ready.set()
                await self._shutdown_event.wait()

    async def _discover_tools(self):
        """Discover tools from the connected session."""
        if self.session is None:
            return
        tools_result = await self.session.list_tools()
        self._tools = (
            tools_result.tools
            if hasattr(tools_result, "tools")
            else []
        )

    async def run(self, config: dict):
        """Long-lived coroutine: connect, discover tools, wait, disconnect.

        Includes automatic reconnection with exponential backoff if the
        connection drops unexpectedly (unless shutdown was requested).
        """
        self._config = config
        self.tool_timeout = config.get("timeout", _DEFAULT_TOOL_TIMEOUT)

        # Set up sampling handler if enabled and SDK types are available
        sampling_config = config.get("sampling", {})
        if sampling_config.get("enabled", True) and _MCP_SAMPLING_TYPES:
            self._sampling = SamplingHandler(self.name, sampling_config)
        else:
            self._sampling = None

        # Validate: warn if both url and command are present
        if "url" in config and "command" in config:
            logger.warning(
                "MCP server '%s' has both 'url' and 'command' in config. "
                "Using HTTP transport ('url'). Remove 'command' to silence "
                "this warning.",
                self.name,
            )
        retries = 0
        backoff = 1.0

        while True:
            try:
                if self._is_http():
                    await self._run_http(config)
                else:
                    await self._run_stdio(config)
                # Normal exit (shutdown requested) -- break out
                break
            except Exception as exc:
                self.session = None

                # If this is the first connection attempt, report the error
                if not self._ready.is_set():
                    self._error = exc
                    self._ready.set()
                    return

                # If shutdown was requested, don't reconnect
                if self._shutdown_event.is_set():
                    logger.debug(
                        "MCP server '%s' disconnected during shutdown: %s",
                        self.name, exc,
                    )
                    return

                retries += 1
                if retries > _MAX_RECONNECT_RETRIES:
                    logger.warning(
                        "MCP server '%s' failed after %d reconnection attempts, "
                        "giving up: %s",
                        self.name, _MAX_RECONNECT_RETRIES, exc,
                    )
                    return

                logger.warning(
                    "MCP server '%s' connection lost (attempt %d/%d), "
                    "reconnecting in %.0fs: %s",
                    self.name, retries, _MAX_RECONNECT_RETRIES,
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
        self.session = None


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_servers: Dict[str, MCPServerTask] = {}

# Dedicated event loop running in a background daemon thread.
_mcp_loop: Optional[asyncio.AbstractEventLoop] = None
_mcp_thread: Optional[threading.Thread] = None

# Protects _mcp_loop, _mcp_thread, and _servers from concurrent access.
_lock = threading.Lock()


def _ensure_mcp_loop():
    """Start the background event loop thread if not already running."""
    global _mcp_loop, _mcp_thread
    with _lock:
        if _mcp_loop is not None and _mcp_loop.is_running():
            return
        _mcp_loop = asyncio.new_event_loop()
        _mcp_thread = threading.Thread(
            target=_mcp_loop.run_forever,
            name="mcp-event-loop",
            daemon=True,
        )
        _mcp_thread.start()


def _run_on_mcp_loop(coro, timeout: float = 30):
    """Schedule a coroutine on the MCP event loop and block until done."""
    with _lock:
        loop = _mcp_loop
    if loop is None or not loop.is_running():
        raise RuntimeError("MCP event loop is not running")
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_mcp_config() -> Dict[str, dict]:
    """Read ``mcp_servers`` from the Hermes config file.

    Returns a dict of ``{server_name: server_config}`` or empty dict.
    Server config can contain either ``command``/``args``/``env`` for stdio
    transport or ``url``/``headers`` for HTTP transport, plus optional
    ``timeout`` and ``connect_timeout`` overrides.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        servers = config.get("mcp_servers")
        if not servers or not isinstance(servers, dict):
            return {}
        return servers
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
        with _lock:
            server = _servers.get(server_name)
        if not server or not server.session:
            return json.dumps({
                "error": f"MCP server '{server_name}' is not connected"
            })

        async def _call():
            result = await server.session.call_tool(tool_name, arguments=args)
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
                })

            # Collect text from content blocks
            parts: List[str] = []
            for block in (result.content or []):
                if hasattr(block, "text"):
                    parts.append(block.text)
            return json.dumps({"result": "\n".join(parts) if parts else ""})

        try:
            return _run_on_mcp_loop(_call(), timeout=tool_timeout)
        except Exception as exc:
            logger.error(
                "MCP tool %s/%s call failed: %s",
                server_name, tool_name, exc,
            )
            return json.dumps({
                "error": _sanitize_error(
                    f"MCP call failed: {type(exc).__name__}: {exc}"
                )
            })

    return _handler


def _make_list_resources_handler(server_name: str, tool_timeout: float):
    """Return a sync handler that lists resources from an MCP server."""

    def _handler(args: dict, **kwargs) -> str:
        with _lock:
            server = _servers.get(server_name)
        if not server or not server.session:
            return json.dumps({
                "error": f"MCP server '{server_name}' is not connected"
            })

        async def _call():
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
            return json.dumps({"resources": resources})

        try:
            return _run_on_mcp_loop(_call(), timeout=tool_timeout)
        except Exception as exc:
            logger.error(
                "MCP %s/list_resources failed: %s", server_name, exc,
            )
            return json.dumps({
                "error": _sanitize_error(
                    f"MCP call failed: {type(exc).__name__}: {exc}"
                )
            })

    return _handler


def _make_read_resource_handler(server_name: str, tool_timeout: float):
    """Return a sync handler that reads a resource by URI from an MCP server."""

    def _handler(args: dict, **kwargs) -> str:
        with _lock:
            server = _servers.get(server_name)
        if not server or not server.session:
            return json.dumps({
                "error": f"MCP server '{server_name}' is not connected"
            })

        uri = args.get("uri")
        if not uri:
            return json.dumps({"error": "Missing required parameter 'uri'"})

        async def _call():
            result = await server.session.read_resource(uri)
            # read_resource returns ReadResourceResult with .contents list
            parts: List[str] = []
            contents = result.contents if hasattr(result, "contents") else []
            for block in contents:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif hasattr(block, "blob"):
                    parts.append(f"[binary data, {len(block.blob)} bytes]")
            return json.dumps({"result": "\n".join(parts) if parts else ""})

        try:
            return _run_on_mcp_loop(_call(), timeout=tool_timeout)
        except Exception as exc:
            logger.error(
                "MCP %s/read_resource failed: %s", server_name, exc,
            )
            return json.dumps({
                "error": _sanitize_error(
                    f"MCP call failed: {type(exc).__name__}: {exc}"
                )
            })

    return _handler


def _make_list_prompts_handler(server_name: str, tool_timeout: float):
    """Return a sync handler that lists prompts from an MCP server."""

    def _handler(args: dict, **kwargs) -> str:
        with _lock:
            server = _servers.get(server_name)
        if not server or not server.session:
            return json.dumps({
                "error": f"MCP server '{server_name}' is not connected"
            })

        async def _call():
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
            return json.dumps({"prompts": prompts})

        try:
            return _run_on_mcp_loop(_call(), timeout=tool_timeout)
        except Exception as exc:
            logger.error(
                "MCP %s/list_prompts failed: %s", server_name, exc,
            )
            return json.dumps({
                "error": _sanitize_error(
                    f"MCP call failed: {type(exc).__name__}: {exc}"
                )
            })

    return _handler


def _make_get_prompt_handler(server_name: str, tool_timeout: float):
    """Return a sync handler that gets a prompt by name from an MCP server."""

    def _handler(args: dict, **kwargs) -> str:
        with _lock:
            server = _servers.get(server_name)
        if not server or not server.session:
            return json.dumps({
                "error": f"MCP server '{server_name}' is not connected"
            })

        name = args.get("name")
        if not name:
            return json.dumps({"error": "Missing required parameter 'name'"})
        arguments = args.get("arguments", {})

        async def _call():
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
            return json.dumps(resp)

        try:
            return _run_on_mcp_loop(_call(), timeout=tool_timeout)
        except Exception as exc:
            logger.error(
                "MCP %s/get_prompt failed: %s", server_name, exc,
            )
            return json.dumps({
                "error": _sanitize_error(
                    f"MCP call failed: {type(exc).__name__}: {exc}"
                )
            })

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

def _convert_mcp_schema(server_name: str, mcp_tool) -> dict:
    """Convert an MCP tool listing to the Hermes registry schema format.

    Args:
        server_name: The logical server name for prefixing.
        mcp_tool:    An MCP ``Tool`` object with ``.name``, ``.description``,
                     and ``.inputSchema``.

    Returns:
        A dict suitable for ``registry.register(schema=...)``.
    """
    # Sanitize: replace hyphens and dots with underscores for LLM API compatibility
    safe_tool_name = mcp_tool.name.replace("-", "_").replace(".", "_")
    safe_server_name = server_name.replace("-", "_").replace(".", "_")
    prefixed_name = f"mcp_{safe_server_name}_{safe_tool_name}"
    return {
        "name": prefixed_name,
        "description": mcp_tool.description or f"MCP tool {mcp_tool.name} from {server_name}",
        "parameters": mcp_tool.inputSchema if mcp_tool.inputSchema else {
            "type": "object",
            "properties": {},
        },
    }


def _sync_mcp_toolsets(server_names: Optional[List[str]] = None) -> None:
    """Expose each MCP server as a standalone toolset and inject into hermes-* sets.

    Creates a real toolset entry in TOOLSETS for each server name (e.g.
    TOOLSETS["github"] = {"tools": ["mcp_github_list_files", ...]}). This
    makes raw server names resolvable in platform_toolsets overrides.

    Also injects all MCP tools into hermes-* umbrella toolsets for the
    default behavior.

    Skips server names that collide with built-in toolsets.
    """
    from toolsets import TOOLSETS

    if server_names is None:
        server_names = list(_load_mcp_config().keys())

    existing = _existing_tool_names()
    all_mcp_tools: List[str] = []

    for server_name in server_names:
        safe_prefix = f"mcp_{server_name.replace('-', '_').replace('.', '_')}_"
        server_tools = sorted(
            t for t in existing if t.startswith(safe_prefix)
        )
        all_mcp_tools.extend(server_tools)

        # Don't overwrite a built-in toolset that happens to share the name.
        existing_ts = TOOLSETS.get(server_name)
        if existing_ts and not str(existing_ts.get("description", "")).startswith("MCP server '"):
            logger.warning(
                "Skipping MCP toolset alias '%s' — a built-in toolset already uses that name",
                server_name,
            )
            continue

        TOOLSETS[server_name] = {
            "description": f"MCP server '{server_name}' tools",
            "tools": server_tools,
            "includes": [],
        }

    # Also inject into hermes-* umbrella toolsets for default behavior.
    for ts_name, ts in TOOLSETS.items():
        if not ts_name.startswith("hermes-"):
            continue
        for tool_name in all_mcp_tools:
            if tool_name not in ts["tools"]:
                ts["tools"].append(tool_name)


def _build_utility_schemas(server_name: str) -> List[dict]:
    """Build schemas for the MCP utility tools (resources & prompts).

    Returns a list of (schema, handler_factory_name) tuples encoded as dicts
    with keys: schema, handler_key.
    """
    safe_name = server_name.replace("-", "_").replace(".", "_")
    return [
        {
            "schema": {
                "name": f"mcp_{safe_name}_list_resources",
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
                "name": f"mcp_{safe_name}_read_resource",
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
                "name": f"mcp_{safe_name}_list_prompts",
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
                "name": f"mcp_{safe_name}_get_prompt",
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


def _select_utility_schemas(server_name: str, server: MCPServerTask, config: dict) -> List[dict]:
    """Select utility schemas based on config and server capabilities."""
    tools_filter = config.get("tools") or {}
    resources_enabled = _parse_boolish(tools_filter.get("resources"), default=True)
    prompts_enabled = _parse_boolish(tools_filter.get("prompts"), default=True)

    selected: List[dict] = []
    for entry in _build_utility_schemas(server_name):
        handler_key = entry["handler_key"]
        if handler_key in {"list_resources", "read_resource"} and not resources_enabled:
            logger.debug("MCP server '%s': skipping utility '%s' (resources disabled)", server_name, handler_key)
            continue
        if handler_key in {"list_prompts", "get_prompt"} and not prompts_enabled:
            logger.debug("MCP server '%s': skipping utility '%s' (prompts disabled)", server_name, handler_key)
            continue

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


async def _discover_and_register_server(name: str, config: dict) -> List[str]:
    """Connect to a single MCP server, discover tools, and register them.

    Also registers utility tools for MCP Resources and Prompts support
    (list_resources, read_resource, list_prompts, get_prompt).

    Returns list of registered tool names.
    """
    from tools.registry import registry
    from toolsets import create_custom_toolset

    connect_timeout = config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
    server = await asyncio.wait_for(
        _connect_server(name, config),
        timeout=connect_timeout,
    )
    with _lock:
        _servers[name] = server

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
        schema = _convert_mcp_schema(name, mcp_tool)
        tool_name_prefixed = schema["name"]

        registry.register(
            name=tool_name_prefixed,
            toolset=toolset_name,
            schema=schema,
            handler=_make_tool_handler(name, mcp_tool.name, server.tool_timeout),
            check_fn=_make_check_fn(name),
            is_async=False,
            description=schema["description"],
        )
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

        registry.register(
            name=schema["name"],
            toolset=toolset_name,
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            is_async=False,
            description=schema["description"],
        )
        registered_names.append(schema["name"])

    server._registered_tool_names = list(registered_names)

    # Create a custom toolset so these tools are discoverable
    if registered_names:
        create_custom_toolset(
            name=toolset_name,
            description=f"MCP tools from {name} server",
            tools=registered_names,
        )

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

def discover_mcp_tools() -> List[str]:
    """Entry point: load config, connect to MCP servers, register tools.

    Called from ``model_tools._discover_tools()``. Safe to call even when
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

    # Only attempt servers that aren't already connected and are enabled
    # (enabled: false skips the server entirely without removing its config)
    with _lock:
        new_servers = {
            k: v
            for k, v in servers.items()
            if k not in _servers and _parse_boolish(v.get("enabled", True), default=True)
        }

    if not new_servers:
        _sync_mcp_toolsets(list(servers.keys()))
        return _existing_tool_names()

    # Start the background event loop for MCP connections
    _ensure_mcp_loop()

    all_tools: List[str] = []
    failed_count = 0

    async def _discover_one(name: str, cfg: dict) -> List[str]:
        """Connect to a single server and return its registered tool names."""
        return await _discover_and_register_server(name, cfg)

    async def _discover_all():
        nonlocal failed_count
        server_names = list(new_servers.keys())
        # Connect to all servers in PARALLEL
        results = await asyncio.gather(
            *(_discover_one(name, cfg) for name, cfg in new_servers.items()),
            return_exceptions=True,
        )
        for name, result in zip(server_names, results):
            if isinstance(result, Exception):
                failed_count += 1
                command = new_servers.get(name, {}).get("command")
                logger.warning(
                    "Failed to connect to MCP server '%s'%s: %s",
                    name,
                    f" (command={command})" if command else "",
                    _format_connect_error(result),
                )
            elif isinstance(result, list):
                all_tools.extend(result)
            else:
                failed_count += 1

    # Per-server timeouts are handled inside _discover_and_register_server.
    # The outer timeout is generous: 120s total for parallel discovery.
    _run_on_mcp_loop(_discover_all(), timeout=120)

    _sync_mcp_toolsets(list(servers.keys()))

    # Print summary
    total_servers = len(new_servers)
    ok_servers = total_servers - failed_count
    if all_tools or failed_count:
        summary = f"  MCP: {len(all_tools)} tool(s) from {ok_servers} server(s)"
        if failed_count:
            summary += f" ({failed_count} failed)"
        logger.info(summary)

    # Return ALL registered tools (existing + newly discovered)
    return _existing_tool_names()


def get_mcp_status() -> List[dict]:
    """Return status of all configured MCP servers for banner display.

    Returns a list of dicts with keys: name, transport, tools, connected.
    Includes both successfully connected servers and configured-but-failed ones.
    """
    result: List[dict] = []

    # Get configured servers from config
    configured = _load_mcp_config()
    if not configured:
        return result

    with _lock:
        active_servers = dict(_servers)

    for name, cfg in configured.items():
        transport = "http" if "url" in cfg else "stdio"
        server = active_servers.get(name)
        if server and server.session is not None:
            entry = {
                "name": name,
                "transport": transport,
                "tools": len(server._registered_tool_names) if hasattr(server, "_registered_tool_names") else len(server._tools),
                "connected": True,
            }
            if server._sampling:
                entry["sampling"] = dict(server._sampling.metrics)
            result.append(entry)
        else:
            result.append({
                "name": name,
                "transport": transport,
                "tools": 0,
                "connected": False,
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
        _run_on_mcp_loop(_probe_all(), timeout=120)
    except Exception as exc:
        logger.debug("MCP probe failed: %s", exc)
    finally:
        _stop_mcp_loop()

    return result


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
        try:
            future = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            future.result(timeout=15)
        except Exception as exc:
            logger.debug("Error during MCP shutdown: %s", exc)

    _stop_mcp_loop()


def _stop_mcp_loop():
    """Stop the background event loop and join its thread."""
    global _mcp_loop, _mcp_thread
    with _lock:
        loop = _mcp_loop
        thread = _mcp_thread
        _mcp_loop = None
        _mcp_thread = None
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        loop.close()
