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

Features:
    - Stdio transport (command + args) and HTTP/StreamableHTTP transport (url)
    - Automatic reconnection with exponential backoff (up to 5 retries)
    - Environment variable filtering for stdio subprocesses (security)
    - Credential stripping in error messages returned to the LLM
    - Configurable per-server timeouts for tool calls and connections
    - Thread-safe architecture with dedicated background event loop

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
    # Sampling types -- separated to avoid breaking MCP support on older SDK
    # versions that may not have these types.
    try:
        from mcp.types import CreateMessageResult, TextContent, ErrorData
        _MCP_SAMPLING_TYPES = True
    except ImportError:
        logger.debug("MCP sampling types not available -- sampling support disabled")
except ImportError:
    logger.debug("mcp package not installed -- MCP tool support disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TOOL_TIMEOUT = 120      # seconds for tool calls
_DEFAULT_CONNECT_TIMEOUT = 60    # seconds for initial connection per server
_MAX_RECONNECT_RETRIES = 5
_MAX_BACKOFF_SECONDS = 60

# Sampling defaults
_DEFAULT_SAMPLING_MAX_TOKENS_CAP = 4096
_DEFAULT_SAMPLING_RATE_LIMIT = 10   # max requests per minute per server
_DEFAULT_SAMPLING_TIMEOUT = 30      # seconds for LLM call
_sampling_counters: Dict[str, List[float]] = {}  # server_name -> [timestamps]
_sampling_metrics: Dict[str, Dict[str, int]] = {}  # server_name -> {requests, errors, ...}
_tool_loop_counters: Dict[str, int] = {}  # server_name -> consecutive tool rounds


def _safe_numeric(value, default, coerce=int, minimum=1):
    """Coerce a config value to a numeric type, returning *default* on failure.

    Handles the case where YAML/JSON config values arrive as strings (e.g.
    ``"10"`` instead of ``10``), or are set to an entirely invalid type.
    Values below *minimum* are clamped to *minimum* to prevent nonsensical
    config (e.g. ``max_rpm: 0`` blocking all requests, ``timeout: 0``
    causing instant timeouts, or ``max_tokens_cap: -1``).
    """
    try:
        result = coerce(value)
        if isinstance(result, float) and not math.isfinite(result):
            return default
        return max(result, minimum)
    except (TypeError, ValueError, OverflowError):
        return default

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


# ---------------------------------------------------------------------------
# Sampling helpers -- server-initiated LLM requests (MCP sampling/createMessage)
# ---------------------------------------------------------------------------

def _sampling_error(message: str, code: int = -1):
    """Return ErrorData (MCP spec) or raise Exception as fallback."""
    if _MCP_SAMPLING_TYPES:
        return ErrorData(code=code, message=message)
    raise Exception(message)


def _check_rate_limit(server_name: str, max_rpm: int) -> bool:
    """Check if a sampling request is within the per-server rate limit.

    Returns True if the request is allowed, False if rate limit exceeded.
    Uses a sliding 60-second window with max ``max_rpm`` requests.
    """
    now = time.time()
    window = now - 60
    timestamps = _sampling_counters.setdefault(server_name, [])
    timestamps[:] = [t for t in timestamps if t > window]
    if len(timestamps) >= max_rpm:
        return False
    timestamps.append(now)
    return True


def _resolve_model(preferences, sampling_config: dict):
    """Resolve the model to use for a sampling request.

    Priority: config override > server hint > None (use default).
    """
    if sampling_config.get("model"):
        return sampling_config["model"]
    if preferences and hasattr(preferences, "hints") and preferences.hints:
        for hint in preferences.hints:
            if hasattr(hint, "name") and hint.name:
                return hint.name
    return None


def _extract_tool_result_text(block) -> str:
    """Extract text from a tool_result content block."""
    if not hasattr(block, "content") or block.content is None:
        return ""
    items = block.content if isinstance(block.content, list) else [block.content]
    return "\n".join(item.text for item in items if hasattr(item, "text"))


def _convert_sampling_messages(params) -> List[dict]:
    """Convert MCP SamplingMessage list to OpenAI messages format.

    Supports text, image, tool_result, and tool_use content blocks.
    """
    messages = []
    for msg in params.messages:
        content = msg.content
        # Single text content
        if hasattr(content, "text"):
            messages.append({"role": msg.role, "content": content.text})
        # Single tool_result content -> OpenAI tool role
        elif hasattr(content, "toolUseId"):
            messages.append({
                "role": "tool",
                "tool_call_id": content.toolUseId,
                "content": _extract_tool_result_text(content),
            })
        elif isinstance(content, list):
            # All blocks are tool_result -> multiple tool role messages
            if all(hasattr(b, "toolUseId") for b in content):
                for block in content:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.toolUseId,
                        "content": _extract_tool_result_text(block),
                    })
            # Any block has tool_use -> assistant message with tool_calls
            elif any(
                hasattr(b, "name") and hasattr(b, "input") for b in content
            ):
                tool_calls = []
                text_parts = []
                for block in content:
                    if hasattr(block, "name") and hasattr(block, "input"):
                        tool_calls.append({
                            "id": getattr(block, "id", f"call_{len(tool_calls)}"),
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": (
                                    json.dumps(block.input)
                                    if isinstance(block.input, dict)
                                    else str(block.input)
                                ),
                            },
                        })
                    elif hasattr(block, "text"):
                        text_parts.append(block.text)
                msg_dict: dict = {"role": msg.role}
                if text_parts:
                    msg_dict["content"] = "\n".join(text_parts)
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                messages.append(msg_dict)
            # Mixed list: some blocks are tool_result, others are text/image
            elif any(hasattr(b, "toolUseId") for b in content):
                for block in content:
                    if hasattr(block, "toolUseId"):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": block.toolUseId,
                            "content": _extract_tool_result_text(block),
                        })
                    elif hasattr(block, "text"):
                        messages.append({"role": msg.role, "content": block.text})
                    else:
                        block_type = type(block).__name__
                        logger.warning(
                            "Unsupported block in mixed tool_result list: %s (skipped)",
                            block_type,
                        )
            else:
                # Text + image + unknown handling
                parts = []
                for block in content:
                    if hasattr(block, "text"):
                        parts.append({"type": "text", "text": block.text})
                    elif hasattr(block, "data") and hasattr(block, "mimeType"):
                        parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{block.mimeType};base64,{block.data}",
                            },
                        })
                    else:
                        block_type = type(block).__name__
                        logger.warning(
                            "Unsupported sampling content block type: %s (skipped)",
                            block_type,
                        )
                messages.append({"role": msg.role, "content": parts if parts else ""})
        else:
            messages.append({"role": msg.role, "content": str(content)})
    return messages


def _map_stop_reason(finish_reason: str) -> str:
    """Map OpenAI finish_reason to MCP stopReason."""
    mapping = {"stop": "endTurn", "length": "maxTokens", "tool_calls": "toolUse"}
    return mapping.get(finish_reason, "endTurn")


def _make_sampling_callback(server_name: str, sampling_config: dict):
    """Return an async sampling callback for MCP server-initiated LLM requests.

    The callback is invoked by the MCP SDK when a server sends a
    ``sampling/createMessage`` request. It forwards the request to the
    agent's auxiliary LLM client and returns the response.

    Requires ``_MCP_SAMPLING_TYPES`` to be True (i.e. the MCP SDK sampling
    types must be importable). Returns ``None`` otherwise to prevent
    NameError on the success path where ``CreateMessageResult``/``TextContent``
    are needed.

    Key design choices:
      - LLM call is offloaded to a thread via ``asyncio.to_thread()`` to
        prevent blocking the MCP background event loop.
      - Errors return ``ErrorData`` (per MCP spec) instead of raising
        exceptions, giving the server a structured error it can handle.
      - Per-server rate limit, LLM timeout, model whitelist, tool loop
        limit, and audit log level are configurable.
    """
    if not _MCP_SAMPLING_TYPES:
        return None

    max_rpm = _safe_numeric(
        sampling_config.get("max_rpm", _DEFAULT_SAMPLING_RATE_LIMIT),
        _DEFAULT_SAMPLING_RATE_LIMIT, int,
    )
    llm_timeout = _safe_numeric(
        sampling_config.get("timeout", _DEFAULT_SAMPLING_TIMEOUT),
        _DEFAULT_SAMPLING_TIMEOUT, float,
    )
    max_tokens_cap = _safe_numeric(
        sampling_config.get("max_tokens_cap", _DEFAULT_SAMPLING_MAX_TOKENS_CAP),
        _DEFAULT_SAMPLING_MAX_TOKENS_CAP, int,
    )
    allowed_models = sampling_config.get("allowed_models", [])
    max_tool_rounds = _safe_numeric(
        sampling_config.get("max_tool_rounds", 5), 5, int, minimum=0,
    )

    _log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
    }
    audit_level = _log_levels.get(
        str(sampling_config.get("log_level", "info")).lower(), logging.INFO,
    )

    async def _sampling_callback(context, params):
        # Rate limit check
        if not _check_rate_limit(server_name, max_rpm):
            logger.warning(
                "MCP server '%s' sampling rate limit exceeded (%d/min)",
                server_name, max_rpm,
            )
            metrics = _sampling_metrics.setdefault(server_name, {
                "requests": 0, "errors": 0, "tokens_used": 0, "tool_use_count": 0,
            })
            metrics["errors"] += 1
            return _sampling_error(
                f"Sampling rate limit exceeded for server '{server_name}' "
                f"({max_rpm} requests/minute)"
            )

        max_tokens = min(params.maxTokens, max_tokens_cap)

        model = _resolve_model(
            getattr(params, "modelPreferences", None), sampling_config,
        )

        # Allowed models check
        resolved_model_name = model  # will be resolved below with default
        from agent.auxiliary_client import get_text_auxiliary_client
        client, default_model = get_text_auxiliary_client()
        if client is None:
            msg = "No LLM provider available for sampling"
            logger.error("MCP server '%s': %s", server_name, msg)
            metrics = _sampling_metrics.setdefault(server_name, {
                "requests": 0, "errors": 0, "tokens_used": 0, "tool_use_count": 0,
            })
            metrics["errors"] += 1
            return _sampling_error(msg)

        resolved_model_name = model or default_model
        if allowed_models and resolved_model_name not in allowed_models:
            logger.warning(
                "MCP server '%s' requested model '%s' not in allowed_models",
                server_name, resolved_model_name,
            )
            metrics = _sampling_metrics.setdefault(server_name, {
                "requests": 0, "errors": 0, "tokens_used": 0, "tool_use_count": 0,
            })
            metrics["errors"] += 1
            return _sampling_error(
                f"Model '{resolved_model_name}' not allowed for server "
                f"'{server_name}'. Allowed: {', '.join(allowed_models)}"
            )

        messages = _convert_sampling_messages(params)

        if hasattr(params, "systemPrompt") and params.systemPrompt:
            messages.insert(0, {"role": "system", "content": params.systemPrompt})

        call_kwargs = {
            "model": resolved_model_name,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        if hasattr(params, "temperature") and params.temperature is not None:
            call_kwargs["temperature"] = params.temperature

        stop_sequences = getattr(params, "stopSequences", None)
        if stop_sequences:
            call_kwargs["stop"] = stop_sequences

        # Forward tools/toolChoice from the server if present
        server_tools = getattr(params, "tools", None)
        if server_tools:
            call_kwargs["tools"] = [
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
            tool_choice = getattr(params, "toolChoice", None)
            if tool_choice:
                mode = getattr(tool_choice, "mode", "auto")
                call_kwargs["tool_choice"] = {
                    "auto": "auto", "required": "required", "none": "none",
                }.get(mode, "auto")

        logger.log(
            audit_level,
            "MCP server '%s' sampling request: model=%s, max_tokens=%d, messages=%d",
            server_name, call_kwargs["model"], max_tokens, len(messages),
        )

        # Offload sync LLM call to a thread so we don't block the MCP
        # event loop (which also serves tool calls and other servers).
        def _sync_llm_call():
            return client.chat.completions.create(**call_kwargs)

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_sync_llm_call),
                timeout=llm_timeout,
            )
        except asyncio.TimeoutError:
            msg = (
                f"Sampling LLM call timed out after {llm_timeout}s "
                f"for server '{server_name}'"
            )
            logger.error("MCP server '%s': %s", server_name, msg)
            metrics = _sampling_metrics.setdefault(server_name, {
                "requests": 0, "errors": 0, "tokens_used": 0, "tool_use_count": 0,
            })
            metrics["errors"] += 1
            return _sampling_error(msg)
        except Exception as exc:
            msg = f"Sampling LLM call failed: {_sanitize_error(str(exc))}"
            logger.error(
                "MCP server '%s' sampling LLM call failed: %s",
                server_name, exc,
            )
            metrics = _sampling_metrics.setdefault(server_name, {
                "requests": 0, "errors": 0, "tokens_used": 0, "tool_use_count": 0,
            })
            metrics["errors"] += 1
            return _sampling_error(msg)

        choice = response.choices[0]

        # Update audit metrics
        metrics = _sampling_metrics.setdefault(server_name, {
            "requests": 0, "errors": 0, "tokens_used": 0, "tool_use_count": 0,
        })
        metrics["requests"] += 1
        total_tokens = getattr(
            getattr(response, "usage", None), "total_tokens", 0,
        )
        if isinstance(total_tokens, int):
            metrics["tokens_used"] += total_tokens

        # Tool use response
        if (
            choice.finish_reason == "tool_calls"
            and hasattr(choice.message, "tool_calls")
            and choice.message.tool_calls
        ):
            metrics["tool_use_count"] += 1

            # Tool loop limit enforcement
            if max_tool_rounds == 0:
                _tool_loop_counters.pop(server_name, None)
                return _sampling_error(
                    f"Tool loops disabled for server '{server_name}' "
                    f"(max_tool_rounds=0)"
                )

            if max_tool_rounds > 0:
                loop_count = _tool_loop_counters.get(server_name, 0) + 1
                _tool_loop_counters[server_name] = loop_count
                if loop_count > max_tool_rounds:
                    _tool_loop_counters.pop(server_name, None)
                    return _sampling_error(
                        f"Tool loop limit exceeded for server '{server_name}' "
                        f"(max {max_tool_rounds} rounds)"
                    )

            content_blocks = []
            for tc in choice.message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        parsed_args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        logger.warning(
                            "MCP server '%s': malformed tool_calls arguments "
                            "from LLM (using raw string): %.100s",
                            server_name, args,
                        )
                        parsed_args = args
                else:
                    parsed_args = args
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": parsed_args,
                })

            logger.log(
                audit_level,
                "MCP server '%s' sampling response: model=%s, tokens=%s, tool_calls=%d",
                server_name, response.model,
                getattr(getattr(response, "usage", None), "total_tokens", "?"),
                len(content_blocks),
            )

            return CreateMessageResult(
                role="assistant",
                content=content_blocks,
                model=response.model,
                stopReason="toolUse",
            )

        # Normal text response -- reset tool loop counter
        _tool_loop_counters.pop(server_name, None)

        response_text = choice.message.content or ""

        logger.log(
            audit_level,
            "MCP server '%s' sampling response: model=%s, tokens=%s",
            server_name, response.model,
            getattr(getattr(response, "usage", None), "total_tokens", "?"),
        )

        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=_sanitize_error(response_text)),
            model=response.model,
            stopReason=_map_stop_reason(choice.finish_reason),
        )

    return _sampling_callback


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
        "_sampling_callback",
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
        self._sampling_callback = None

    def _is_http(self) -> bool:
        """Check if this server uses HTTP transport."""
        return "url" in self._config

    def _build_session_kwargs(self) -> dict:
        """Build keyword arguments for ClientSession."""
        kwargs = {}
        if self._sampling_callback is not None:
            kwargs["sampling_callback"] = self._sampling_callback
        return kwargs

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
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=safe_env if safe_env else None,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(
                read_stream, write_stream, **self._build_session_kwargs(),
            ) as session:
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

        async with streamablehttp_client(
            url,
            headers=headers,
            timeout=float(connect_timeout),
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(
                read_stream, write_stream, **self._build_session_kwargs(),
            ) as session:
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

        # Set up sampling callback if enabled and SDK types are available
        sampling_config = config.get("sampling", {})
        if sampling_config.get("enabled", True) and _MCP_SAMPLING_TYPES:
            self._sampling_callback = _make_sampling_callback(
                self.name, sampling_config,
            )
        else:
            self._sampling_callback = None

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


def _existing_tool_names() -> List[str]:
    """Return tool names for all currently connected servers."""
    names: List[str] = []
    for sname, server in _servers.items():
        for mcp_tool in server._tools:
            schema = _convert_mcp_schema(sname, mcp_tool)
            names.append(schema["name"])
        # Also include utility tool names
        for entry in _build_utility_schemas(sname):
            names.append(entry["schema"]["name"])
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

    for mcp_tool in server._tools:
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

    # Register MCP Resources & Prompts utility tools
    _handler_factories = {
        "list_resources": _make_list_resources_handler,
        "read_resource": _make_read_resource_handler,
        "list_prompts": _make_list_prompts_handler,
        "get_prompt": _make_get_prompt_handler,
    }
    check_fn = _make_check_fn(name)
    for entry in _build_utility_schemas(name):
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

    # Only attempt servers that aren't already connected
    with _lock:
        new_servers = {k: v for k, v in servers.items() if k not in _servers}

    if not new_servers:
        return _existing_tool_names()

    # Start the background event loop for MCP connections
    _ensure_mcp_loop()

    all_tools: List[str] = []
    failed_count = 0

    async def _discover_one(name: str, cfg: dict) -> List[str]:
        """Connect to a single server and return its registered tool names."""
        transport_desc = cfg.get("url", f'{cfg.get("command", "?")} {" ".join(cfg.get("args", [])[:2])}')
        try:
            registered = await _discover_and_register_server(name, cfg)
            transport_type = "HTTP" if "url" in cfg else "stdio"
            return registered
        except Exception as exc:
            logger.warning(
                "Failed to connect to MCP server '%s': %s",
                name, exc,
            )
            return []

    async def _discover_all():
        nonlocal failed_count
        # Connect to all servers in PARALLEL
        results = await asyncio.gather(
            *(_discover_one(name, cfg) for name, cfg in new_servers.items()),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                failed_count += 1
                logger.warning("MCP discovery error: %s", result)
            elif isinstance(result, list):
                all_tools.extend(result)
            else:
                failed_count += 1

    # Per-server timeouts are handled inside _discover_and_register_server.
    # The outer timeout is generous: 120s total for parallel discovery.
    _run_on_mcp_loop(_discover_all(), timeout=120)

    if all_tools:
        # Dynamically inject into all hermes-* platform toolsets
        from toolsets import TOOLSETS
        for ts_name, ts in TOOLSETS.items():
            if ts_name.startswith("hermes-"):
                for tool_name in all_tools:
                    if tool_name not in ts["tools"]:
                        ts["tools"].append(tool_name)

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
        metrics = _sampling_metrics.get(name)
        if server and server.session is not None:
            entry = {
                "name": name,
                "transport": transport,
                "tools": len(server._tools),
                "connected": True,
            }
            if metrics:
                entry["sampling"] = dict(metrics)
            result.append(entry)
        else:
            entry = {
                "name": name,
                "transport": transport,
                "tools": 0,
                "connected": False,
            }
            if metrics:
                entry["sampling"] = dict(metrics)
            result.append(entry)

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
