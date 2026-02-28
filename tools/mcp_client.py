"""
MCP (Model Context Protocol) Client -- JSON-RPC 2.0 implementation.

Low-level protocol client with two transport backends:
  - StdioTransport: communicate with MCP servers via subprocess stdin/stdout
  - HttpTransport: communicate with MCP servers via HTTP POST (Streamable HTTP)

Zero external dependencies -- uses only Python stdlib.

Protocol lifecycle:
  1. Client sends ``initialize`` with protocolVersion + capabilities
  2. Server responds with its capabilities + serverInfo
  3. Client sends ``notifications/initialized``
  4. Client calls ``tools/list`` to discover available tools
  5. Client calls ``tools/call`` to invoke a tool

Security:
  - Subprocess environment is isolated (only safe env vars passed)
  - HTTP responses are size-limited to prevent JSON bomb attacks
  - Process cleanup handles stdin/stdout/stderr properly
  - Error messages are sanitized to prevent credential leakage

Usage:
    from tools.mcp_client import MCPClient, StdioTransport, HttpTransport

    transport = StdioTransport(command="npx", args=["-y", "@mcp/server"])
    client = MCPClient(transport)
    client.connect()
    tools = client.list_tools()
    result = client.call_tool("tool_name", {"arg": "value"})
    client.disconnect()
"""

import json
import logging
import os
import re
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Protocol version we advertise
MCP_PROTOCOL_VERSION = "2024-11-05"

# Timeouts
CONNECT_TIMEOUT = 30
REQUEST_TIMEOUT = 60
READ_TIMEOUT = 5

# Safety limits
MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10 MB max response
MAX_CONSECUTIVE_PARSE_ERRORS = 10

# Safe environment variables to pass to MCP server subprocesses.
# Only these (plus user-specified env from config) are forwarded.
_SAFE_ENV_VARS: Set[str] = {
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM",
    "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "TMPDIR", "TEMP", "TMP",
    "SYSTEMROOT", "COMSPEC",              # Windows essentials
    "APPDATA", "LOCALAPPDATA", "USERPROFILE",  # Windows paths
    "NODE_PATH", "NODE_ENV",              # Node.js (most MCP servers are npm)
    "PYTHON", "PYTHONPATH",               # Python servers
    "XDG_CONFIG_HOME", "XDG_DATA_HOME",   # Linux XDG
}


class MCPTransportError(Exception):
    """Raised when transport-level communication fails."""


class MCPProtocolError(Exception):
    """Raised when the MCP server returns an error response."""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.error_message = message
        self.data = data
        super().__init__(f"MCP error {code}: {message}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_safe_env(custom_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build a minimal environment for MCP server subprocesses.

    Only safe system vars + user-specified vars from config are included.
    This prevents leaking API keys, database passwords, etc.
    """
    safe = {}
    for var in _SAFE_ENV_VARS:
        val = os.environ.get(var)
        if val:
            safe[var] = val
    if custom_env:
        safe.update(custom_env)
    return safe


def sanitize_error(msg: str) -> str:
    """Remove credentials and tokens from error messages."""
    # URL credentials: https://user:pass@host
    msg = re.sub(r'(https?://)([^:]+):([^@]+)@', r'\1***:***@', msg)
    # Bearer tokens
    msg = re.sub(r'Bearer\s+[A-Za-z0-9_\-\.]{8,}', 'Bearer [redacted]', msg, flags=re.IGNORECASE)
    # Common key patterns
    msg = re.sub(
        r'(api[_-]?key|token|password|secret|authorization)["\s:=]+\S+',
        r'\1=[redacted]', msg, flags=re.IGNORECASE,
    )
    return msg


def _safe_json_loads(data: str) -> dict:
    """Parse JSON with size validation."""
    if len(data) > MAX_RESPONSE_SIZE:
        raise MCPTransportError(
            f"Response too large: {len(data)} bytes (max {MAX_RESPONSE_SIZE})"
        )
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise MCPTransportError(f"Invalid JSON response: {e}")


# ---------------------------------------------------------------------------
# StdioTransport
# ---------------------------------------------------------------------------

class StdioTransport:
    """Communicate with an MCP server via subprocess stdin/stdout.

    Messages are newline-delimited JSON (one JSON-RPC message per line).
    A background reader thread continuously reads stdout into a queue.
    """

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._response_queue: Queue = Queue()
        self._running = False
        self._parse_errors = 0
        # Notification routing
        self._notification_handlers: Dict[str, List[Callable]] = {}
        self._notification_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._running and self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """Spawn the subprocess and start the reader thread."""
        if self._running:
            return

        cmd = [self.command] + self.args
        proc_env = _create_safe_env(self.env)

        try:
            kwargs: Dict[str, Any] = dict(
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=proc_env,
                bufsize=0,
            )
            if self.cwd:
                kwargs["cwd"] = self.cwd

            # Windows compatibility: don't use os.setsid
            if sys.platform != "win32":
                kwargs["preexec_fn"] = os.setsid

            self._process = subprocess.Popen(cmd, **kwargs)
        except FileNotFoundError:
            raise MCPTransportError(
                f"Command not found: {self.command}. "
                f"Make sure the MCP server is installed."
            )
        except Exception as e:
            raise MCPTransportError(f"Failed to start MCP server: {sanitize_error(str(e))}")

        self._running = True
        self._parse_errors = 0
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name=f"mcp-stdio-reader-{self.command}",
        )
        self._reader_thread.start()

    def stop(self) -> None:
        """Terminate the subprocess and stop the reader."""
        self._running = False

        # Wait for reader thread to notice _running=False
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)

        if self._process:
            try:
                # Close stdin to signal the server to exit
                if self._process.stdin and not self._process.stdin.closed:
                    try:
                        self._process.stdin.close()
                    except Exception:
                        pass

                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        try:
                            self._process.wait(timeout=2)
                        except Exception:
                            pass
            except Exception as e:
                logger.debug("MCP process cleanup error: %s", e)
            finally:
                # Close all streams to prevent FD leaks
                for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                    if stream and not stream.closed:
                        try:
                            stream.close()
                        except Exception:
                            pass
                self._process = None

    def send(self, message: dict) -> None:
        """Send a JSON-RPC message to the server."""
        if not self.is_connected:
            raise MCPTransportError("Transport not connected")
        try:
            line = json.dumps(message) + "\n"
            self._process.stdin.write(line.encode("utf-8"))
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._running = False
            raise MCPTransportError(f"Failed to send message: {sanitize_error(str(e))}")

    def on_notification(self, method: str, handler: Callable) -> None:
        """Register a handler for a specific notification method.

        Args:
            method: Notification method name (e.g. "notifications/progress").
            handler: Callable that receives the notification params dict.
        """
        with self._notification_lock:
            if method not in self._notification_handlers:
                self._notification_handlers[method] = []
            self._notification_handlers[method].append(handler)

    def remove_notification_handler(self, method: str, handler: Callable) -> None:
        """Remove a previously registered notification handler."""
        with self._notification_lock:
            handlers = self._notification_handlers.get(method, [])
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    def _dispatch_notification(self, msg: dict) -> None:
        """Dispatch a notification to registered handlers (error-isolated)."""
        method = msg.get("method", "")
        params = msg.get("params", {})
        with self._notification_lock:
            handlers = list(self._notification_handlers.get(method, []))
        for handler in handlers:
            try:
                handler(params)
            except Exception as e:
                logger.debug(
                    "MCP notification handler error (method=%s): %s",
                    method, e,
                )

    def receive(self, timeout: float = REQUEST_TIMEOUT) -> dict:
        """Wait for and return the next JSON-RPC response."""
        try:
            data = self._response_queue.get(timeout=timeout)
            if isinstance(data, Exception):
                raise data
            return data
        except Empty:
            raise MCPTransportError(
                f"Timeout waiting for MCP server response ({timeout}s)"
            )

    def _reader_loop(self) -> None:
        """Background thread: read JSON lines from stdout."""
        try:
            while self._running and self._process and self._process.poll() is None:
                line = self._process.stdout.readline()
                if not line:
                    break
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    self._parse_errors = 0  # reset on success
                    # Route: notifications (no id) vs responses (has id)
                    if "id" in msg:
                        self._response_queue.put(msg)
                    elif "method" in msg:
                        self._dispatch_notification(msg)
                    else:
                        # Unknown message shape -- queue for backward compat
                        self._response_queue.put(msg)
                except json.JSONDecodeError:
                    self._parse_errors += 1
                    logger.debug(
                        "MCP stdio: non-JSON line (%d/%d): %s",
                        self._parse_errors, MAX_CONSECUTIVE_PARSE_ERRORS, line[:200],
                    )
                    if self._parse_errors >= MAX_CONSECUTIVE_PARSE_ERRORS:
                        self._response_queue.put(
                            MCPTransportError(
                                f"Too many consecutive JSON parse errors "
                                f"({self._parse_errors}). Server may be "
                                f"outputting non-JSON to stdout."
                            )
                        )
                        break
        except Exception as e:
            if self._running:
                self._response_queue.put(
                    MCPTransportError(f"Reader error: {sanitize_error(str(e))}")
                )
        finally:
            self._running = False


# ---------------------------------------------------------------------------
# HttpTransport
# ---------------------------------------------------------------------------

class _MCPRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Preserve Authorization header through HTTP redirects.

    Standard urllib drops auth headers on redirect. MCP OAuth flows
    (e.g. Slack) require the header to survive 302 redirects.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            # Preserve auth-related headers
            for hdr in ("Authorization", "Mcp-Session-Id"):
                val = req.get_header(hdr)
                if val and not new_req.has_header(hdr):
                    new_req.add_unredirected_header(hdr, val)
        return new_req


class HttpTransport:
    """Communicate with an MCP server via HTTP POST (Streamable HTTP).

    Each JSON-RPC message is sent as an HTTP POST request.
    The server responds with the JSON-RPC response in the body.
    Session tracking via ``Mcp-Session-Id`` header.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.url = url
        self.headers = headers or {}
        self._session_id: Optional[str] = None
        self._connected = False
        # Build opener with redirect handler
        self._opener = urllib.request.build_opener(_MCPRedirectHandler)
        # SSL context with certificate verification
        self._ssl_ctx = ssl.create_default_context()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        self._connected = True

    def stop(self) -> None:
        self._connected = False
        self._session_id = None

    def send_and_receive(self, message: dict, timeout: float = REQUEST_TIMEOUT) -> dict:
        """Send a JSON-RPC message via HTTP POST and return the response."""
        if not self._connected:
            raise MCPTransportError("Transport not connected")

        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "HermesAgent/1.0 MCP-Client",
        }
        req_headers.update(self.headers)
        if self._session_id:
            req_headers["Mcp-Session-Id"] = self._session_id

        body = json.dumps(message).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers=req_headers,
            method="POST",
        )

        try:
            with self._opener.open(req, timeout=timeout) as resp:
                # Capture session ID from response
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self._session_id = sid

                raw = resp.read()
                if len(raw) > MAX_RESPONSE_SIZE:
                    raise MCPTransportError(
                        f"Response too large: {len(raw)} bytes"
                    )
                data = raw.decode("utf-8")
                return _safe_json_loads(data)

        except urllib.error.HTTPError as e:
            # Detect session expiration: 404 when we had a session ID
            if e.code == 404 and self._session_id:
                logger.warning("MCP HTTP session expired (404), clearing session")
                self._session_id = None
                raise MCPTransportError(
                    "MCP session expired (HTTP 404). Reconnection needed."
                )
            raise MCPTransportError(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise MCPTransportError(f"Connection failed: {sanitize_error(str(e.reason))}")
        except MCPTransportError:
            raise
        except json.JSONDecodeError as e:
            raise MCPTransportError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise MCPTransportError(f"HTTP request failed: {sanitize_error(str(e))}")

    # Compatibility shims so MCPClient can use a uniform interface
    def send(self, message: dict) -> None:
        raise NotImplementedError("HttpTransport uses send_and_receive()")

    def receive(self, timeout: float = REQUEST_TIMEOUT) -> dict:
        raise NotImplementedError("HttpTransport uses send_and_receive()")


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------

class MCPClient:
    """High-level MCP client.

    Wraps a transport and implements the MCP protocol lifecycle:
    connect (initialize) -> list_tools -> call_tool -> disconnect.
    """

    def __init__(self, transport):
        self.transport = transport
        self._request_id = 0
        self._server_info: Optional[dict] = None
        self._server_capabilities: Optional[dict] = None
        self._connected = False
        # Progress notification support
        self._progress_token_counter = 0
        self._progress_callbacks: Dict[str, Callable] = {}
        self._progress_lock = threading.Lock()
        self._progress_handler_registered = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.transport.is_connected

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send_request(
        self, method: str, params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """Send a JSON-RPC request and wait for the response.

        Args:
            method: JSON-RPC method name.
            params: Method parameters.
            timeout: Per-request timeout override (seconds).
        """
        effective_timeout = timeout or REQUEST_TIMEOUT
        msg_id = self._next_id()
        msg = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        if isinstance(self.transport, HttpTransport):
            response = self.transport.send_and_receive(msg, timeout=effective_timeout)
        else:
            self.transport.send(msg)
            response = self.transport.receive(timeout=effective_timeout)

        # Validate response ID matches request ID
        resp_id = response.get("id")
        if resp_id is not None and resp_id != msg_id:
            logger.warning(
                "MCP response ID mismatch: sent %s, got %s (method=%s)",
                msg_id, resp_id, method,
            )

        # Handle JSON-RPC error
        if "error" in response:
            err = response["error"]
            raise MCPProtocolError(
                code=err.get("code", -1),
                message=err.get("message", "Unknown error"),
                data=err.get("data"),
            )

        return response.get("result", {})

    def on_notification(self, method: str, handler: Callable) -> None:
        """Register a handler for server notifications.

        Delegates to the transport's notification routing for StdioTransport.
        For HttpTransport, notifications are not supported (no persistent connection).
        """
        if hasattr(self.transport, 'on_notification'):
            self.transport.on_notification(method, handler)

    def remove_notification_handler(self, method: str, handler: Callable) -> None:
        """Remove a previously registered notification handler."""
        if hasattr(self.transport, 'remove_notification_handler'):
            self.transport.remove_notification_handler(method, handler)

    def _send_notification(self, method: str, params: Optional[dict] = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        msg = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        if isinstance(self.transport, HttpTransport):
            try:
                self.transport.send_and_receive(msg, timeout=10)
            except Exception:
                pass  # Notifications don't require responses
        else:
            self.transport.send(msg)

    def connect(self, timeout: float = CONNECT_TIMEOUT) -> dict:
        """Initialize the MCP connection.

        Returns the server's capabilities dict.
        """
        self.transport.start()

        # Step 1: Send initialize request
        result = self._send_request("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
                "resources": {"subscribe": True, "listChanged": True},
                "prompts": {"listChanged": True},
                "logging": {},
            },
            "clientInfo": {
                "name": "hermes-agent",
                "version": "1.0.0",
            },
        }, timeout=timeout)

        self._server_info = result.get("serverInfo", {})
        self._server_capabilities = result.get("capabilities", {})

        # Step 2: Send initialized notification
        self._send_notification("notifications/initialized")

        self._connected = True
        logger.info(
            "MCP connected to %s (version %s)",
            self._server_info.get("name", "unknown"),
            self._server_info.get("version", "?"),
        )

        return {
            "serverInfo": self._server_info,
            "capabilities": self._server_capabilities,
        }

    def list_tools(self) -> List[dict]:
        """Discover available tools from the server.

        Returns a list of tool definitions, each with:
          - name: str
          - description: str
          - inputSchema: dict (JSON Schema for parameters)
        """
        if not self.is_connected:
            raise MCPTransportError("Not connected")

        result = self._send_request("tools/list", {})
        return result.get("tools", [])

    def call_tool(
        self, name: str, arguments: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """Call a tool on the MCP server.

        Args:
            name: Tool name as returned by list_tools().
            arguments: Tool arguments matching the inputSchema.
            timeout: Per-request timeout override (seconds).

        Returns:
            dict with ``content`` (list of content blocks) and ``isError`` (bool).
        """
        if not self.is_connected:
            raise MCPTransportError("Not connected")

        params = {"name": name}
        if arguments:
            params["arguments"] = arguments

        return self._send_request("tools/call", params, timeout=timeout)

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    def list_resources(self) -> List[dict]:
        """Discover available resources from the server.

        Returns a list of resource definitions (uri, name, mimeType, etc.).
        Requires server to advertise ``resources`` capability.
        """
        if not self.is_connected:
            raise MCPTransportError("Not connected")
        caps = self._server_capabilities or {}
        if "resources" not in caps:
            return []
        result = self._send_request("resources/list", {})
        return result.get("resources", [])

    def read_resource(self, uri: str) -> dict:
        """Read a resource by URI.

        Returns dict with ``contents`` list.
        """
        if not self.is_connected:
            raise MCPTransportError("Not connected")
        return self._send_request("resources/read", {"uri": uri})

    def subscribe_resource(self, uri: str) -> None:
        """Subscribe to updates for a resource URI.

        Requires server ``resources.subscribe`` capability.
        """
        if not self.is_connected:
            raise MCPTransportError("Not connected")
        caps = self._server_capabilities or {}
        res_caps = caps.get("resources", {})
        if not res_caps.get("subscribe"):
            raise MCPProtocolError(
                code=-32601,
                message="Server does not support resource subscriptions",
            )
        self._send_request("resources/subscribe", {"uri": uri})

    def unsubscribe_resource(self, uri: str) -> None:
        """Unsubscribe from resource updates."""
        if not self.is_connected:
            raise MCPTransportError("Not connected")
        self._send_request("resources/unsubscribe", {"uri": uri})

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def list_prompts(self) -> List[dict]:
        """Discover available prompt templates from the server.

        Returns a list of prompt definitions (name, description, arguments).
        Requires server to advertise ``prompts`` capability.
        """
        if not self.is_connected:
            raise MCPTransportError("Not connected")
        caps = self._server_capabilities or {}
        if "prompts" not in caps:
            return []
        result = self._send_request("prompts/list", {})
        return result.get("prompts", [])

    def get_prompt(self, name: str, arguments: Optional[dict] = None) -> dict:
        """Get a rendered prompt template.

        Args:
            name: Prompt template name.
            arguments: Template arguments.

        Returns dict with ``messages`` list.
        """
        if not self.is_connected:
            raise MCPTransportError("Not connected")
        params: Dict[str, Any] = {"name": name}
        if arguments:
            params["arguments"] = arguments
        return self._send_request("prompts/get", params)

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def call_tool_with_progress(
        self,
        name: str,
        arguments: Optional[dict] = None,
        progress_callback: Optional[Callable] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """Call a tool with progress notification support.

        Args:
            name: Tool name.
            arguments: Tool arguments.
            progress_callback: Called with (progress, total, message) on each update.
            timeout: Per-request timeout override.

        Returns:
            dict with ``content`` and ``isError``.
        """
        if not self.is_connected:
            raise MCPTransportError("Not connected")

        params: Dict[str, Any] = {"name": name}
        if arguments:
            params["arguments"] = arguments

        token = None
        if progress_callback:
            self._ensure_progress_handler()
            with self._progress_lock:
                self._progress_token_counter += 1
                token = f"hermes-progress-{self._progress_token_counter}"
                self._progress_callbacks[token] = progress_callback
            params["_meta"] = {"progressToken": token}

        try:
            return self._send_request("tools/call", params, timeout=timeout)
        finally:
            if token:
                with self._progress_lock:
                    self._progress_callbacks.pop(token, None)

    def _ensure_progress_handler(self) -> None:
        """Register the progress notification handler once."""
        if self._progress_handler_registered:
            return
        self._progress_handler_registered = True
        self.on_notification("notifications/progress", self._on_progress)

    def _on_progress(self, params: dict) -> None:
        """Dispatch progress notification to the registered callback."""
        token = params.get("progressToken")
        if not token:
            return
        with self._progress_lock:
            callback = self._progress_callbacks.get(token)
        if callback:
            try:
                callback(
                    params.get("progress", 0),
                    params.get("total"),
                    params.get("message", ""),
                )
            except Exception as e:
                logger.debug("Progress callback error: %s", e)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def set_log_level(self, level: str) -> None:
        """Set the server's logging level.

        Args:
            level: MCP log level (debug, info, notice, warning, error,
                   critical, alert, emergency).
        """
        if not self.is_connected:
            raise MCPTransportError("Not connected")
        self._send_request("logging/setLevel", {"level": level})

    def disconnect(self) -> None:
        """Gracefully close the connection."""
        self._connected = False
        try:
            self.transport.stop()
        except Exception as e:
            logger.debug("MCP disconnect error: %s", e)

    @property
    def server_name(self) -> str:
        if self._server_info:
            return self._server_info.get("name", "unknown")
        return "unknown"


# ---------------------------------------------------------------------------
# MCP Log Level mapping
# ---------------------------------------------------------------------------

_MCP_LOG_LEVELS: Dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO,       # Python has no NOTICE; map to INFO
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "alert": logging.CRITICAL,    # Map to CRITICAL
    "emergency": logging.CRITICAL,
}
