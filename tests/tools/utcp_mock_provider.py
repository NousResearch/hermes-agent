"""Local UTCP provider used by the UTCP tests and benchmark harness.

A stdlib-only HTTP server that serves a UTCP manual at
``/.well-known/utcp`` and a handful of deterministic endpoints.  Keeping the
server dependency-free (``ThreadingHTTPServer``) means the tests exercise the
real SDK transports (aiohttp client side) without needing an async test
harness or a third-party server framework.

Tools are declared with :func:`utcp_tool`; the ``{base}`` placeholder in a
tool's URL is replaced with the bound server address when the provider starts,
so tests never hardcode a port.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

DEFAULT_MANUAL_PATH = "/.well-known/utcp"

# A route handler receives a :class:`MockRequest` and returns
# ``(status, payload, headers)``.  ``payload`` may be a dict/list (JSON) or str.
RouteHandler = Callable[["MockRequest"], tuple[int, Any, dict[str, str]]]


class MockRequest:
    """Parsed inbound request, shaped for test assertions."""

    def __init__(self, method: str, path: str, query: dict[str, list[str]], headers: dict[str, str], body: bytes):
        self.method = method
        self.path = path
        self.query = query
        self.headers = headers
        self.body = body

    def json(self) -> Any:
        if not self.body:
            return None
        return json.loads(self.body.decode("utf-8"))

    def header(self, name: str) -> Optional[str]:
        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return None


def utcp_tool(
    name: str,
    path: str,
    *,
    method: str = "GET",
    description: str = "",
    inputs: Optional[dict[str, Any]] = None,
    outputs: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    body_field: Optional[str] = None,
    header_fields: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build one UTCP tool definition.

    ``path`` is relative to the provider base URL (``{base}`` is prefixed when
    the provider starts).
    """
    template: dict[str, Any] = {
        "name": name,
        "call_template_type": "http",
        "url": "{base}" + path,
        "http_method": method,
    }
    if headers:
        template["headers"] = dict(headers)
    if body_field:
        template["body_field"] = body_field
    if header_fields:
        template["header_fields"] = list(header_fields)
    tool: dict[str, Any] = {
        "name": name,
        "description": description or f"Mock tool {name}",
        "inputs": inputs if inputs is not None else {"type": "object", "properties": {}},
        "tool_call_template": template,
    }
    if outputs is not None:
        tool["outputs"] = outputs
    if tags:
        tool["tags"] = list(tags)
    return tool


def echo_route(request: MockRequest) -> tuple[int, Any, dict[str, str]]:
    """Echo query params, JSON body, and a selected header."""
    return (
        200,
        {
            "method": request.method,
            "path": request.path,
            "query": {key: values[0] for key, values in request.query.items()},
            "body": request.json(),
            "api_key": request.header("X-Api-Key"),
            "authorization": request.header("Authorization"),
        },
        {},
    )


def sum_route(request: MockRequest) -> tuple[int, Any, dict[str, str]]:
    body = request.json() or {}
    return 200, {"sum": body.get("a", 0) + body.get("b", 0), "structured": {"ok": True, "items": [1, 2, 3]}}, {}


def error_route(request: MockRequest) -> tuple[int, Any, dict[str, str]]:
    return 500, {"error": "boom", "detail": "mock provider failure"}, {}


def unauthorized_route(request: MockRequest) -> tuple[int, Any, dict[str, str]]:
    return 401, {"error": "unauthorized"}, {}


def slow_route(request: MockRequest) -> tuple[int, Any, dict[str, str]]:
    time.sleep(1.5)
    return 200, {"slept": 1.5}, {}


def text_route(request: MockRequest) -> tuple[int, Any, dict[str, str]]:
    return 200, "plain text payload", {"Content-Type": "text/plain"}


class MockUtcpProvider:
    """A UTCP provider running on 127.0.0.1 with a configurable manual."""

    def __init__(
        self,
        tools: Optional[list[dict[str, Any]]] = None,
        *,
        manual_path: str = DEFAULT_MANUAL_PATH,
        default_routes: bool = True,
    ) -> None:
        self._tool_defs: list[dict[str, Any]] = list(tools or [])
        self.manual_path = manual_path
        self._routes: dict[str, RouteHandler] = {}
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.base_url = ""
        self.requests: list[MockRequest] = []
        self._lock = threading.Lock()
        if default_routes:
            self._routes.update(
                {
                    "/echo": echo_route,
                    "/sum": sum_route,
                    "/error": error_route,
                    "/unauthorized": unauthorized_route,
                    "/slow": slow_route,
                    "/text": text_route,
                }
            )
        self._manual_override: Optional[dict[str, Any]] = None

    # ── configuration ────────────────────────────────────────────────────
    def add_route(self, path: str, handler: RouteHandler) -> MockUtcpProvider:
        self._routes[path] = handler
        return self

    def add_tool(self, tool: dict[str, Any]) -> MockUtcpProvider:
        self._tool_defs.append(tool)
        return self

    def set_manual(self, manual: dict[str, Any]) -> MockUtcpProvider:
        """Serve an arbitrary payload instead of the generated manual."""
        self._manual_override = manual
        return self

    @property
    def manual_url(self) -> str:
        return self.base_url.rstrip("/") + self.manual_path

    def manual(self) -> dict[str, Any]:
        if self._manual_override is not None:
            return self._manual_override
        defs = []
        for tool in self._tool_defs:
            cloned = json.loads(json.dumps(tool))
            template = cloned.get("tool_call_template") or {}
            if isinstance(template.get("url"), str):
                template["url"] = template["url"].replace("{base}", self.base_url.rstrip("/"))
            defs.append(cloned)
        return {"utcp_version": "1.0.3", "manual_version": "1.0.0", "tools": defs}

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> MockUtcpProvider:
        provider = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):  # silence test output
                pass

            def _dispatch(self) -> None:
                parsed = urlparse(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                request = MockRequest(
                    self.command,
                    parsed.path,
                    parse_qs(parsed.query),
                    {key: value for key, value in self.headers.items()},
                    body,
                )
                with provider._lock:
                    provider.requests.append(request)

                if parsed.path == provider.manual_path:
                    status, payload, headers = 200, provider.manual(), {"Content-Type": "application/json"}
                else:
                    handler = provider._routes.get(parsed.path)
                    if handler is None:
                        status, payload, headers = 404, {"error": "not found"}, {}
                    else:
                        status, payload, headers = handler(request)

                if isinstance(payload, (dict, list)):
                    raw = json.dumps(payload).encode("utf-8")
                    headers = {"Content-Type": "application/json", **headers}
                elif isinstance(payload, bytes):
                    raw = payload
                else:
                    raw = str(payload).encode("utf-8")
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    # The client gave up (e.g. the timeout test cancels the
                    # request); the assertion is on the client side.
                    pass

            def do_GET(self) -> None:
                self._dispatch()

            def do_POST(self) -> None:
                self._dispatch()

            def do_PUT(self) -> None:
                self._dispatch()

            def do_DELETE(self) -> None:
                self._dispatch()

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        self._server = server
        self.base_url = f"http://127.0.0.1:{server.server_address[1]}"
        thread = threading.Thread(target=server.serve_forever, name="mock-utcp-provider", daemon=True)
        thread.start()
        self._thread = thread
        return self

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> MockUtcpProvider:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()
