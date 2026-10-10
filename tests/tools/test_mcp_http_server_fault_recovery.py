"""Real SDK/HTTP coverage for server faults versus SSE transport rejection (#121933)."""

import asyncio
import json
import queue
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mcp.shared.exceptions import MCPError

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_errors import _unwrap_exception_group


@contextmanager
def _mcp_server(*, sse_only=False):
    """A sessionless HTTP server with one discovery fault, or a genuine legacy SSE server."""
    requests = []
    messages = queue.Queue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, status, body=b"", content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = request["method"]
            requests.append(("POST", self.path, method))
            if sse_only and self.path == "/mcp":
                self._reply(405, b"Use the SSE endpoint", "text/plain")
                return
            if (not sse_only and method == "tools/list"
                    and sum(rpc == "tools/list" for _, _, rpc in requests) == 1):
                self._reply(503, b"upstream temporarily unavailable", "text/plain")
                return
            if "id" not in request:
                self._reply(202)
                return
            if method == "initialize":
                result = {
                    "protocolVersion": request["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "loopback-mcp", "version": "1"},
                }
            elif method == "tools/list":
                result = {"tools": [{
                    "name": "echo", "description": "Return the supplied message",
                    "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}},
                                    "required": ["message"]},
                }]}
            elif method == "tools/call":
                result = {"content": [{"type": "text", "text": request["params"]["arguments"]["message"]}]}
            else:
                result = {}  # ping
            response = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
            if sse_only:
                self._reply(202)
                messages.put(response)
            else:
                # No MCP-Session-Id: a valid sessionless server has no background GET stream.
                self._reply(200, response)

        def do_GET(self):  # noqa: N802
            requests.append(("GET", self.path, None))
            if not sse_only:
                self._reply(405, b"Streamable HTTP only", "text/plain")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"event: endpoint\ndata: /messages\n\n")
            self.wfile.flush()
            while (message := messages.get()) is not None:
                self.wfile.write(b"event: message\ndata: " + message + b"\n\n")
                self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/mcp", requests
    finally:
        messages.put(None)
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        assert not worker.is_alive()


async def _connect_and_call(task, url):
    try:
        await asyncio.wait_for(task.start({
            "url": url, "protocol": "legacy", "skip_preflight": True,
            "connect_timeout": 5, "keepalive_interval": 60,
            "sampling": {"enabled": False}, "elicitation": {"enabled": False},
        }), timeout=15)
        assert [tool.name for tool in task._tools] == ["echo"]
        result = await asyncio.wait_for(
            task.session.call_tool("echo", {"message": "recovered"}), timeout=5)
        assert not result.is_error
        assert [content.text for content in result.content] == ["recovered"]
    finally:
        await task.shutdown()


def test_real_sdk_server_fault_retries_http_and_recovers(monkeypatch):
    """A plain-text 503 must survive SDK error folding and retry without trying legacy SSE."""
    async def run(url, requests):
        task = MCPServerTask("loopback-server-fault")
        failures = []
        on_initial_error = task._on_initial_connect_error

        async def observe_initial_error(exc, root, failure_class, budget):
            # Observe the real failure before the next attempt clears the recorder; retain the
            # production retry method and its real jittered backoff.
            failures.append((exc, failure_class, dict(task._http_rejection)))
            return await on_initial_error(exc, root, failure_class, budget)

        monkeypatch.setattr(task, "_on_initial_connect_error", observe_initial_error)
        await _connect_and_call(task, url)
        assert not any(method == "GET" for method, _, _ in requests), requests
        assert task._sse_fallback is False
        assert len(failures) == 1
        exc, failure_class, rejection = failures[0]
        sdk_error = _unwrap_exception_group(exc.__cause__)
        assert isinstance(sdk_error, MCPError)
        assert sdk_error.error.code == -32603
        assert failure_class == "transient"
        assert rejection == {"status": 503, "method": "POST", "url": url,
                             "body": "upstream temporarily unavailable"}
        assert "HTTP 503" in str(exc) and "upstream temporarily unavailable" in str(exc)

    with _mcp_server() as (url, requests):
        asyncio.run(run(url, requests))
    assert sum(rpc == "initialize" for _, _, rpc in requests) == 2, requests
    assert sum(rpc == "tools/list" for _, _, rpc in requests) == 2, requests


def test_real_sdk_transport_rejection_still_connects_over_sse():
    """A genuine 405 transport mismatch must still discover and call tools over legacy SSE."""
    async def run(url):
        task = MCPServerTask("loopback-sse-only")
        await _connect_and_call(task, url)
        assert task._sse_fallback is True

    with _mcp_server(sse_only=True) as (url, requests):
        asyncio.run(run(url))
    assert ("POST", "/mcp", "initialize") in requests
    assert ("GET", "/mcp", None) in requests
    assert ("POST", "/messages", "tools/list") in requests
    assert ("POST", "/messages", "tools/call") in requests
