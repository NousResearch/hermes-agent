"""SEP-2243 routing headers on streamable HTTP, verified on the wire.

The SDK stamps ``Mcp-Method``/``Mcp-Name`` only in modern (stateless) mode; Hermes'
default legacy sessions rely on the transport hook in ``tools/mcp_tool_transport.py``.
These tests run a real client session against a stub MCP server over real HTTP and
assert the headers arrive — no mocked transport.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import pytest

httpx = pytest.importorskip("httpx")

from tools.mcp_tool_transport import _make_method_header_hook


class _StubServer(ThreadingHTTPServer):
    def __init__(self, *args, **kwargs):
        self.session_id = "stub-session"
        self.captured: list = []
        self.stop_event = threading.Event()
        super().__init__(*args, **kwargs)


class _StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def _stub(self) -> _StubServer:
        return cast("_StubServer", self.server)

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Mcp-Session-Id", self._stub.session_id)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self._stub.captured.append(
            (dict(self.headers), body))
        method = body.get("method", "")
        if method == "initialize":
            self._send_json({"jsonrpc": "2.0", "id": body.get("id"),
                             "result": {
                                 "protocolVersion": body["params"]["protocolVersion"],
                                 "capabilities": {},
                                 "serverInfo": {"name": "stub", "version": "0.1"}}})
        elif "id" not in body:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.send_header("Mcp-Session-Id", self._stub.session_id)
            self.end_headers()
        elif method == "tools/call":
            self._send_json({"jsonrpc": "2.0", "id": body.get("id"),
                             "result": {"content": [{"type": "text", "text": "ok"}]}})
        elif method == "tools/list":
            self._send_json({"jsonrpc": "2.0", "id": body.get("id"),
                             "result": {"tools": [
                                 {"name": "search", "description": "stub",
                                  "inputSchema": {"type": "object"}}]}})
        else:
            self._send_json({"jsonrpc": "2.0", "id": body.get("id"),
                             "error": {"code": -32601, "message": "Method not found"}})

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Mcp-Session-Id", self._stub.session_id)
        self.end_headers()
        try:
            self.wfile.flush()
            self._stub.stop_event.wait(25)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_DELETE(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.send_header("Mcp-Session-Id", self._stub.session_id)
        self.end_headers()


def _posts_to(server, method):
    return [(dict((k.lower(), v) for k, v in headers.items()), body)
            for headers, body in server.captured
            if body.get("method") == method]


async def _run_session(url):
    from mcp.client.session import ClientSession

    from tools.mcp_tool import MCPServerTask
    from tools.mcp_tool_common import _core

    _core._ensure_mcp_sdk()  # production order: _run_http ensures before transport
    server = MCPServerTask("wire-test")
    async with server._streamable_http_transport(
            url, {}, 5.0, True, None, None, False, set()) as streams:
        read, write = streams
        async with ClientSession(read, write, read_timeout_seconds=10) as session:
            await session.initialize()
            result = await session.call_tool("search", {"q": "x"})
    return result


@pytest.fixture
def stub_server():
    server = _StubServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.stop_event.set()
    server.shutdown()
    thread.join(timeout=5)


class TestMethodHeadersOnTheWire:
    def test_initialize_posts_method_without_name(self, stub_server):
        asyncio.run(asyncio.wait_for(
            _run_session(f"http://127.0.0.1:{stub_server.server_port}/mcp"), 25))
        posts = _posts_to(stub_server, "initialize")
        assert len(posts) == 1
        assert posts[0][0]["mcp-method"] == "initialize"
        assert "mcp-name" not in posts[0][0]

    def test_tools_call_posts_method_and_name(self, stub_server):
        asyncio.run(asyncio.wait_for(
            _run_session(f"http://127.0.0.1:{stub_server.server_port}/mcp"), 25))
        posts = _posts_to(stub_server, "tools/call")
        assert len(posts) == 1
        assert posts[0][0]["mcp-method"] == "tools/call"
        assert posts[0][0]["mcp-name"] == "search"


class TestMethodHeaderHookGuards:
    def test_existing_headers_win(self):
        hook = _make_method_header_hook()
        assert hook is not None
        request = httpx.Request(
            "POST", "https://mcp.example/mcp",
            content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                "params": {"name": "a"}}).encode(),
            headers={"mcp-method": "tools/call", "mcp-name": "sdk-value"})
        asyncio.run(hook(request))  # hooks are async: httpx2 awaits them
        assert request.headers["mcp-name"] == "sdk-value"

    def test_missing_sdk_table_means_no_hook(self, monkeypatch):
        real_import = builtins.__import__

        def _raise_for_inbound(name, *args, **kwargs):
            if name == "mcp.shared.inbound":
                raise ImportError("no table")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _raise_for_inbound)
        assert _make_method_header_hook() is None
