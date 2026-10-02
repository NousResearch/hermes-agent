"""Large MCP tool results over Streamable HTTP, end to end.

A real SDK ``ClientSession`` runs over the client Hermes builds for a Streamable HTTP server
(``MCPServerTask._streamable_http_transport``), against an in-process server that streams its
answer in 1,400-byte pieces — the TLS record size a real server used for a 16 MB Superhuman
thread. Two contracts:

- a result between httpx2's own 1 MiB SSE event default and Hermes' wire-body cap arrives whole
  (httpx2 >= 2.10 would otherwise refuse it inside ``EventSource`` and the SDK would report only
  "SSE stream ended without a response");
- a result over the wire-body cap is answered at once with an explicit "result too large" error
  carrying the byte count, and the session stays usable for the next call.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tools import mcp_tool as _core
from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_errors import _MCP_HTTP_MAX_BODY_BYTES

URL = "http://mcp.test/mcp"
PIECE = 1400


def _answer(message_id, text: str) -> dict:
    return {"jsonrpc": "2.0", "id": message_id, "result": {"content": [{"type": "text", "text": text}]}}


def _server(httpx, sizes: dict, *, framing: str = "sse"):
    """MockTransport handler: ``tools/call`` of tool *name* answers a text of ``sizes[name]`` chars, framed
    as one SSE event (``sse``), a JSON body with a Content-Length (``json``) or a JSON body streamed
    without one (``json-streamed``)."""

    class _Pieces(httpx.AsyncByteStream):
        def __init__(self, body: bytes):
            self._body = body

        async def __aiter__(self):
            for start in range(0, len(self._body), PIECE):
                yield self._body[start:start + PIECE]

    def handle(request):
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        if message["method"] == "initialize":
            return httpx.Response(200, headers={"mcp-session-id": "s"}, json={
                "jsonrpc": "2.0", "id": message["id"], "result": {
                    "protocolVersion": message["params"]["protocolVersion"], "capabilities": {"tools": {}},
                    "serverInfo": {"name": "large", "version": "1"}}})
        if message["method"] == "tools/list":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": {"tools": [
                {"name": name, "inputSchema": {"type": "object"}} for name in sizes]}})
        if message["method"] != "tools/call":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": {}})
        body = json.dumps(_answer(message["id"], "x" * sizes[message["params"]["name"]])).encode()
        if framing == "json":
            return httpx.Response(200, headers={"content-type": "application/json"}, content=body)
        if framing == "json-streamed":
            return httpx.Response(200, headers={"content-type": "application/json"}, stream=_Pieces(body))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=_Pieces(b"event: message\ndata: " + body + b"\n\n"))

    return handle


def _call_tools(monkeypatch, sizes: dict, names: list, *, framing: str = "sse") -> list:
    """Each of *names* called in turn on one session; a result's text length, or the MCPError."""
    assert _core._ensure_mcp_sdk() and _core._MCP_NEW_HTTP
    httpx = _core.sdk_httpx()
    handler = _server(httpx, sizes, framing=framing)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **_kw: httpx.MockTransport(handler))
    from mcp import ClientSession
    from mcp.shared.exceptions import MCPError

    async def run():
        transport = MCPServerTask("large")._streamable_http_transport(
            URL, {}, 30.0, True, None, None, False, set())
        outcomes = []
        async with transport as (read, write, *_), ClientSession(read, write) as session:
            await session.initialize()
            for name in names:
                try:
                    result = await asyncio.wait_for(session.call_tool(name, {}), timeout=30)
                    outcomes.append(len(result.content[0].text))
                except MCPError as exc:
                    outcomes.append(exc)
        return outcomes

    return asyncio.run(run())


def test_a_result_above_httpx2s_event_default_arrives_whole(monkeypatch):
    two_mib = 2 * 1024 * 1024
    assert _call_tools(monkeypatch, {"thread": two_mib}, ["thread"]) == [two_mib]


@pytest.mark.parametrize("framing", ["sse", "json", "json-streamed"])
def test_a_result_over_the_wire_cap_is_answered_as_too_large_and_the_session_survives(monkeypatch, framing):
    too_large, small = _MCP_HTTP_MAX_BODY_BYTES + 1, 64
    refused, after = _call_tools(monkeypatch, {"thread": too_large, "list": small}, ["thread", "list"],
                                 framing=framing)
    assert "MCP result too large" in str(refused) and "tools/call" in str(refused)
    received = refused.error.data["received_bytes"]
    assert received > _MCP_HTTP_MAX_BODY_BYTES
    assert f"{received:,} bytes" in str(refused)
    assert after == small
