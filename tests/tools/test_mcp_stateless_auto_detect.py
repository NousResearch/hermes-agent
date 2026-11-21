"""A server that only speaks the MCP 2026-07-28 stateless protocol connects under ``protocol: auto``.

Such a server routes every POST through its per-request validation, so a handshake-era
``initialize`` (no ``params._meta`` envelope) is refused with -32602 invalid-params. ``auto``
must read that refusal as "this server is stateless" and retry with ``server/discover``, the
way it already does for -32022 and method-not-found. The server here is the official SDK's own
2026-07-28 request handler mounted for every request, so the refusal is the real wire shape.
"""

import asyncio
import socket
import threading

import pytest

pytest.importorskip("mcp.server._streamable_http_modern")

from tools.mcp_tool_errors import _handshake_rejected_as_modern


class _Err(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.error = type("E", (), {"code": code})()


def test_missing_envelope_rejection_reads_as_stateless_but_a_bare_invalid_params_does_not():
    assert _handshake_rejected_as_modern(_Err(-32602, "Required request metadata is missing"))
    assert _handshake_rejected_as_modern(_Err(-32602, "params._meta is missing the required envelope key(s)"))
    assert not _handshake_rejected_as_modern(_Err(-32602, "Invalid request parameters"))


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _modern_only_server(port: int):
    """The official SDK's 2026-07-28 handler for every request: no handshake-era path at all."""
    import uvicorn
    from mcp.server._streamable_http_modern import handle_modern_request
    from mcp.server.mcpserver import MCPServer

    mcp_server = MCPServer("modern-only")

    @mcp_server.tool()
    def echo(text: str) -> str:
        return text

    lowlevel = mcp_server._lowlevel_server

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while (await receive())["type"] != "lifespan.shutdown":
                await send({"type": "lifespan.startup.complete"})
            await send({"type": "lifespan.shutdown.complete"})
            return
        await handle_modern_request(lowlevel, None, True, None, scope, receive, send)

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on"))
    threading.Thread(target=server.run, daemon=True).start()
    return server


def test_auto_connects_to_a_stateless_only_server():
    from tools.mcp_tool import MCPServerTask
    from tests.waiting import wait_until

    port = _free_port()
    server = _modern_only_server(port)
    try:
        assert wait_until(lambda: server.started, timeout=15)
        task = MCPServerTask("modern-only")

        async def connect():
            try:
                await asyncio.wait_for(task.start({"url": f"http://127.0.0.1:{port}/mcp", "protocol": "auto"}), 30)
                return [tool.name for tool in task._tools]
            finally:
                await task.shutdown()

        assert asyncio.run(connect()) == ["echo"]
    finally:
        server.should_exit = True
