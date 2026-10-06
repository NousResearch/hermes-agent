"""Exercise Hermes' owned transports and lifecycle against the installed MCP SDK."""

import asyncio
import socket
import sys

import pytest

pytest.importorskip("mcp.server", reason="MCP SDK required")

_SERVER = '''
import sys
from mcp.server import MCPServer
from pydantic import BaseModel
mcp = MCPServer("hermes-sdk-fixture", version="1")
class Sum(BaseModel):
    value: int
class Envelope(BaseModel):
    total: Sum
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b
@mcp.tool()
def nested(a: int, b: int) -> Envelope:
    """Return a sum with a schema-local reference."""
    return Envelope(total=Sum(value=a + b))
if len(sys.argv) > 1:
    mcp.run(transport="streamable-http", host="127.0.0.1", port=int(sys.argv[1]),
            streamable_http_path="/mcp/", json_response=True)
else:
    mcp.run()
'''


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "streamable-http"])
async def test_hermes_sdk_initialize_list_call_shutdown(transport, tmp_path, monkeypatch):
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    server = MCPServerTask("sdk-fixture")
    process = None
    try:
        config = {"sampling": {"enabled": False}, "elicitation": {"enabled": False}}
        if transport == "stdio":
            config.update(command=sys.executable, args=["-c", _SERVER])
        else:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _SERVER, str(port),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            async with asyncio.timeout(10):
                while True:
                    assert process.returncode is None, "HTTP fixture exited before becoming ready"
                    try:
                        _, writer = await asyncio.open_connection("127.0.0.1", port)
                    except OSError:
                        await asyncio.sleep(0.05)
                        continue
                    writer.close()
                    await writer.wait_closed()
                    break
            # The slash redirect must work with Hermes' caller-owned HTTP client.
            config.update(url=f"http://127.0.0.1:{port}/mcp", auth="none")
        await asyncio.wait_for(server.start(config), timeout=15)
        assert server.initialize_result.server_info.name == "hermes-sdk-fixture"
        assert {tool.name for tool in server._tools} == {"add", "nested"}
        result = await asyncio.wait_for(server.session.call_tool("add", {"a": 2, "b": 3}), timeout=5)
        assert not result.is_error
        assert result.structured_content == {"result": 5}
        assert any(block.text == "5" for block in result.content if block.type == "text")
        result = await asyncio.wait_for(server.session.call_tool("nested", {"a": 2, "b": 3}), timeout=5)
        assert not result.is_error
        assert result.structured_content == {"total": {"value": 5}}
    finally:
        await server.shutdown()
        if process is not None and process.returncode is None:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5)
    assert server.session is None
    assert server._task.done()


@pytest.mark.asyncio
async def test_owned_http_client_does_not_follow_cross_origin_redirect(monkeypatch):
    from mcp import ClientSession
    from mcp.shared.exceptions import MCPError
    from tools.mcp_tool import MCPServerTask, sdk_httpx

    httpx = sdk_httpx()
    endpoint = "https://mcp.example/mcp"
    hits = []

    def respond(request):
        hits.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://other.example/mcp"}, request=request)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(respond))
    server = MCPServerTask("redirect-fixture")
    async with server._streamable_http_transport(
            endpoint, {"X-Api-Key": "fixture-only"}, 5, True, None, None, False, {"x-api-key"}) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            with pytest.raises(MCPError, match="(?i)redirect"):
                await asyncio.wait_for(session.initialize(), timeout=5)
    assert hits == [endpoint]
