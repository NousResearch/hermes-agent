"""One MCP response delivered as a single SSE event larger than 1 MiB.

httpx2 2.12 bounds every ``EventSource`` at 1 MiB by default, and the mcp SDK's Streamable HTTP
client inherits that bound unless the caller passes ``max_sse_event_size``. A server whose
``tools/list`` is one ~1 MiB event (a 162-tool catalog) then failed to connect and was parked with
0 tools, although Hermes' own wire-body cap is 10 MiB per event. Each test serves a real catalog
from a local MCP server subprocess and drives ``hermes mcp test`` / the shared connect path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import pytest

import hermes_yaml as yaml
from hermes_constants import get_hermes_home
from tests.waiting import wait_until

_SERVER = """
import asyncio, socket, sys
import uvicorn
from mcp.server import MCPServer
transport, count, description_bytes, port_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
server = MCPServer("catalog")

def noop() -> str:
    return "ok"

for i in range(count):
    server.add_tool(noop, name=f"tool_{i}", description="d" * description_bytes)
app = server.sse_app() if transport == "sse" else server.streamable_http_app()
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
uv = uvicorn.Server(uvicorn.Config(app, log_level="warning"))

async def main():
    task = asyncio.create_task(uv.serve(sockets=[sock]))
    while not uv.started:
        await asyncio.sleep(0.01)
    open(port_path, "w", encoding="utf-8").write(str(sock.getsockname()[1]))
    await task

asyncio.run(main())
"""

_SDK_DEFAULT_EVENT_LIMIT = 1024 * 1024
# 162 tools x 6.6 KB: the shape (and size) of a real catalog that crossed the 1 MiB default.
_JUST_OVER_ONE_MIB = (162, 6_600)


@pytest.fixture
def catalog_server(tmp_path):
    """Start a local MCP server ``(transport, tool_count, description_bytes)``; yields its URL."""
    procs = []

    def start(transport: str, count: int, description_bytes: int) -> str:
        script, port_file = tmp_path / "catalog_server.py", tmp_path / f"port-{len(procs)}"
        script.write_text(_SERVER, encoding="utf-8")
        procs.append(subprocess.Popen([sys.executable, str(script), transport, str(count),
                                       str(description_bytes), str(port_file)]))
        wait_until(lambda: port_file.exists() and port_file.read_text(encoding="utf-8"), timeout=30,
                   what="MCP test server to bind")
        path = "/sse" if transport == "sse" else "/mcp"
        return f"http://127.0.0.1:{port_file.read_text(encoding='utf-8')}{path}"

    yield start
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    shutdown_mcp_servers()
    for proc in procs:
        proc.terminate()
        proc.wait(10)


def _mcp_test(capsys, entry: dict) -> tuple[int, str]:
    """``hermes mcp test catalog`` against *entry*; returns (exit code, printed output)."""
    from hermes_cli.mcp_config import cmd_mcp_test

    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {"catalog": entry}}),
                                                   encoding="utf-8")
    capsys.readouterr()
    code = cmd_mcp_test(argparse.Namespace(name="catalog"))
    return code, capsys.readouterr().out


def test_streamable_http_tools_list_over_one_mib_connects(catalog_server, capsys):
    count, description_bytes = _JUST_OVER_ONE_MIB
    assert count * description_bytes > _SDK_DEFAULT_EVENT_LIMIT
    url = catalog_server("http", count, description_bytes)

    code, out = _mcp_test(capsys, {"url": url})

    assert code == 0, out
    assert f"Tools discovered: {count}" in out
