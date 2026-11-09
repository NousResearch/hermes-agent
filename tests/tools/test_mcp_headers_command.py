"""``headers_command``: a remote MCP server whose bearer comes from a command, not a static header.

A real local MCP server accepts only ``Bearer fresh``. The command prints ``stale`` on its first run
and ``fresh`` afterwards, which is what a token that expired between runs looks like: the 401 must
re-run the command once and retry, and the connection must come up without a reconnect.
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
port_path = sys.argv[1]
server = MCPServer("guarded")

def ping() -> str:
    return "pong"

server.add_tool(ping, name="ping", description="ping")
inner = server.streamable_http_app()

async def app(scope, receive, send):
    if scope["type"] == "http" and dict(scope["headers"]).get(b"authorization") != b"Bearer fresh":
        await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"bad token"})
        return
    await inner(scope, receive, send)

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

# First run prints the expired token, every later run the fresh one; counts its runs.
_MINT = """
import pathlib, sys
runs = pathlib.Path(sys.argv[1])
n = int(runs.read_text()) if runs.exists() else 0
runs.write_text(str(n + 1))
print('{"Authorization": "Bearer %s"}' % ("stale" if n == 0 else "fresh"))
"""


@pytest.fixture
def guarded_server(tmp_path):
    script, port_file = tmp_path / "guarded_server.py", tmp_path / "port"
    script.write_text(_SERVER, encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script), str(port_file)])
    try:
        wait_until(lambda: port_file.exists() and port_file.read_text(encoding="utf-8-sig"), timeout=30,
                   what="MCP test server to bind")
        yield f"http://127.0.0.1:{port_file.read_text(encoding='utf-8-sig')}/mcp"
    finally:
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        shutdown_mcp_servers()
        proc.terminate()
        proc.wait(10)


def _mcp_test(capsys, entry: dict) -> tuple[int, str]:
    from hermes_cli.mcp_config import cmd_mcp_test

    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {"guarded": entry}}),
                                                   encoding="utf-8")
    capsys.readouterr()
    code = cmd_mcp_test(argparse.Namespace(name="guarded"))
    return code, capsys.readouterr().out


def test_expired_command_token_is_reminted_on_401(guarded_server, tmp_path, capsys):
    mint, runs = tmp_path / "mint.py", tmp_path / "runs"
    mint.write_text(_MINT, encoding="utf-8")

    code, out = _mcp_test(capsys, {"url": guarded_server, "headers_command": [sys.executable, str(mint), str(runs)]})

    assert code == 0, out
    assert "Tools discovered: 1" in out
    assert runs.read_text(encoding="utf-8-sig") == "2"  # stale → 401 → one re-run → fresh; never a loop


def test_command_output_rejects_header_injection():
    from tools.mcp_header_command import HeadersCommandError, parse_headers_output

    assert parse_headers_output("s", "Authorization: Bearer abc\nX-Org: 7\n") == {
        "Authorization": "Bearer abc", "X-Org": "7"}
    for bad in ('{"Authorization": "Bearer a\\r\\nX-Evil: 1"}', "not a header line", "", '{"bad name": "v"}'):
        with pytest.raises(HeadersCommandError):
            parse_headers_output("s", bad)
