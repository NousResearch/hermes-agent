"""``manage_connections`` ``authorize`` saves ``auth: oauth`` only for an authorization that happened.

Regression: a server added with ``hermes mcp add --auth header`` answers ``initialize`` on its
static header. Authorizing it obtained no OAuth token, yet a tool discovery failure after that
handshake was read as "authorized" and the entry was saved with ``auth: oauth``. Every later
gateway start then parked the server: "non-interactive environment and no cached tokens found".
"""

from __future__ import annotations

import json
import subprocess
import sys

import hermes_yaml as yaml
import pytest

from tests.waiting import wait_until
from tools.connectors.contract import TargetState
from tools.connectors.tool import MANAGE_CONNECTIONS_SCHEMA
from tools.registry import registry

# Answers ``initialize`` and fails ``tools/list``; argv[2] is the Authorization value it requires.
_SERVER = """
import asyncio, socket, sys
import uvicorn
from mcp.server import MCPServer
port_path, required = sys.argv[1], sys.argv[2].encode()

class Notes(MCPServer):
    async def list_tools(self):
        raise RuntimeError("tools unavailable")

def guarded(app):
    async def wrapped(scope, receive, send):
        if scope["type"] == "http" and required and dict(scope["headers"]).get(b"authorization") != required:
            await send({"type": "http.response.start", "status": 401, "headers": []})
            return await send({"type": "http.response.body", "body": b""})
        return await app(scope, receive, send)
    return wrapped

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
uv = uvicorn.Server(uvicorn.Config(guarded(Notes("notes").streamable_http_app()), log_level="warning"))

async def main():
    task = asyncio.create_task(uv.serve(sockets=[sock]))
    while not uv.started:
        await asyncio.sleep(0.01)
    open(port_path, "w", encoding="utf-8").write(str(sock.getsockname()[1]))
    await task

asyncio.run(main())
"""


@pytest.fixture
def serve(tmp_path):
    procs = []

    def start(required: str) -> str:
        script, port_file = tmp_path / "server.py", tmp_path / "port"
        script.write_text(_SERVER, encoding="utf-8")
        procs.append(subprocess.Popen([sys.executable, str(script), str(port_file), required]))
        port = wait_until(lambda: port_file.exists() and port_file.read_text(encoding="utf-8"),
                          timeout=30, what="the MCP server's port")
        return f"http://127.0.0.1:{port}/mcp"

    yield start
    for proc in procs:
        proc.terminate()
        proc.wait(10)


@pytest.mark.parametrize("auth, required, refusal", [
    ({"headers": {"Authorization": "Bearer ${MCP_NOTES_API_KEY}"}}, "Bearer static-token", "not OAuth"),
    ({}, "", "tools unavailable"),
], ids=["static-header", "no-auth"])
def test_authorize_without_a_token_leaves_the_saved_entry_alone(
        tmp_path, monkeypatch, serve, auth, required, refusal):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setattr("tools.mcp_tool._MAX_INITIAL_CONNECT_RETRIES", 0)  # one connect is the whole case
    entry = {"url": serve(required), **auth, "enabled": True}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"mcp_servers": {"notes": entry}}), encoding="utf-8")
    (tmp_path / ".env").write_text("MCP_NOTES_API_KEY=static-token\n", encoding="utf-8")

    out = json.loads(registry.dispatch(
        MANAGE_CONNECTIONS_SCHEMA["name"], {"action": "authorize", "connectors": [{"name": "notes", "mcp": True}]}))

    assert yaml.safe_load(config.read_text(encoding="utf-8"))["mcp_servers"]["notes"] == entry
    (target,) = out["targets"]
    assert target["state"] == TargetState.failed.value and refusal in target["detail"], out
