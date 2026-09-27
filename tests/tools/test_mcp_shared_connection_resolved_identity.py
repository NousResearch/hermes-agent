"""A multiplexed profile never adopts another profile's live MCP connection when the inputs that
connection was opened with resolve differently for it: external secret-source env on a stdio
child, or ``identity_header.value_from: profile`` on HTTP. Both profiles have byte-identical
``mcp_servers`` config, so only the resolved values tell the identities apart."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import hermes_yaml as yaml

_MODEL = {"default": "x", "provider": "custom", "base_url": "http://127.0.0.1:9/v1"}

_STDIO_SERVER = """
import os
from mcp.server import MCPServer
server = MCPServer("gh")

@server.tool()
def whoami() -> str:
    return "GH_TOKEN=" + str(os.environ.get("GH_TOKEN"))

server.run("stdio")
"""

_HTTP_SERVER = """
import asyncio, json, socket, sys
import uvicorn
from mcp.server import MCPServer
log_path, port_path = sys.argv[1], sys.argv[2]
server = MCPServer("team")

@server.tool()
def save_note(text: str) -> str:
    return "saved"

def recording(app):
    async def wrapped(scope, receive, send):
        if scope["type"] == "http" and scope.get("method") == "POST":
            body = b""
            while True:
                event = await receive()
                body += event.get("body", b"")
                if not event.get("more_body"):
                    break
            headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
            try:
                method = json.loads(body).get("method")
            except Exception:
                method = None
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps([method, headers.get("x-hermes-profile")]) + "\\n")
            replayed = False

            async def replay():
                nonlocal replayed
                if replayed:
                    return await receive()
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await app(scope, replay, send)
        return await app(scope, receive, send)
    return wrapped

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
uv = uvicorn.Server(uvicorn.Config(recording(server.streamable_http_app()), log_level="warning"))

async def main():
    task = asyncio.create_task(uv.serve(sockets=[sock]))
    while not uv.started:
        await asyncio.sleep(0.01)
    open(port_path, "w", encoding="utf-8").write(str(sock.getsockname()[1]))
    await task

asyncio.run(main())
"""


@pytest.fixture
def two_profile_homes(tmp_path, monkeypatch):
    """default + worker profile homes under a temp HOME; MCP connections shut down afterwards."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    default_home = tmp_path / ".hermes"
    worker_home = default_home / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("NO_PROXY", "*")
    yield {"default": default_home, "worker": worker_home}
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    shutdown_mcp_servers()


def _discover_and_call(homes: dict, tool: str, args: dict) -> dict:
    import gateway.run as gateway_run
    from gateway.config import GatewayConfig
    from tools.registry import registry

    asyncio.run(gateway_run._discover_gateway_mcp_tools(GatewayConfig(multiplex_profiles=True)))
    results = {}
    for name, home in homes.items():
        with gateway_run._profile_runtime_scope(home):
            results[name] = json.loads(registry.dispatch(tool, args)).get("result")
    return results


def test_profile_with_other_secret_source_value_gets_its_own_stdio_connection(two_profile_homes, tmp_path):
    server = tmp_path / "gh_server.py"
    server.write_text(textwrap.dedent(_STDIO_SERVER), encoding="utf-8")
    for name, home in two_profile_homes.items():
        (home / "secrets.env").write_text(f"GH_TOKEN=fake-token-{name}\n", encoding="utf-8")
        (home / "config.yaml").write_text(yaml.safe_dump({
            "model": _MODEL,
            "secrets": {"command": {"enabled": True, "command": f"cat {home / 'secrets.env'}"}},
            "mcp_servers": {"gh": {"command": sys.executable, "args": [str(server)]}}}), encoding="utf-8")

    results = _discover_and_call(two_profile_homes, "mcp__gh__whoami", {})

    assert results == {"default": "GH_TOKEN=fake-token-default", "worker": "GH_TOKEN=fake-token-worker"}


def test_profile_with_other_profile_identity_header_gets_its_own_http_connection(two_profile_homes, tmp_path):
    log, port_file = tmp_path / "calls.log", tmp_path / "port"
    script = tmp_path / "team_server.py"
    script.write_text(_HTTP_SERVER, encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script), str(log), str(port_file)])
    try:
        deadline = time.monotonic() + 30
        while not (port_file.exists() and port_file.read_text(encoding="utf-8-sig")) and time.monotonic() < deadline:
            time.sleep(0.05)
        port = port_file.read_text(encoding="utf-8-sig")
        team = {"url": f"http://127.0.0.1:{port}/mcp",
                "identity_header": {"name": "X-Hermes-Profile", "value_from": "profile"}}
        for home in two_profile_homes.values():
            (home / "config.yaml").write_text(yaml.safe_dump({"model": _MODEL, "mcp_servers": {"team": team}}), encoding="utf-8")

        _discover_and_call(two_profile_homes, "mcp__team__save_note", {"text": "hi"})

        calls = [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()]
        assert [profile for method, profile in calls if method == "tools/call"] == ["default", "worker"]
    finally:
        proc.terminate()
        proc.wait(10)
