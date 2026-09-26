"""A multiplexed profile never adopts another profile's live MCP connection when the inputs that
connection was opened with resolve differently for it: external secret-source env on a stdio
child, or ``identity_header.value_from: profile`` on HTTP. Both profiles have byte-identical
``mcp_servers`` config, so only the resolved values tell the identities apart."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

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


def _adoptable_by_each_profile(homes: dict, name: str, config: dict) -> dict:
    """Record the owner's identity in the default profile's scope, as the connecting task does,
    then ask each profile whether it may adopt that live connection."""
    import gateway.run as gateway_run
    from tools.mcp_tool_registration import _resolved_identity, _same_server_route

    with gateway_run._profile_runtime_scope(homes["default"]):
        owner = SimpleNamespace(name=name, _config=config, _resolved_identity=_resolved_identity(name, config))
    adoptable = {}
    for profile, home in homes.items():
        with gateway_run._profile_runtime_scope(home):
            adoptable[profile] = _same_server_route(
                owner, config, cross_profile=True, resolved_identity=_resolved_identity(name, config))
    return adoptable


def test_profile_whose_bare_npx_resolves_under_its_own_home_does_not_adopt(two_profile_homes, tmp_path):
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    for home in two_profile_homes.values():
        npx = home / "node" / "bin" / "npx"
        npx.parent.mkdir(parents=True)
        npx.write_text("#!/bin/sh\n", encoding="utf-8")
        npx.chmod(0o755)
    config = {"command": "npx", "args": ["-y", "some-server"], "env": {"PATH": str(empty_path)}, "cwd": str(tmp_path)}

    assert _adoptable_by_each_profile(two_profile_homes, "svc", config) == {"default": True, "worker": False}


def test_profile_whose_runtime_file_names_another_endpoint_does_not_adopt(two_profile_homes, tmp_path, monkeypatch):
    import hermes_cli.agent_plugins as agent_plugins
    from hermes_constants import get_hermes_home
    from hermes_platform import declaration

    executable = tmp_path / "example-app"
    executable.write_text("fixture", encoding="utf-8")
    declaration.register("svc", declaration.parse_declaration(
        "Example App", {sys.platform: {"presence": "executable", "location": str(executable)}}, {"app": True},
        where="test"))
    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: {
        "kind": "server_json", "path": str(get_hermes_home() / "server.json")}, raising=False)
    for port, home in enumerate(two_profile_homes.values(), start=4101):
        (home / "server.json").write_text(json.dumps(
            {"http": f"http://127.0.0.1:{port}", "token": f"token-{home.name}", "pid": os.getpid()}), encoding="utf-8")
    try:
        adoptable = _adoptable_by_each_profile(two_profile_homes, "svc", {"url": "http://127.0.0.1:9/mcp"})
    finally:
        declaration.unregister("svc")

    assert adoptable == {"default": True, "worker": False}


def test_published_identity_describes_the_endpoint_the_live_session_connected_to(monkeypatch):
    """The runtime file rotates from endpoint A to B on every read: the identity published with
    the live session must describe the endpoint the transport actually connected with."""
    from tools import mcp_tool, mcp_tool_transport
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_tool_registration import _resolved_identity

    endpoints = iter([("http://127.0.0.1:4101", {"Authorization": "Bearer token-a"}),
                      ("http://127.0.0.1:4102", {"Authorization": "Bearer token-b"})])
    last: list = []

    def rotating(_name):
        last[:] = [next(endpoints, last[0] if last else None)]
        return last[0]

    monkeypatch.setattr(mcp_tool_transport, "_live_endpoint", rotating)
    monkeypatch.setattr(mcp_tool, "_MCP_HTTP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MCP_NEW_HTTP", True)
    live: dict = {}

    class _Task(MCPServerTask):
        async def _prepare_run(self, config):
            self._config = config
            return True

        def _streamable_http_transport(self, url, headers, *_rest):
            live["url"] = url
            return None

        async def _serve_transport(self, _transport, _label, _timeout):
            live["published"] = self._resolved_identity
            self._shutdown_event.set()
            return "shutdown"

    config = {"url": "http://127.0.0.1:9/mcp"}
    asyncio.run(asyncio.wait_for(_Task("svc").run(config), timeout=5))

    monkeypatch.setattr(mcp_tool_transport, "_live_endpoint", lambda _name: last[0])
    assert live["url"] == last[0][0]
    assert live["published"] == _resolved_identity("svc", config)
