"""Exercise CLI device login and cold OAuth discovery over the real byte route.

Linux substitutes only the Windows executable; sockets, MCP SDK, HTTP, OAuth
requests, scoped storage, refresh and the shared MCP task are not mocked.
"""
import asyncio
import json
import sys
import time
from urllib.parse import urlsplit

import pytest

from evals.mcp_device_flow import DEVICE_GRANT, oauth_fixture

_BYTE_RELAY = r'''
import os, socket, sys, threading
s = socket.create_connection(("127.0.0.1", int(sys.argv[1])))
sys.stdout.buffer.write(b"\0"); sys.stdout.buffer.flush()
def receive():
    try:
        while True:
            chunk = s.recv(65536)
            if not chunk: break
            sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
    except (OSError, BrokenPipeError):
        pass
threading.Thread(target=receive, daemon=True).start()
try:
    while True:
        data = os.read(0, 65536)
        if not data: break
        s.sendall(data)
finally:
    s.close()
'''


@pytest.mark.linux_only
@pytest.mark.asyncio
@pytest.mark.parametrize("network", ["auto", "windows"])
async def test_cli_device_login_and_cold_refresh_share_windows_routing(tmp_path, monkeypatch, network):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.mcp_config import _reauth_oauth_server
    from tools import mcp_windows
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = set_hermes_home_override(tmp_path)
    monkeypatch.setattr(mcp_windows, "is_wsl", lambda: True)
    cfg = {"url": "http://localhost:1/mcp", "auth": "oauth", "protocol": "legacy", "network": network,
           "oauth": {"flow": "device", "cimd": False, "timeout": 15, "scope": "fixture.read"}}
    task = None
    try:
        with oauth_fixture("network", advertised_base="http://localhost:1") as (direct, wire):
            port = urlsplit(direct).port
            monkeypatch.setattr(mcp_windows, "_windows_tunnel_command",
                                lambda *_: [sys.executable, "-u", "-c", _BYTE_RELAY, str(port)])
            assert await asyncio.wait_for(asyncio.to_thread(_reauth_oauth_server, "network-fixture", cfg, flow="device"), timeout=35)
            assert any(row.get("data", {}).get("grant_type") == DEVICE_GRANT for row in wire)
            tokens_path = tmp_path / "mcp-tokens" / "network-fixture.json"
            saved = json.loads(tokens_path.read_text(encoding="utf-8"))
            assert saved["access_token"] == "fixture-access"
            saved["expires_at"] = time.time() - 60
            tokens_path.write_text(json.dumps(saved), encoding="utf-8")
            (tmp_path / "mcp-tokens" / "network-fixture.meta.json").unlink(missing_ok=True)
            get_manager().evict("network-fixture")
            boundary = len(wire)
            task = MCPServerTask("network-fixture")
            await asyncio.wait_for(task.start(cfg), timeout=20)
            assert [tool.name for tool in task._tools] == ["fixture_echo"]
            fresh = wire[boundary:]
            assert any("oauth-protected-resource" in row["path"] for row in fresh)
            assert any("oauth-authorization-server" in row["path"] for row in fresh)
            assert any(row.get("data", {}).get("grant_type") == "refresh_token" for row in fresh)
            await task.shutdown()
            task = None
    finally:
        if task is not None:
            await task.shutdown()
        get_manager().remove("network-fixture")
        reset_hermes_home_override(token)


@pytest.mark.linux_only
@pytest.mark.asyncio
async def test_legacy_omitted_network_device_oauth_sends_nothing_to_windows_canary(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.mcp_config import _reauth_oauth_server
    from tools import mcp_windows
    from tools.mcp_oauth_manager import get_manager

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = set_hermes_home_override(tmp_path)
    monkeypatch.setattr(mcp_windows, "is_wsl", lambda: True)
    cfg = {"url": "http://localhost:1/mcp", "auth": "oauth", "protocol": "legacy",
           "oauth": {"flow": "device", "cimd": False, "timeout": 3, "scope": "fixture.read"}}
    bridge_calls = []
    try:
        with oauth_fixture("legacy-network", advertised_base="http://localhost:1") as (direct, wire):
            canary_port = urlsplit(direct).port

            def windows_canary(host, advertised_port):
                bridge_calls.append((host, advertised_port))
                return [sys.executable, "-u", "-c", _BYTE_RELAY, str(canary_port)]

            monkeypatch.setattr(mcp_windows, "_windows_tunnel_command", windows_canary)
            assert not await asyncio.wait_for(
                asyncio.to_thread(_reauth_oauth_server, "legacy-network-fixture", cfg, flow="device"),
                timeout=15,
            )
            assert bridge_calls == []
            assert wire == []
    finally:
        get_manager().remove("legacy-network-fixture")
        reset_hermes_home_override(token)


def test_oauth_provider_cache_is_rebuilt_for_network_and_tls_not_unrelated_fields(tmp_path, monkeypatch):
    from tools.mcp_oauth_manager import MCPOAuthManager
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = MCPOAuthManager()
    monkeypatch.setattr(manager, "_build_provider", lambda *_: type("FixtureProvider", (), {})())
    first = manager.get_or_build_provider("fixture", "http://localhost/mcp", {}, http_config={})
    same = manager.get_or_build_provider("fixture", "http://localhost/mcp", {}, http_config={"network": "local", "enabled": False})
    other = manager.get_or_build_provider("fixture", "http://localhost/mcp", {}, http_config={"network": "windows"})
    tls = manager.get_or_build_provider("fixture", "http://localhost/mcp", {}, http_config={"network": "windows", "ssl_verify": False})
    assert first is same
    assert other is not first
    assert tls is not other
