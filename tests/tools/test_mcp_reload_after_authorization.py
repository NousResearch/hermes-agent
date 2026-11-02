"""A saved OAuth grant wakes only its owner's parked connection before reload reports readiness."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.waiting import wait_until


@pytest.mark.no_isolate
@pytest.mark.parametrize("connects", [True, False])
def test_reload_adopts_new_credentials_and_waits_for_tools(monkeypatch, tmp_path, connects):
    from hermes_constants import hermes_home_key
    from mcp.types import Tool
    from tools import mcp_tool, mcp_tool_config, mcp_tool_discovery as discovery, mcp_tool_loop
    from tools.mcp_oauth import HermesTokenStorage, OAuthNonInteractiveError

    # Registry scopes are resolved home keys: a registration under a made-up label would land in an
    # overlay the active home never reads, so the owner scope IS this home and the sibling another.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    owner, other = hermes_home_key(tmp_path), hermes_home_key(tmp_path / "profiles" / "sibling")
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: owner)
    config = {"url": "https://mcp.example.test/mcp", "auth": "oauth", "connect_timeout": 2}
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config", lambda: {"shop": config})
    storage = HermesTokenStorage("shop", hermes_home=tmp_path)
    entered, release = threading.Event(), asyncio.Event()
    attempts = []

    class Server(mcp_tool.MCPServerTask):
        async def _run_http(self, cfg):
            attempts.append(1)
            if not storage.has_cached_tokens():
                raise OAuthNonInteractiveError("authorization required")
            entered.set()
            await release.wait()
            if not connects:
                raise OAuthNonInteractiveError("provider denied the saved grant")
            self.session = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(
                tools=[Tool(name="catalog", description="Read the synthetic catalog", inputSchema={})],
                next_cursor=None)))
            await self._discover_tools()
            self._ready.set()
            return await self._wait_for_lifecycle_event()

    mcp_tool_loop._ensure_mcp_loop()
    server, sibling = Server("shop"), mcp_tool.MCPServerTask("shop")
    key, sibling_key = (owner, "shop"), (other, "shop")
    monkeypatch.setitem(mcp_tool._servers, key, server)
    monkeypatch.setitem(mcp_tool._server_scope_keys, key, owner)
    # The live connect path recorded the initial failure (``_note_connect_failure``); the error entry
    # alone keeps a server "not connected" until a published session clears it.
    monkeypatch.setitem(mcp_tool._server_connect_errors, key, "authorization required")
    monkeypatch.setitem(mcp_tool._servers, sibling_key, sibling)
    monkeypatch.setitem(mcp_tool._server_scope_keys, sibling_key, other)
    sibling._auth_park = (storage, storage.credential_stamp())
    sibling._park_reason = "authentication required"

    async def start():
        server._task = asyncio.create_task(server.run(config))

    asyncio.run_coroutine_threadsafe(start(), mcp_tool._mcp_loop).result(5)
    try:
        wait_until(lambda: server._auth_park and server._park_reason, what="initial authentication park")
        assert discovery.reconcile_mcp_servers_with_config()["added"] == []
        assert len(attempts) == 1
        token_file = storage._tokens_path()
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(json.dumps({"access_token": "synthetic", "token_type": "Bearer"}))

        with ThreadPoolExecutor(max_workers=2) as executor:
            reload = executor.submit(discovery.reconcile_mcp_servers_with_config)
            try:
                assert entered.wait(5), "reload must wake the existing parked task after authorization"
                assert not reload.done(), "reload must wait for tool discovery, not just signal reconnect"
                concurrent_reload = executor.submit(discovery.reconcile_mcp_servers_with_config)
            finally:
                mcp_tool._mcp_loop.call_soon_threadsafe(release.set)
            result = reload.result(5)
            assert concurrent_reload.result(5)["added"] == []

        assert result["added"] == (["shop"] if connects else [])
        assert discovery.mcp_servers_not_connected() == ([] if connects else ["shop"])
        assert discovery.mcp_servers_ready() == (["shop"] if connects else [])
        if connects:
            assert server._registered_tool_names, "the live registry must contain the discovered tools"
            assert key not in mcp_tool._server_connect_errors, "the recorded initial failure must clear"
            assert server._park_reason, "readiness must not erase the transport's unproven failure budget"
        assert not sibling._reconnect_event.is_set(), "another profile's same-named server is untouched"
        assert len(attempts) == 2
        assert discovery.reconcile_mcp_servers_with_config()["added"] == []
        assert len(attempts) == 2, "unchanged credentials must not start another reconnect"
    finally:
        mcp_tool._mcp_loop.call_soon_threadsafe(release.set)
        asyncio.run_coroutine_threadsafe(server.shutdown(), mcp_tool._mcp_loop).result(5)
        mcp_tool._servers.pop(key, None)
        mcp_tool._servers.pop(sibling_key, None)
        mcp_tool_loop._stop_mcp_loop()


@pytest.mark.parametrize("condition", ["unchanged", "disabled", "live", "foreign", "non_auth", "missing"])
def test_reload_does_not_reconnect_unrelated_or_unchanged_servers(monkeypatch, tmp_path, condition):
    from tools import mcp_tool, mcp_tool_config, mcp_tool_discovery as discovery, mcp_tool_loop
    from tools.mcp_oauth import HermesTokenStorage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: "owner")
    config = {"url": "https://mcp.example.test/mcp", "auth": "oauth", "enabled": condition != "disabled"}
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config", lambda: {"shop": config})
    storage = HermesTokenStorage("shop", hermes_home=tmp_path)
    server = mcp_tool.MCPServerTask("shop")
    server._park_reason = "authentication required"
    server._auth_park = None if condition == "non_auth" else (storage, storage.credential_stamp())
    if condition == "missing":
        client_file = storage._client_info_path()
        client_file.parent.mkdir(parents=True, exist_ok=True)
        client_file.write_text('{"client_id":"synthetic"}')
    elif condition != "unchanged":
        token_file = storage._tokens_path()
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text('{"access_token":"synthetic","token_type":"Bearer"}')
    if condition == "live":
        server.session = object()
        server._ready.set()
    key = ("foreign" if condition == "foreign" else "owner", "shop")
    monkeypatch.setitem(mcp_tool._servers, key, server)
    monkeypatch.setitem(mcp_tool._server_scope_keys, key, key[0])
    monkeypatch.setitem(mcp_tool._server_tool_scopes, key, {"owner", key[0]})
    monkeypatch.setattr(discovery, "discover_mcp_tools", list)
    monkeypatch.setattr(discovery._lifecycle, "shutdown_mcp_servers", lambda **kw: None)

    def unexpected_reconnect(*args, **kwargs):
        pytest.fail("reload must not reconnect this server")

    monkeypatch.setattr(mcp_tool_loop, "_signal_reconnect_and_wait", unexpected_reconnect)
    discovery.reconcile_mcp_servers_with_config()
