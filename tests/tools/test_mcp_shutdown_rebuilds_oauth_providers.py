"""A full MCP shutdown (``/reload-mcp``, TUI reload) stops the MCP event loop and the next
discovery starts a new one. Cached OAuth providers are bound to the loop that created them
(the manager entry's ``asyncio.Lock`` and ``pending_401`` futures, the SDK auth-flow state),
so carrying one across the restart fails with ``Lock ... is bound to a different event loop``
on the first contended 401. The shutdown must drop the in-process providers while keeping the
tokens on disk, and a scoped shutdown that leaves the loop running must keep them."""

import asyncio

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required for OAuth support")

URL = "https://mcp.example/mcp"


@pytest.fixture
def oauth_server(tmp_path, monkeypatch):
    """A server with cached tokens on disk and a fresh process-wide OAuth manager."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import reset_manager_for_tests

    tokens = HermesTokenStorage("srv")._tokens_path()
    tokens.parent.mkdir(parents=True, exist_ok=True)
    tokens.write_text('{"access_token": "T", "token_type": "Bearer", "expires_in": 3600}')
    reset_manager_for_tests()
    yield tokens
    reset_manager_for_tests()


def _contend_provider_lock():
    """Two waiters on the provider entry's lock, as concurrent 401s on the MCP loop produce."""
    from tools.mcp_oauth_manager import get_manager

    async def contend():
        manager = get_manager()
        entry = manager._entries[manager._key("srv")]

        async def hold():
            async with entry.lock:
                await asyncio.sleep(0)

        await asyncio.gather(hold(), hold())

    return contend


def test_full_shutdown_rebuilds_oauth_providers_for_the_next_loop(oauth_server):
    from tools import mcp_tool_loop as _loop
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    try:
        _loop._ensure_mcp_loop()
        first = get_manager().get_or_build_provider("srv", URL, {})
        _loop._run_on_mcp_loop(_contend_provider_lock(), timeout=10)

        shutdown_mcp_servers()

        assert oauth_server.exists(), "a shutdown must not delete persisted OAuth tokens"
        _loop._ensure_mcp_loop()
        second = get_manager().get_or_build_provider("srv", URL, {})
        _loop._run_on_mcp_loop(_contend_provider_lock(), timeout=10)
        assert second is not first
    finally:
        shutdown_mcp_servers()


def test_scoped_shutdown_keeps_providers_while_the_loop_runs(oauth_server):
    import tools.mcp_tool as core
    from tools import mcp_tool_loop as _loop
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    other_owner = ("other-profile", "other")
    try:
        _loop._ensure_mcp_loop()
        provider = get_manager().get_or_build_provider("srv", URL, {})
        with core._lock:
            core._server_connecting.add(other_owner)

        shutdown_mcp_servers(scope="this-profile")

        assert _loop._running_loop() is not None
        assert get_manager().get_or_build_provider("srv", URL, {}) is provider
    finally:
        with core._lock:
            core._server_connecting.discard(other_owner)
        shutdown_mcp_servers()
