"""A full MCP shutdown (``/reload-mcp``, TUI reload) stops the MCP event loop and the next
discovery starts a new one. Cached OAuth providers are bound to the loop that created them
(the manager entry's ``asyncio.Lock`` and ``pending_401`` futures, the SDK auth-flow state),
so carrying one across the restart fails with ``Lock ... is bound to a different event loop``
on the first contended 401. Whatever stops the loop must drop the in-process providers while
keeping the tokens on disk, and a scoped shutdown that leaves the loop running must keep them."""

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


@pytest.mark.parametrize("stop", ["full shutdown", "idle cleanup after a probe"])
def test_stopping_the_loop_rebuilds_oauth_providers_for_the_next_one(oauth_server, stop):
    """Both ways the loop ends: ``shutdown_mcp_servers`` and the ``_stop_mcp_loop_if_idle`` that
    ``hermes mcp login`` / ``test`` and the tool-list probe run when they are done."""
    from tools import mcp_tool_loop as _loop
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool_lifecycle import _stop_mcp_loop_if_idle, shutdown_mcp_servers

    try:
        _loop._ensure_mcp_loop()
        first = get_manager().get_or_build_provider("srv", URL, {})
        _loop._run_on_mcp_loop(_contend_provider_lock(), timeout=10)

        if stop == "full shutdown":
            shutdown_mcp_servers()
        else:
            assert _stop_mcp_loop_if_idle()

        assert oauth_server.exists(), "stopping the loop must not delete persisted OAuth tokens"
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


def test_provider_built_for_the_next_loop_survives_the_old_loops_teardown(oauth_server, monkeypatch):
    """The loop handle is detached before the old loop is drained and joined. A discovery on
    another thread can start the next loop in that window; its provider must not be swept by
    the teardown still in progress. The window is entered through the teardown's last step."""
    from tools import mcp_tool_lifecycle as _lifecycle
    from tools import mcp_tool_loop as _loop
    from tools.mcp_oauth_manager import get_manager

    seen = {}
    reap = _lifecycle._kill_orphaned_mcp_children

    def _next_discovery_runs_first(*args, **kwargs):
        if not seen:
            _loop._ensure_mcp_loop()
            seen["provider"] = get_manager().get_or_build_provider("srv", URL, {})
        return reap(*args, **kwargs)

    try:
        _loop._ensure_mcp_loop()
        old = get_manager().get_or_build_provider("srv", URL, {})
        monkeypatch.setattr(_lifecycle, "_kill_orphaned_mcp_children", _next_discovery_runs_first)

        _lifecycle.shutdown_mcp_servers()

        assert seen["provider"] is not old, "the next loop was handed the stopped loop's provider"
        assert get_manager().get_or_build_provider("srv", URL, {}) is seen["provider"]
    finally:
        _lifecycle.shutdown_mcp_servers()
