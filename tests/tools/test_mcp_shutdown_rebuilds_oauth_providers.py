"""Stopping the MCP event loop (``/reload-mcp``, TUI reload, the idle cleanup after a probe) and
starting the next one must not carry cached OAuth providers across: the manager entry's
``asyncio.Lock`` and the SDK auth-flow state are bound to the loop that created them, and the
first contended 401 then fails with ``Lock ... is bound to a different event loop``."""

import asyncio
import threading

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required for OAuth support")

URL = "https://mcp.example/mcp"


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    """The MCP lifecycle module, with cached tokens for ``srv`` and a fresh OAuth manager."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool_lifecycle
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import reset_manager_for_tests

    tokens = HermesTokenStorage("srv")._tokens_path()
    tokens.parent.mkdir(parents=True, exist_ok=True)
    tokens.write_text('{"access_token": "T", "token_type": "Bearer", "expires_in": 3600}')
    reset_manager_for_tests()
    yield mcp_tool_lifecycle
    mcp_tool_lifecycle.shutdown_mcp_servers()
    reset_manager_for_tests()


@pytest.mark.parametrize(
    "stop",
    [
        pytest.param("shutdown_mcp_servers", id="full-shutdown"),
        pytest.param("_stop_mcp_loop_if_idle", id="idle-cleanup"),
        # Another profile's server is still connecting, so a scoped shutdown leaves the loop up.
        # Its providers stay cached: a second provider for a live server would race the first
        # one's token refresh.
        pytest.param(None, id="scoped-shutdown-keeps-the-loop"),
    ],
)
def test_providers_are_rebuilt_exactly_when_their_loop_stops(lifecycle, stop):
    import tools.mcp_tool as core
    from tools import mcp_tool_loop as _loop
    from tools.mcp_oauth_manager import get_manager

    other = ("other-profile", "other")
    _loop._ensure_mcp_loop()
    first = get_manager().get_or_build_provider("srv", URL, {})
    if stop:
        getattr(lifecycle, stop)()
    else:
        with core._lock:
            core._server_connecting.add(other)
        try:
            lifecycle.shutdown_mcp_servers(scope="this-profile")
        finally:
            with core._lock:
                core._server_connecting.discard(other)
        assert _loop._running_loop() is not None
    _loop._ensure_mcp_loop()
    rebuilt = get_manager().get_or_build_provider("srv", URL, {}) is not first
    assert rebuilt is bool(stop)


def test_teardown_window_neither_sweeps_the_next_loop_nor_lets_the_old_one_repopulate(lifecycle, monkeypatch):
    """The loop handle is detached before the old loop is drained and joined. In that window a
    connection still queued on the old loop resumes (it must not cache a provider bound to the
    dying loop), and a discovery on another thread starts the next loop (its provider must
    survive the rest of the teardown)."""
    from tools import mcp_tool, mcp_tool_loop as _loop
    from tools.mcp_oauth_manager import get_manager

    _loop._ensure_mcp_loop()
    old_loop = _loop._running_loop()
    manager = get_manager()
    old = manager.get_or_build_provider("srv", URL, {})
    resumed, finished = asyncio.Event(), threading.Event()
    built, seen = [], {}

    async def queued_connection():
        await resumed.wait()
        try:
            task = mcp_tool.MCPServerTask("srv")
            task._auth_type = "oauth"
            built.append(task._build_oauth_auth(URL, {}))
        except RuntimeError:
            pass  # an already-detached loop must refuse new providers
        finally:
            finished.set()

    pending = asyncio.run_coroutine_threadsafe(queued_connection(), old_loop)
    clear, reap = manager.clear_cached_providers, lifecycle._kill_orphaned_mcp_children

    def clear_then_resume_old_loop(*args, **kwargs):
        count = clear(*args, **kwargs)
        old_loop.call_soon_threadsafe(resumed.set)
        assert finished.wait(5), "queued connection did not resume in the teardown window"
        return count

    def next_discovery_runs_first(*args, **kwargs):  # the teardown's last step
        if not seen:
            _loop._ensure_mcp_loop()
            seen["provider"] = manager.get_or_build_provider("srv", URL, {})
        return reap(*args, **kwargs)

    monkeypatch.setattr(manager, "clear_cached_providers", clear_then_resume_old_loop)
    monkeypatch.setattr(lifecycle, "_kill_orphaned_mcp_children", next_discovery_runs_first)
    lifecycle.shutdown_mcp_servers()
    pending.result(5)
    monkeypatch.setattr(manager, "clear_cached_providers", clear)  # the old loop is closed now

    assert seen["provider"] is not old, "the next loop was handed the stopped loop's provider"
    assert seen["provider"] not in built, "the stopped loop repopulated the cache"
    assert manager.get_or_build_provider("srv", URL, {}) is seen["provider"]
