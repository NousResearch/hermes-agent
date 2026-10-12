"""Stopping the MCP event loop (``/reload-mcp``, TUI reload, the idle cleanup after a probe) and
starting the next one must not carry cached OAuth providers across: the manager entry's
``asyncio.Lock`` and the SDK auth-flow state are bound to the loop that created them, and the
first contended 401 then fails with ``Lock ... is bound to a different event loop``."""

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


@pytest.mark.parametrize("stop", ["shutdown_mcp_servers", "_stop_mcp_loop_if_idle"])
def test_stopping_the_loop_rebuilds_oauth_providers_for_the_next_one(lifecycle, stop):
    from tools import mcp_tool_loop as _loop
    from tools.mcp_oauth_manager import get_manager

    _loop._ensure_mcp_loop()
    first = get_manager().get_or_build_provider("srv", URL, {})
    getattr(lifecycle, stop)()
    _loop._ensure_mcp_loop()
    assert get_manager().get_or_build_provider("srv", URL, {}) is not first


def test_provider_built_for_the_next_loop_survives_the_old_loops_teardown(lifecycle, monkeypatch):
    """The loop handle is detached before the old loop is drained and joined. A discovery on
    another thread can start the next loop in that window; its provider must not be swept by
    the teardown still in progress. The window is entered through the teardown's last step."""
    from tools import mcp_tool_loop as _loop
    from tools.mcp_oauth_manager import get_manager

    seen = {}
    reap = lifecycle._kill_orphaned_mcp_children

    def _next_discovery_runs_first(*args, **kwargs):
        if not seen:
            _loop._ensure_mcp_loop()
            seen["provider"] = get_manager().get_or_build_provider("srv", URL, {})
        return reap(*args, **kwargs)

    _loop._ensure_mcp_loop()
    old = get_manager().get_or_build_provider("srv", URL, {})
    monkeypatch.setattr(lifecycle, "_kill_orphaned_mcp_children", _next_discovery_runs_first)
    lifecycle.shutdown_mcp_servers()

    assert seen["provider"] is not old, "the next loop was handed the stopped loop's provider"
    assert get_manager().get_or_build_provider("srv", URL, {}) is seen["provider"]
