"""The caller's context reaches coroutines scheduled onto the MCP loop.

A per-request profile scope (dashboard ?profile= endpoints, e.g. the MCP
"Test server" probe) and a dashboard OAuth flow are ContextVars on the
calling thread. Anything inside the coroutine that resolves
get_hermes_home() (OAuth token-store paths) or the active dashboard flow
must see the caller's values, not the loop thread's.
"""
import os

import pytest


@pytest.fixture
def mcp_loop():
    from tools.mcp import loop as _mcp_loop

    _mcp_loop._ensure_mcp_loop()
    yield _mcp_loop
    _mcp_loop._stop_mcp_loop()


def test_override_propagates_to_mcp_loop(tmp_path, monkeypatch, mcp_loop):
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    process_home = tmp_path / "proc-home"
    profile_home = tmp_path / "profile-home"
    process_home.mkdir()
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(process_home))

    async def read_home():
        return str(get_hermes_home())

    # Unscoped: the loop task sees the process home.
    assert mcp_loop._run_on_mcp_loop(read_home(), timeout=10) == str(process_home)

    # Scoped: the caller's override must reach the loop task.
    token = set_hermes_home_override(str(profile_home))
    try:
        assert mcp_loop._run_on_mcp_loop(read_home(), timeout=10) == str(profile_home)
        assert mcp_loop._run_on_mcp_loop(lambda: read_home(), timeout=10) == str(
            profile_home
        )
    finally:
        reset_hermes_home_override(token)

    # The loop thread's default context is untouched afterwards.
    assert mcp_loop._run_on_mcp_loop(read_home(), timeout=10) == str(process_home)


def test_concurrent_scopes_do_not_interfere(tmp_path, monkeypatch, mcp_loop):
    """Two threads carrying DIFFERENT overrides scheduling onto the same
    loop must each see their own home."""
    import threading

    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    process_home = tmp_path / "proc-home"
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    for h in (process_home, home_a, home_b):
        h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(process_home))

    async def read_home():
        return str(get_hermes_home())

    results: dict = {}

    def scoped_call(key, home):
        token = set_hermes_home_override(str(home))
        try:
            results[key] = mcp_loop._run_on_mcp_loop(read_home(), timeout=10)
        finally:
            reset_hermes_home_override(token)

    threads = [
        threading.Thread(target=scoped_call, args=("a", home_a)),
        threading.Thread(target=scoped_call, args=("b", home_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert results == {"a": str(home_a), "b": str(home_b)}


def test_dashboard_oauth_flow_propagates_to_mcp_loop(mcp_loop):
    from tools.mcp.dashboard_oauth import dashboard_oauth_flow, get_dashboard_oauth_flow

    async def read_flow():
        return get_dashboard_oauth_flow()

    flow = object()
    assert mcp_loop._run_on_mcp_loop(read_flow(), timeout=10) is None
    with dashboard_oauth_flow(flow):
        assert mcp_loop._run_on_mcp_loop(read_flow(), timeout=10) is flow
        assert mcp_loop._run_on_mcp_loop(lambda: read_flow(), timeout=10) is flow
    assert mcp_loop._run_on_mcp_loop(read_flow(), timeout=10) is None
