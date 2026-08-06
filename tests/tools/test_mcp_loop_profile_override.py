"""Regression tests for HERMES_HOME override propagation onto the MCP loop.

Tasks scheduled via run_coroutine_threadsafe are created inside the MCP
event-loop thread, so they copy THAT thread's context — not the scheduling
thread's. A per-request profile scope (dashboard ?profile= endpoints, e.g.
the MCP "Test server" probe) would silently vanish for anything resolving
get_hermes_home() inside the coroutine, most visibly OAuth token-store
paths. _run_on_mcp_loop now wraps scheduled coroutines with the caller's
override (mcp_tool._wrap_with_home_override).
"""
import os

import pytest


@pytest.fixture
def mcp_loop():
    import tools.mcp_tool as mcp_tool

    mcp_tool._ensure_mcp_loop()
    yield mcp_tool
    mcp_tool._stop_mcp_loop()


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
        # Factory form must be wrapped too.
        assert mcp_loop._run_on_mcp_loop(lambda: read_home(), timeout=10) == str(
            profile_home
        )
    finally:
        reset_hermes_home_override(token)

    # The loop thread's default context is untouched afterwards.
    assert mcp_loop._run_on_mcp_loop(read_home(), timeout=10) == str(process_home)


def test_oauth_token_paths_follow_override(tmp_path, monkeypatch, mcp_loop):
    """The actual symptom path: HermesTokenStorage resolving inside the
    probe's MCP-loop coroutine must land in the selected profile's
    mcp-tokens dir, not the process home's."""
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    process_home = tmp_path / "proc-home"
    profile_home = tmp_path / "profile-home"
    process_home.mkdir()
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(process_home))

    async def token_path():
        from tools.mcp_oauth import HermesTokenStorage

        return str(HermesTokenStorage("probe-srv")._tokens_path())

    token = set_hermes_home_override(str(profile_home))
    try:
        path = mcp_loop._run_on_mcp_loop(token_path(), timeout=10)
    finally:
        reset_hermes_home_override(token)
    assert path.startswith(str(profile_home))
    assert os.path.join("mcp-tokens", "probe-srv.json") in path


def test_concurrent_scopes_do_not_interfere(tmp_path, monkeypatch, mcp_loop):
    """Two threads carrying DIFFERENT overrides scheduling onto the same
    loop must each see their own home — the wrapper is task-local."""
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


def test_wrap_is_noop_without_override(mcp_loop):
    """No active override → the coroutine passes through unwrapped."""

    async def trivial():
        return 42

    coro = trivial()
    wrapped = mcp_loop._wrap_with_home_override(coro)
    assert wrapped is coro
    coro.close()
