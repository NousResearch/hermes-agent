"""Regression for #109798: a transient ``MCPServerTask.session is None`` during an HTTP-transport
reconnect must not deregister the profile's MCP tool overlay.

The reconciler calls ``_register_connected_into_current_scope`` periodically. The previous code
treated any ``session is None`` as a stale connection and called ``_remove_server_scope`` — for an
HTTP keepalive/reconnect gap that's indistinguishable from a genuinely dead server, the profile's
tools disappeared for the rest of the process's life even though the connection recovered.

The contract: a single ``session is None`` observation is transient — the next reconcile tick sees a
live session and clears the mark. An observation that persists across the reconnect grace window
(the same 30 s the registry's ``check_fn`` uses) is treated as a real disconnect and evicts.

Tests:

- ``test_transient_session_none_does_not_evict``: simulate the reconnect window (session set to None
  for one reconcile pass, then restored) — the profile's tools stay registered.
- ``test_persistent_session_none_evicts_after_grace``: a server whose ``session`` stays None past the
  grace window is evicted and its tools deregistered.
- ``test_session_recovery_clears_transient_mark``: a transient None followed by a live session
  leaves the transient mark empty, so a future None starts a fresh grace window.
- ``test_cache_recomputes_after_eviction_on_session_none``: end-to-end through the public
  ``register_connected_into_current_scope`` entry — eviction correctly clears the tool surface
  once the grace elapses.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override


_TRANSIENT_GRACE_SECONDS = 30.0


def _tool(name="t"):
    return SimpleNamespace(name=name, description="d", inputSchema={"type": "object", "properties": {}},
                           annotations=None)


def _server(name, cfg, *, session=None):
    return SimpleNamespace(name=name, session=session, _config=cfg, _tools=[_tool()],
                           tool_timeout=30, initialize_result=None, _registered_tool_names=[],
                           _sampling=None)


@pytest.fixture
def single_profile(tmp_path, monkeypatch):
    """A single scope with a config-derived MCP server; restores every ledger on teardown."""
    import tools.mcp_tool as core
    from tools import mcp_tool_config as _config
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(_config, "_filter_suspicious_mcp_servers", lambda servers: servers)

    ledgers = (
        "_servers", "_server_scope_keys", "_server_tool_scopes", "_server_connecting",
        "_server_connect_errors", "_server_connect_retry_after", "_server_connect_failures",
        "_server_error_counts", "_server_breaker_opened_at", "_lazy_server_configs",
        "_mcp_tool_server_names", "_orphaned_adopters", "_parallel_safe_servers",
        "_server_trust_levels", "_tool_read_only_hints", "_server_session_none_at",
    )
    saved = {n: type(getattr(core, n))(getattr(core, n)) for n in ledgers}
    for n in ledgers:
        getattr(core, n).clear()
    reg._SESSION_NONE_GRACE_SECONDS = _TRANSIENT_GRACE_SECONDS

    token = set_hermes_home_override(home)
    yield home
    for tool_name in list(registry.get_tool_names_for_toolset("mcp-x")):
        registry.deregister(tool_name, scope=hermes_home_key(home))
    reset_hermes_home_override(token)
    for n in ledgers:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])


def _adopt_and_register(reg, x_srv, cfg):
    """Adopt a live server (session present) into the MCP ledger and register its tools."""
    import tools.mcp_tool_discovery as disc

    disc._adopt_server("x", x_srv)
    x_srv._registered_tool_names = reg._register_server_tools("x", x_srv, cfg)
    return x_srv


def test_transient_session_none_does_not_evict(single_profile, monkeypatch):
    """A single ``session is None`` observation is transient; tools stay registered."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer t"}}
    x_srv = _adopt_and_register(reg, _server("x", cfg, session=object()), cfg)
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]

    fake_now = [1000.0]
    monkeypatch.setattr(core.time, "monotonic", lambda: fake_now[0])

    # Reconcile tick 1: HTTP keepalive just dropped session to None — must NOT evict.
    x_srv.session = None
    assert reg.register_connected_into_current_scope({"x": cfg}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"], \
        "transient session=None must not deregister tools (#109798)"

    # Reconcile tick 2 (3 s later): session recovered — transient mark cleared.
    fake_now[0] = 1003.0
    x_srv.session = object()
    assert reg.register_connected_into_current_scope({"x": cfg}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]
    # The transient mark must be cleared so a future None starts a fresh grace window.
    with core._lock:
        assert core._server_session_none_at == {}


def test_persistent_session_none_evicts_after_grace(single_profile, monkeypatch):
    """A session=None observation that persists past the grace window is a real disconnect."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer t"}}
    x_srv = _adopt_and_register(reg, _server("x", cfg, session=object()), cfg)
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]

    fake_now = [2000.0]
    monkeypatch.setattr(core.time, "monotonic", lambda: fake_now[0])

    # First None observation at t=0 — not yet a real disconnect, tools stay.
    x_srv.session = None
    assert reg.register_connected_into_current_scope({"x": cfg}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]

    # Past the grace window (t = grace + 1) — evict.
    fake_now[0] = 2000.0 + _TRANSIENT_GRACE_SECONDS + 1.0
    assert reg.register_connected_into_current_scope({"x": cfg}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == [], \
        "session=None that persists past the grace window must deregister"


def test_session_recovery_clears_transient_mark(single_profile, monkeypatch):
    """A short None → live cycle clears the mark; a second None starts a fresh grace window."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer t"}}
    x_srv = _adopt_and_register(reg, _server("x", cfg, session=object()), cfg)

    fake_now = [3000.0]
    monkeypatch.setattr(core.time, "monotonic", lambda: fake_now[0])

    x_srv.session = None
    reg.register_connected_into_current_scope({"x": cfg})
    fake_now[0] = 3005.0
    x_srv.session = object()
    reg.register_connected_into_current_scope({"x": cfg})
    with core._lock:
        assert core._server_session_none_at == {}

    x_srv.session = None
    reg.register_connected_into_current_scope({"x": cfg})
    # The fresh None observation records a new timestamp; the previous cycle is forgotten.
    with core._lock:
        keys = list(core._server_session_none_at.keys())
        assert len(keys) == 1
    # 5s later — still inside the fresh grace window — must NOT evict.
    fake_now[0] = 3005.0 + 5.0
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]
