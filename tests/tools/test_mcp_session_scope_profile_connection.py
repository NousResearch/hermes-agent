"""A session tool scope reaches the serving profile's own MCP connection, never another's (#143502).

An ACP session whose editor supplies MCP servers runs each turn inside
``session_tool_scope('editor-session:<hash>')``. Under profile multiplexing a profile's config
servers are keyed ``(profile scope, name)``: a call from the session scope must reach the serving
profile's live connection and stay blind to every other profile's (#106005).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def ledgers():
    """Empty MCP connection ledgers for the test, restored afterwards."""
    import tools.mcp_tool as core
    from tools.mcp_tool_scope import _KEYED_LEDGERS, _KEYED_SETS
    names = _KEYED_LEDGERS + _KEYED_SETS
    saved = {name: type(getattr(core, name))(getattr(core, name)) for name in names}
    for name in names:
        getattr(core, name).clear()
    yield
    for name in names:
        getattr(core, name).clear()
        getattr(core, name).update(saved[name])


@pytest.mark.parametrize("owner, reachable", [("a", True), ("b", False)],
                         ids=["serving-profile", "other-profile"])
def test_session_scope_reaches_only_the_serving_profiles_connection(ledgers, tmp_path, monkeypatch,
                                                                    owner, reachable):
    from tools.mcp_tool_discovery import _adopt_server
    from tools.mcp_tool_handlers import _acquire_call_server
    from tools.registry import session_tool_scope

    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    homes = {name: tmp_path / "profiles" / name for name in ("a", "b")}
    for home in homes.values():
        home.mkdir(parents=True)
    token = set_hermes_home_override(homes[owner])
    live = SimpleNamespace(name="tracker", session=object())
    _adopt_server("tracker", live)  # published as discovery does: (owner scope, name)
    reset_hermes_home_override(token)

    token = set_hermes_home_override(homes["a"])  # the turn serves profile a
    try:
        with session_tool_scope("editor-session:" + "ab" * 32):
            server, _ = _acquire_call_server("tracker", 0.01)
    finally:
        reset_hermes_home_override(token)

    assert server is (live if reachable else None)
