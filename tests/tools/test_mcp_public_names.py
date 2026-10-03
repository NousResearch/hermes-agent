"""Public names for reading the MCP client's live state.

A plugin that decides which MCP servers a run may use has to read the process's
live connections under the lock that guards them, scoped to the current
profile, and has to tell "the ``mcp`` package is not installed" apart from
"the server did not connect". These names give it that without reaching into
``_servers`` / ``_lock`` / ``_MCP_AVAILABLE``. The two function aliases are the
SAME object as the private spelling, so internal callers are unchanged.
"""

import pytest

from tools import mcp_tool, mcp_tool_loop, mcp_tool_registration, mcp_tool_scope


@pytest.mark.parametrize(
    ("module", "public", "private"),
    [
        (mcp_tool_loop, "wait_for_server_session_ready", "_wait_for_server_session_ready"),
        (mcp_tool_registration, "register_server_tools", "_register_server_tools"),
    ],
)
def test_public_name_is_the_private_helper(module, public, private):
    assert getattr(module, public) is getattr(module, private)


@pytest.mark.parametrize("flag", [True, False])
def test_mcp_sdk_available_reads_the_import_probe(monkeypatch, flag):
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", flag)
    assert mcp_tool.mcp_sdk_available() is flag


def test_current_mcp_servers_unscoped_returns_every_connection_by_name(monkeypatch):
    alpha, beta = object(), object()
    monkeypatch.setattr(mcp_tool, "_servers", {"alpha": alpha, "beta": beta})
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: None)
    assert mcp_tool_scope.current_mcp_servers() == {"alpha": alpha, "beta": beta}


def test_current_mcp_servers_is_scoped_to_the_current_profile(monkeypatch):
    own, foreign, adopted = object(), object(), object()
    monkeypatch.setattr(mcp_tool, "_servers", {
        ("p1", "alpha"): own,
        ("p2", "alpha"): foreign,
        ("p2", "beta"): adopted,
        ("p2", "gamma"): object(),
    })
    monkeypatch.setattr(mcp_tool, "_server_tool_scopes", {("p2", "beta"): {"p1"}})
    monkeypatch.setattr(mcp_tool, "_lazy_server_configs", {})
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: "p1")
    assert mcp_tool_scope.current_mcp_servers() == {"alpha": own, "beta": adopted}


def test_current_mcp_servers_returns_a_copy(monkeypatch):
    servers = {"alpha": object()}
    monkeypatch.setattr(mcp_tool, "_servers", servers)
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: None)
    mcp_tool_scope.current_mcp_servers().clear()
    assert "alpha" in servers
