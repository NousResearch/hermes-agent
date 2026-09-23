"""Behavior tests for manage_connections.

DI-callable idiom: a fake client injected through manage_connections'
seams; no module mocks, no network.
"""

import json
from unittest.mock import patch

import pytest

import tools.connectors.tool  # registers the tool
from tools.connectors.tool import MANAGE_CONNECTIONS_SCHEMA, manage_connections


class FakeClient:
    def __init__(self):
        self.calls = []

    def list_connectors(self, **_):
        self.calls.append(("list",))
        return [
            {"connector": "gmail", "enabled": True, "connected": False},
            {"connector": "linear", "enabled": True, "connected": True},
            {"connector": "googlecalendar", "enabled": True, "connected": False},
        ]

    def connections(self, connectors, *, reinitiate=False):
        self.calls.append(("connections", tuple(connectors), reinitiate))
        return {
            "results": [
                {
                    "connector": c,
                    "status": "initiated",
                    "connect_url": f"https://connect.example/{c}",
                    "instruction": f"finish authorizing {c} in the browser",
                    "reinitiated": reinitiate,
                }
                for c in connectors
            ],
            "summary": {"total": len(connectors), "initiated": len(connectors)},
        }


def test_status_lists_and_filters_connectors():
    client = FakeClient()
    out = json.loads(
        manage_connections(
            {"action": "status", "connectors": ["GMAIL"]},
            client_factory=lambda: client,
        )
    )
    assert out["connectors"] == [
        {"connector": "gmail", "enabled": True, "connected": False}
    ]


def test_status_names_unknown_connectors_and_suggests_close_matches():
    client = FakeClient()
    out = json.loads(
        manage_connections(
            {"action": "status", "connectors": ["google-calendar", "slack"]},
            client_factory=lambda: client,
        )
    )

    assert out["connectors"] == []
    assert out["unknown"] == [
        {"name": "google-calendar", "did_you_mean": "googlecalendar"},
        {"name": "slack"},
    ]


def test_connect_off_desktop_returns_a_link_per_target():
    client = FakeClient()
    out = json.loads(
        manage_connections(
            {"action": "connect", "connectors": ["gmail"]},
            client_factory=lambda: client,
        )
    )
    assert out["targets"][0]["connect_url"] == "https://connect.example/gmail"
    assert out["status"] == "initiated"
    assert client.calls == [("connections", ("gmail",), False)]


def test_reconnect_sets_reinitiate():
    client = FakeClient()
    manage_connections(
        {"action": "reconnect", "connectors": ["gmail"]},
        client_factory=lambda: client,
    )
    assert ("connections", ("gmail",), True) in client.calls


def test_reconnect_on_a_connected_target_makes_no_mint():
    client = FakeClient()
    manage_connections(
        {"action": "reconnect", "connectors": ["linear"]},
        client_factory=lambda: client,
    )
    assert not any(call[0] == "connections" for call in client.calls)


def test_connect_without_connectors_is_a_usage_error():
    out = json.loads(
        manage_connections({"action": "connect"}, client_factory=FakeClient)
    )
    assert "requires 'connectors'" in out["error"]


def test_disconnect_is_refused_before_any_gateway_call():
    # De-authentication is user-only: the tool rejects it up front and the
    # gateway never hears about it.
    client = FakeClient()
    out = json.loads(
        manage_connections(
            {"action": "disconnect", "connectors": ["gmail"]},
            client_factory=lambda: client,
        )
    )
    assert "error" in out
    assert client.calls == []


def test_gateway_failure_is_a_model_actionable_error():
    def exploding():
        raise RuntimeError("gateway on fire")

    out = json.loads(
        manage_connections({"action": "status"}, client_factory=exploding)
    )
    assert "connector gateway request failed" in out["error"]


def test_mcp_actions_belong_to_mcp_targets_only():
    # The MCP verbs are in the enum for mcp:true targets only. The callback that an earlier
    # fold could not reach through registry.dispatch now arrives via the inline executor.
    enum = MANAGE_CONNECTIONS_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert {"install", "enable", "authorize"} <= set(enum)

    out = json.loads(manage_connections({"action": "install", "connectors": ["linear"]}))
    assert "mcp" in out["error"] and "install" in out["error"]

    out = json.loads(
        manage_connections({"action": "connect", "connectors": [{"name": "linear", "mcp": True}]})
    )
    assert "managed-connector action" in out["error"]

    out = json.loads(manage_connections({"action": "uninstall", "connectors": ["gmail"]}))
    assert "action must be one of" in out["error"]


# ---------------------------------------------------------------------------
# blocking operations are never parallelized
# ---------------------------------------------------------------------------


def test_wait_never_rides_a_parallel_batch():
    """A three-minute block must not hold a gathered batch's siblings hostage."""
    from agent.tool_dispatch_helpers import _NEVER_PARALLEL_TOOLS

    assert "manage_connections" in _NEVER_PARALLEL_TOOLS


# ---------------------------------------------------------------------------
# reachability: a registered tool nobody enables is a tool nobody can call
# ---------------------------------------------------------------------------


def _session_tool_names(enabled_toolsets, *, connectors, disabled_toolsets=None):
    """Tool names a session would actually receive, through the real assembly.

    Skips the tool_search step so the assertion is about NAME resolution and
    check_fn, not about how many MCP servers the developer running the suite
    happens to have configured.
    """
    from model_tools import _compute_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    with patch("tools.connectors.gateway.config.connectors_available",
               return_value=connectors):
        invalidate_check_fn_cache()
        try:
            defs = _compute_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
        finally:
            invalidate_check_fn_cache()
    return {d["function"]["name"] for d in defs}



def test_cli_session_gets_the_tool_outside_a_code_workspace(tmp_path, monkeypatch):
    """The path a plain `hermes` run takes: _get_platform_tools, no git cwd."""
    from hermes_cli.tools_config import _get_platform_tools

    monkeypatch.chdir(tmp_path)
    enabled = sorted(_get_platform_tools({}, "cli", include_default_mcp_servers=True))

    assert "connections" in enabled
    assert "manage_connections" in _session_tool_names(enabled, connectors=True)


def test_cli_session_gets_the_tool_inside_a_code_workspace(monkeypatch):
    """Same resolver, run from this repo — the surface the live miss was on."""
    from pathlib import Path

    from hermes_cli.tools_config import _get_platform_tools

    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    enabled = sorted(_get_platform_tools({}, "cli", include_default_mcp_servers=True))
    assert "manage_connections" in _session_tool_names(enabled, connectors=True)


def test_tui_and_desktop_sessions_get_the_tool(monkeypatch):
    """The path the TUI/desktop gateway takes to build its selection."""
    from tui_gateway.server import _load_enabled_toolsets

    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)
    for platform in ("tui", "desktop"):
        selection = _load_enabled_toolsets(platform)
        names = _session_tool_names(selection, connectors=True)
        assert "manage_connections" in names, platform


def test_focus_mode_coding_posture_gets_the_tool(monkeypatch):
    """An engineer pinned to the coding posture still sees their accounts."""
    from pathlib import Path

    from agent.coding_context import coding_selection

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo)
    selection = coding_selection(
        platform="cli", cwd=str(repo), config={"agent": {"coding_context": "focus"}}
    )
    assert selection == ["coding"]  # posture collapse still collapses
    assert "manage_connections" in _session_tool_names(selection, connectors=True)


def test_session_the_portal_has_not_enabled_never_receives_the_tool(tmp_path, monkeypatch):
    """The portal gate is the tool's check_fn: a session whose account the portal has not enabled
    for connectors does not get ``manage_connections`` in its schema on any surface, so the
    model cannot call it and read the gateway's 404 back to the user. The handler keeps the same
    gate for the direct RPC path."""
    from hermes_cli.tools_config import _get_platform_tools
    from tools.registry import registry
    from tui_gateway.server import _load_enabled_toolsets

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)
    selections = [
        sorted(_get_platform_tools({}, "cli", include_default_mcp_servers=True)),
        _load_enabled_toolsets("tui"),
        _load_enabled_toolsets("desktop"),
        ["coding"],
    ]
    for selection in selections:
        assert "manage_connections" not in _session_tool_names(selection, connectors=False), selection

    with patch("tools.connectors.gateway.config.connectors_available", return_value=False):
        out = json.loads(registry.dispatch("manage_connections", {"action": "status"}))
    assert "not available in this session" in out["error"]


def test_operator_can_still_turn_it_off(tmp_path, monkeypatch):
    """`agent.disabled_toolsets: [connections]` wins; a bundle name does not.

    The name is added before the disabled subtraction, so the toolset behaves
    like any other. Naming a platform composite instead must NOT strip it —
    that branch preserves core tools on purpose (#33924).
    """
    from hermes_cli.tools_config import _get_platform_tools

    monkeypatch.chdir(tmp_path)
    enabled = sorted(_get_platform_tools({}, "cli", include_default_mcp_servers=True))

    assert "manage_connections" not in _session_tool_names(
        enabled, connectors=True, disabled_toolsets=["connections"]
    )
    assert "manage_connections" in _session_tool_names(
        enabled, connectors=True, disabled_toolsets=["hermes-cli"]
    )


def test_tool_is_never_deferrable():
    from tools.tool_search import is_deferrable_tool_name

    # Core names short-circuit before the toolset check, so listing
    # "connections" in _DIRECT_SURFACE_TOOLSETS would be redundant.
    assert is_deferrable_tool_name("manage_connections") is False
