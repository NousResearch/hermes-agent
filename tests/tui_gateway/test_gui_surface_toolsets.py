"""GUI capability follows the SESSION's client, not the backend's process env.

The desktop app is a client. It can drive a backend that Electron spawned
locally, one reached over SSH, one behind a plain URL+token, or Hermes Cloud —
and only the first two run with ``HERMES_DESKTOP=1`` in their environment.
Gating the pane/browser/reaction tools on that env var therefore stripped every
one of them from URL and cloud gateways, while the same backend still told the
model "You are chatting inside the Hermes desktop app".

These tests pin the contract that replaced it: eligibility is resolved from the
session's own ``source`` (``session.create``'s ``source: 'desktop'``), so the
answer is identical on every connection topology.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tui_gateway.server as server
from toolsets import TOOLSETS, resolve_toolset

GUI_TOOLS = {
    "annotate_preview",
    "close_preview",
    "drive_preview",
    "close_terminal",
    "focus_pane",
    "open_preview",
    "read_preview",
    "read_terminal",
    "read_window_below",
    "react_to_message",
    "setup_mcp",
    "tour",
}


@pytest.fixture
def no_desktop_env(monkeypatch):
    """A backend nobody told about the desktop — i.e. every remote gateway."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)
    return monkeypatch


class TestDesktopUiToolset:
    def test_holds_exactly_the_gui_affordances(self):
        assert set(resolve_toolset("desktop_ui")) == GUI_TOOLS

    def test_stays_off_the_core_tool_list(self):
        """Core ships on every API call — a GUI-only tool must not be there."""
        from toolsets import _HERMES_CORE_TOOLS

        assert GUI_TOOLS.isdisjoint(_HERMES_CORE_TOOLS)

    def test_no_platform_bundle_carries_it(self):
        """Messaging/CLI bundles must not pick these up by listing them."""
        for name, spec in TOOLSETS.items():
            if name == "desktop_ui":
                continue
            assert GUI_TOOLS.isdisjoint(set(spec.get("tools") or ())), name


class TestSurfaceResolution:
    def test_desktop_session_gets_them_with_no_desktop_env(self, no_desktop_env):
        """THE regression: a desktop client on a remote/cloud backend."""
        assert "desktop_ui" in server._gui_surface_toolsets("desktop")

    def test_tui_session_does_not(self, no_desktop_env):
        assert "desktop_ui" not in server._gui_surface_toolsets("tui")

    def test_desktop_env_alone_does_not_grant_them(self, no_desktop_env):
        """A desktop-spawned backend serving a TUI session stays clean.

        The embedded terminal pane runs `hermes --tui` against this same
        backend; env-keyed gating handed it GUI tools it cannot answer.
        """
        no_desktop_env.setenv("HERMES_DESKTOP", "1")
        assert "desktop_ui" not in server._gui_surface_toolsets("tui")

    def test_project_tools_ride_on_every_gui_surface(self, no_desktop_env):
        for platform in ("desktop", "tui"):
            assert "project" in server._gui_surface_toolsets(platform)


class TestResolverPlumbing:
    def test_posture_path_folds_in_the_session_surface(self, no_desktop_env):
        """Focus-mode returns early — the surface toolsets must survive it."""
        import agent.coding_context as cc

        no_desktop_env.setattr(cc, "coding_selection", lambda **_: ["coding"])

        assert server._load_enabled_toolsets("desktop") == [
            "coding",
            "desktop_ui",
            "project",
        ]
        assert server._load_enabled_toolsets("tui") == ["coding", "project"]

    def test_config_path_folds_in_the_session_surface(self, no_desktop_env):
        import agent.coding_context as cc
        import hermes_cli.config as config_mod

        no_desktop_env.setattr(cc, "coding_selection", lambda **_: None)
        no_desktop_env.setattr(
            config_mod, "load_config", lambda: {"platform_toolsets": {"cli": ["memory"]}}
        )

        desktop = server._load_enabled_toolsets("desktop")
        tui = server._load_enabled_toolsets("tui")

        assert desktop is not None and tui is not None
        assert "desktop_ui" in desktop
        assert "desktop_ui" not in tui

    def test_explicit_empty_config_overrides_posture_and_gui_surfaces(
        self, no_desktop_env
    ):
        import agent.coding_context as cc
        import hermes_cli.config as config_mod

        no_desktop_env.setattr(cc, "coding_selection", lambda **_: ["coding"])
        no_desktop_env.setattr(
            config_mod, "load_config", lambda: {"platform_toolsets": {"cli": []}}
        )

        assert server._load_enabled_toolsets("desktop") == []
        assert server._load_enabled_toolsets("tui") == []

    def test_invalid_config_fails_closed_without_gui_surfaces(self, no_desktop_env):
        import agent.coding_context as cc
        import hermes_cli.config as config_mod

        no_desktop_env.setattr(cc, "coding_selection", lambda **_: None)
        no_desktop_env.setattr(
            config_mod,
            "load_config",
            lambda: {"platform_toolsets": {"cli": ["not-a-real-toolset"]}},
        )

        assert server._load_enabled_toolsets("desktop") == []

    def test_explicit_env_pin_still_wins(self, no_desktop_env):
        """HERMES_TUI_TOOLSETS is an operator override; surface can't re-add."""
        no_desktop_env.setenv("HERMES_TUI_TOOLSETS", "web,memory")

        assert server._load_enabled_toolsets("desktop") == ["web", "memory"]


def test_make_agent_preserves_explicit_empty_toolsets(monkeypatch):
    """A fresh desktop session must construct its live agent without tools."""
    import agent.coding_context as cc
    import hermes_cli.config as config_mod

    config = {"platform_toolsets": {"cli": []}}
    runtime = SimpleNamespace(
        runtime={
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
            "credential_pool": None,
        },
        used_fallback=False,
        selected_model="",
    )
    monkeypatch.setattr(cc, "coding_selection", lambda **_: ["coding"])
    monkeypatch.setattr(config_mod, "load_config", lambda: config)

    with (
        patch("tui_gateway.server._load_cfg", return_value={}),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._resolve_startup_runtime", return_value=("test-model", None)),
        patch("tui_gateway.server._resolve_runtime_with_fallback", return_value=runtime),
        patch("tui_gateway.server._load_provider_routing", return_value={}),
        patch("tui_gateway.server._load_fallback_model", return_value=[]),
        patch("run_agent.AIAgent") as agent_cls,
    ):
        server._make_agent("empty-tools", "empty-tools", platform_override="desktop")

    assert agent_cls.call_args.kwargs["enabled_toolsets"] == []


def test_zero_tool_reports_remain_empty(monkeypatch):
    """The live session and the request-level tool inspector agree on zero."""
    agent = SimpleNamespace(enabled_toolsets=[], tools=[], model="test-model")
    session = {"agent": agent, "session_key": "empty-tools", "source": "desktop"}
    monkeypatch.setitem(server._sessions, "empty-tools", session)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})

    try:
        response = server._methods["tools.show"](
            "empty-tools", {"session_id": "empty-tools"}
        )
        toolsets = server._methods["toolsets.list"](
            "empty-toolsets", {"session_id": "empty-tools"}
        )
        info = server._session_info(agent, session)
    finally:
        server._sessions.pop("empty-tools", None)

    assert response["result"] == {"sections": [], "total": 0}
    assert not any(item["enabled"] for item in toolsets["result"]["toolsets"])
    assert info["tools"] == {}


def test_background_preview_and_mcp_reload_preserve_explicit_empty_toolsets(
    monkeypatch,
):
    """No secondary TUI agent may turn an explicit deny-all back into tools."""
    import agent.coding_context as cc
    import hermes_cli.config as config_mod
    import tools.mcp_tool as mcp_tool

    config = {"platform_toolsets": {"cli": []}}
    parent = SimpleNamespace(enabled_toolsets=[], model="test-model", tools=[])
    monkeypatch.setattr(cc, "coding_selection", lambda **_: ["coding"])
    monkeypatch.setattr(config_mod, "load_config", lambda: config)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_get_db", MagicMock())
    monkeypatch.setattr(server, "_load_reasoning_config", lambda *_: None)
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_agent_fallback_model", lambda _: [])
    monkeypatch.setattr(server, "_mcp_reload_gen", 0)
    monkeypatch.setattr(server, "_mcp_reload_loaded_rev", "")

    assert server._background_agent_kwargs(parent, "background")["enabled_toolsets"] == []
    assert server._ephemeral_preview_agent_kwargs(parent, "preview")["enabled_toolsets"] == []

    seen: dict[str, object] = {}
    monkeypatch.setitem(
        server._sessions,
        "empty-tools",
        {"agent": parent, "session_key": "empty-tools", "source": "desktop"},
    )
    monkeypatch.setattr(mcp_tool, "shutdown_mcp_servers", lambda: None)
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", lambda: None)
    monkeypatch.setattr(
        mcp_tool,
        "refresh_agent_mcp_tools",
        lambda _agent, **kwargs: seen.update(kwargs),
    )
    monkeypatch.setattr(server, "_compute_mcp_rev", lambda: "empty-tools")
    monkeypatch.setattr(server, "_session_info", lambda *_: {})
    monkeypatch.setattr(server, "_emit", lambda *_: None)

    try:
        response = server._methods["reload.mcp"](
            "reload-empty", {"session_id": "empty-tools", "confirm": True}
        )
    finally:
        server._sessions.pop("empty-tools", None)

    assert response["result"]["status"] == "reloaded"
    assert seen["enabled_override"] == []
