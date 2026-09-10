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

import pytest

import tui_gateway.server as server
from toolsets import TOOLSETS, resolve_toolset

GUI_TOOLS = {
    "annotate_preview",
    "desktop_preview",
    "drive_preview",
    "close_terminal",
    "focus_pane",
    "read_terminal",
    "read_window_below",
    "react_to_message",
    "setup_mcp",
    "show_tip",
    "gui_tour",
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
        # apply_layout registers into desktop_ui via the registry (not the
        # static toolsets.py list), so force discovery first — otherwise the
        # result depends on which earlier test imported tool modules
        # (pre-existing ordering flake, surfaced by the #97979 test sweep).
        from tools.registry import discover_builtin_tools
        discover_builtin_tools()
        assert set(resolve_toolset("desktop_ui")) == GUI_TOOLS | {"apply_layout"}

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
        from unittest.mock import Mock, call

        import agent.coding_context as cc
        import hermes_cli.config as config_mod

        cfg = {"platform_toolsets": {"cli": ["file"]}}
        load = Mock(return_value=cfg)
        select = Mock(return_value=["coding"])
        no_desktop_env.setattr(config_mod, "load_config", load)
        no_desktop_env.setattr(cc, "coding_selection", select)

        assert server._load_enabled_toolsets("desktop") == [
            "coding",
            "desktop_ui",
            "project",
        ]
        assert server._load_enabled_toolsets("tui") == ["coding", "project"]
        assert load.call_count == 2
        assert select.call_args_list == [
            call(platform="desktop", config=cfg), call(platform="tui", config=cfg),
        ]

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

    def test_explicit_empty_config_beats_focus_and_gui_surfaces(self, no_desktop_env):
        import agent.coding_context as cc
        import hermes_cli.config as config_mod

        no_desktop_env.setattr(cc, "coding_selection", lambda **_: ["coding"])
        no_desktop_env.setattr(
            config_mod, "load_config", lambda: {"platform_toolsets": {"cli": []}}
        )

        assert server._load_enabled_toolsets("desktop") == []
        assert server._load_enabled_toolsets("tui") == []

    def test_explicit_env_pin_still_wins(self, no_desktop_env):
        """HERMES_TUI_TOOLSETS is an operator override; surface can't re-add."""
        no_desktop_env.setenv("HERMES_TUI_TOOLSETS", "web,memory")

        assert server._load_enabled_toolsets("desktop") == ["web", "memory"]


def test_zero_tool_inspectors_agree_with_the_live_agent(monkeypatch):
    agent = SimpleNamespace(enabled_toolsets=[], tools=[], model="test-model")
    session = {"agent": agent, "session_key": "empty-tools", "source": "desktop"}
    monkeypatch.setitem(server._sessions, "empty-tools", session)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})

    try:
        shown = server._methods["tools.show"](
            "show-empty", {"session_id": "empty-tools"}
        )
        listed = server._methods["tools.list"](
            "list-empty", {"session_id": "empty-tools"}
        )
        info = server._session_info(agent, session)
    finally:
        server._sessions.pop("empty-tools", None)

    assert shown["result"] == {"sections": [], "total": 0}
    assert not any(row["enabled"] for row in listed["result"]["toolsets"])
    assert info["tools"] == {}


@pytest.mark.parametrize("selection", [[], (), set()])
def test_preview_agent_preserves_any_explicit_empty_selection(monkeypatch, selection):
    agent = SimpleNamespace(
        enabled_toolsets=selection, disabled_toolsets=["file"],
        model="test-model", provider="test-provider"
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)

    kwargs = server._ephemeral_preview_agent_kwargs(agent, "empty-preview")

    assert kwargs["enabled_toolsets"] == []
    assert kwargs["disabled_toolsets"] == ["file"]


@pytest.mark.parametrize("selection", [None, [], ["memory"]])
def test_background_agent_preserves_selection_and_disabled_policy(monkeypatch, selection):
    agent = SimpleNamespace(enabled_toolsets=selection, disabled_toolsets=["file"], model="test")
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda platform: ["terminal"])
    kwargs = server._background_agent_kwargs(agent, "background")
    assert kwargs["enabled_toolsets"] == (["terminal"] if selection is None else selection)
    assert kwargs["disabled_toolsets"] == ["file"]


@pytest.mark.parametrize("selection", [None, [], ["terminal", "file", "memory"]])
def test_toolset_rows_apply_composite_disabled_policy(monkeypatch, selection):
    agent = SimpleNamespace(enabled_toolsets=selection, disabled_toolsets=["debugging"])
    monkeypatch.setitem(server._sessions, "disabled-tools", {"agent": agent})
    for method in ("tools.list", "toolsets.list"):
        result = server._methods[method]("list", {"session_id": "disabled-tools"})
        rows = {row["name"]: row for row in result["result"]["toolsets"]}
        assert not rows["terminal"]["enabled"]
        assert not rows["file"]["enabled"]
        assert rows["memory"]["enabled"] is (selection != [])


def test_tools_show_forwards_disabled_policy(monkeypatch):
    import model_tools

    agent = SimpleNamespace(enabled_toolsets=["terminal"], disabled_toolsets=["debugging"])
    monkeypatch.setitem(server._sessions, "disabled-tools", {"agent": agent})
    expected = model_tools.get_tool_definitions(
        enabled_toolsets=agent.enabled_toolsets, disabled_toolsets=agent.disabled_toolsets,
        quiet_mode=True, skip_tool_search_assembly=True,
    )
    result = server._methods["tools.show"]("show", {"session_id": "disabled-tools"})
    assert result["result"]["total"] == len(expected) == 0


def test_real_empty_config_survives_gui_resolution(tmp_path, monkeypatch, no_desktop_env):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("platform_toolsets:\n  cli: []\n", encoding="utf-8")
    assert server._load_enabled_toolsets("desktop") == []
    assert server._load_enabled_toolsets("tui") == []
