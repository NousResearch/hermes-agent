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

import pytest

from tui_gateway import server
from toolsets import TOOLSETS

GUI_TOOLS = {
    "annotate_preview",
    "desktop_preview",
    "drive_preview",
    "close_terminal",
    "focus_pane",
    "read_terminal",
    "read_window_below",
    "react_to_message",
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
            "catalog",
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

    def test_config_path_resolves_the_session_platform(self, no_desktop_env):
        """A toolset granted under platform_toolsets.<session platform> but not
        under cli must reach that session's agent (#89547): the resolver
        hardcoded "cli" at the resolution call, silently filtering every
        non-cli grant from the per-turn tool definitions."""
        import agent.coding_context as cc
        import hermes_cli.config as config_mod

        no_desktop_env.setattr(cc, "coding_selection", lambda **_: None)
        no_desktop_env.setattr(
            config_mod,
            "load_config",
            lambda: {
                "platform_toolsets": {
                    "cli": ["terminal"],
                    "desktop": ["terminal", "memory"],
                }
            },
        )

        desktop = server._load_enabled_toolsets("desktop")
        tui = server._load_enabled_toolsets("tui")

        # desktop resolves against platform_toolsets.desktop: memory rides.
        assert desktop is not None and "memory" in desktop
        # tui has no platform_toolsets.tui row and no registry entry, so it
        # falls back to the cli row (review follow-up on #89550): the
        # unknown-platform composite hermes-tui resolves to nothing.
        assert tui is not None and "memory" not in tui
        assert "terminal" in tui
        assert "desktop_ui" not in tui

    def test_bare_tui_session_still_receives_core_toolsets(self, no_desktop_env):
        """Review follow-up on #89550: with default config (no
        platform_toolsets rows at all), resolving for "tui" derives the
        unknown-platform composite hermes-tui → zero toolsets. The resolver
        must fall back to the cli row so a bare `hermes --tui` / desktop
        session keeps terminal/file/web instead of going empty."""
        import agent.coding_context as cc
        import hermes_cli.config as config_mod

        no_desktop_env.setattr(cc, "coding_selection", lambda **_: None)
        no_desktop_env.setattr(config_mod, "load_config", dict)

        tui = server._load_enabled_toolsets("tui")
        desktop = server._load_enabled_toolsets("desktop")

        assert tui is not None and desktop is not None
        for surface in (tui, desktop):
            assert "terminal" in surface
            assert "web" in surface

    def test_explicit_env_pin_still_wins(self, no_desktop_env):
        """HERMES_TUI_TOOLSETS is an operator override; surface can't re-add."""
        no_desktop_env.setenv("HERMES_TUI_TOOLSETS", "web,memory")

        assert server._load_enabled_toolsets("desktop") == ["web", "memory"]


class TestExplicitEmptySelection:
    """#82010: an explicitly saved empty platform_toolsets list is a zero-tool state.

    ``platform_toolsets.cli: []`` used to fall open — the resolver returned an empty set and
    the gateway converted it to None, which downstream means 'no restriction' (every
    toolset). Only an ABSENT key means that; the explicit empty list must return [].
    """

    @staticmethod
    def _config(no_desktop_env, platform_toolsets):
        import agent.coding_context as cc
        import hermes_cli.config as config_mod

        no_desktop_env.setattr(cc, "coding_selection", lambda **_: None)
        no_desktop_env.setattr(
            config_mod, "load_config", lambda: {"platform_toolsets": platform_toolsets}
        )

    def test_explicit_empty_list_yields_no_toolsets(self, no_desktop_env):
        self._config(no_desktop_env, {"cli": []})

        assert server._load_enabled_toolsets("desktop") == []
        assert server._load_enabled_toolsets("tui") == []

    def test_explicit_empty_list_on_custom_platform_yields_no_toolsets(self, no_desktop_env):
        """#89550 rebase follow-up: the fail-closed check must use the SAME
        resolved key as the resolution. With ``platform_toolsets.support: []``
        and no cli row, keying the check to "cli" fell open — the session got
        the unrestricted default instead of its explicit zero-tool state."""
        self._config(no_desktop_env, {"support": []})

        assert server._load_enabled_toolsets("support") == []

    def test_absent_key_keeps_the_default_selection(self, no_desktop_env):
        self._config(no_desktop_env, {})

        result = server._load_enabled_toolsets("tui")
        assert result  # the platform default applies — not the zero state, not None-as-all

    def test_named_selection_is_honored(self, no_desktop_env):
        self._config(no_desktop_env, {"cli": ["memory"]})

        result = server._load_enabled_toolsets("tui")
        assert result is not None
        assert "memory" in result
        assert "terminal" not in result
