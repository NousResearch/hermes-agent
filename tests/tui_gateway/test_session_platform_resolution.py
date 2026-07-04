"""Platform/source tagging for the desktop chat surface.

The desktop app's chat panel uses ``hermes serve`` (the ``tui_gateway``
backend), so every chat session historically got ``platform="tui"`` stamped
on it — even though the user is in a graphical chat surface, not a
terminal. That mis-tag is why the agent suggested TUI-only slash commands
(like ``/reload-mcp``) to desktop chat users.

These tests pin the env-var matrix that resolves the session platform at
``tui_gateway`` session-creation time:

  HERMES_DESKTOP=1, HERMES_DESKTOP_TERMINAL unset  -> platform="desktop"
  HERMES_DESKTOP=1, HERMES_DESKTOP_TERMINAL=1     -> platform="tui"  (embedded pane)
  neither set                                      -> platform="tui"  (standalone)

The resolver helper is import-safe (no heavy module side effects) so it
can be unit-tested without spinning up the full gateway.
"""

import importlib

import pytest


def _reload_resolver():
    import tui_gateway.server as _srv
    importlib.reload(_srv)
    return _srv


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    return monkeypatch


class TestResolveSessionPlatform:
    def test_standalone_tui_neither_env_set(self, clean_env):
        _srv = _reload_resolver()
        assert _srv._resolve_session_platform() == "tui"

    def test_desktop_chat_backend_gets_desktop_tag(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP", "1")
        _srv = _reload_resolver()
        assert _srv._resolve_session_platform() == "desktop"

    def test_desktop_embedded_terminal_pane_stays_tui(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP", "1")
        clean_env.setenv("HERMES_DESKTOP_TERMINAL", "1")
        _srv = _reload_resolver()
        assert _srv._resolve_session_platform() == "tui"

    def test_desktop_terminal_alone_means_standalone_tui(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP_TERMINAL", "1")
        _srv = _reload_resolver()
        assert _srv._resolve_session_platform() == "tui"

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "Yes", "ON"])
    def test_truthy_variants_recognized(self, clean_env, val):
        clean_env.setenv("HERMES_DESKTOP", val)
        _srv = _reload_resolver()
        assert _srv._resolve_session_platform() == "desktop"

    @pytest.mark.parametrize("val", ["0", "false", "", "no", "off", "False"])
    def test_falsy_variants_fall_back_to_tui(self, clean_env, val):
        clean_env.setenv("HERMES_DESKTOP", val)
        _srv = _reload_resolver()
        assert _srv._resolve_session_platform() == "tui"

    def test_embedded_terminal_overrides_desktop_when_both_set(self, clean_env):
        """The terminal-pane qualifier must short-circuit the desktop-backend
        marker. An embedded TUI is a TUI, not a desktop chat surface."""
        clean_env.setenv("HERMES_DESKTOP", "1")
        clean_env.setenv("HERMES_DESKTOP_TERMINAL", "true")
        _srv = _reload_resolver()
        assert _srv._resolve_session_platform() == "tui"


class TestResolveSessionSource:
    def test_explicit_source_param_wins(self, clean_env):
        _srv = _reload_resolver()
        assert _srv._resolve_session_source("telegram") == "telegram"

    def test_explicit_empty_source_falls_back_to_env(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP", "1")
        _srv = _reload_resolver()
        assert _srv._resolve_session_source("") == "desktop"

    def test_explicit_none_source_falls_back_to_env(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP", "1")
        _srv = _reload_resolver()
        assert _srv._resolve_session_source(None) == "desktop"

    def test_no_env_no_param_defaults_to_tui(self, clean_env):
        _srv = _reload_resolver()
        assert _srv._resolve_session_source(None) == "tui"

    def test_embedded_terminal_default_is_tui(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP", "1")
        clean_env.setenv("HERMES_DESKTOP_TERMINAL", "1")
        _srv = _reload_resolver()
        assert _srv._resolve_session_source(None) == "tui"

    def test_explicit_source_param_resists_env_drift(self, clean_env):
        """A caller that explicitly passes source="cli" must not be silently
        rewritten to "desktop" by env vars — the resolver only fills in the
        default when one is missing."""
        clean_env.setenv("HERMES_DESKTOP", "1")
        _srv = _reload_resolver()
        assert _srv._resolve_session_source("cli") == "cli"


class TestResolveAgentPlatform:
    def test_explicit_desktop_source_drives_agent_platform_without_env(self, clean_env):
        _srv = _reload_resolver()
        assert _srv._resolve_agent_platform("desktop") == "desktop"

    def test_missing_source_falls_back_to_env_resolved_platform(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP", "1")
        _srv = _reload_resolver()
        assert _srv._resolve_agent_platform(None) == "desktop"

    def test_explicit_tui_source_keeps_embedded_terminal_as_tui(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP", "1")
        _srv = _reload_resolver()
        assert _srv._resolve_agent_platform("tui") == "tui"


class TestSessionSourceFallback:
    def test_session_source_uses_existing_session_value(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP", "1")
        _srv = _reload_resolver()
        assert _srv._session_source({"source": "telegram"}) == "telegram"

    def test_session_source_defaults_to_desktop_under_desktop_backend(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP", "1")
        _srv = _reload_resolver()
        assert _srv._session_source({}) == "desktop"
        assert _srv._session_source(None) == "desktop"

    def test_session_source_defaults_to_tui_for_embedded_terminal(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP", "1")
        clean_env.setenv("HERMES_DESKTOP_TERMINAL", "1")
        _srv = _reload_resolver()
        assert _srv._session_source({}) == "tui"
        assert _srv._session_source(None) == "tui"
