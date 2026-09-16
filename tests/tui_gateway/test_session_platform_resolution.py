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

import pytest


def _reload_resolver():
    # Plain import — every resolver under test reads the env at CALL time, so
    # no reload is needed. importlib.reload(tui_gateway.server) would
    # re-register the module's atexit hooks (thread-pool shutdown +
    # _shutdown_sessions) on every test; duplicated hooks race the stderr
    # buffer at interpreter shutdown (Fatal Python error:
    # _enter_buffered_busy) — same flake class as PR #34217. Name kept for
    # the existing call sites.
    import tui_gateway.server as _srv
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


    def test_no_env_no_param_defaults_to_tui(self, clean_env):
        _srv = _reload_resolver()
        assert _srv._resolve_session_source(None) == "tui"


class TestResolveAgentPlatform:

    def test_missing_source_falls_back_to_env_resolved_platform(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP", "1")
        _srv = _reload_resolver()
        assert _srv._resolve_agent_platform(None) == "desktop"


class TestSessionSourceFallback:
    def test_session_source_uses_existing_session_value(self, clean_env):
        clean_env.setenv("HERMES_DESKTOP", "1")
        _srv = _reload_resolver()
        assert _srv._session_source({"source": "telegram"}) == "telegram"


class TestReattachAdoptsSource:
    """A live record keeps the ``source`` it was minted with, and nothing else rewrote it.

    That is correct while one client owns the session, but a reattach can arrive from a
    different surface — in practice an automatic reconnect after the WebSocket dropped, which
    is not a surface change the user made. The record then keeps a source no client is on, and
    because the next agent build reads it through ``_session_source`` -> ``platform_override``,
    the agent gets pinned to the wrong surface and is told it has capabilities (MEDIA:
    delivery, inline widgets) the live renderer does not have.
    """

    def _adopt(self):
        from tui_gateway.methods_session import _adopt_reattach_source
        return _adopt_reattach_source

    def test_reconnect_from_another_surface_refreshes_the_record(self):
        session = {"source": "tui"}
        self._adopt()(session, "desktop")
        assert session["source"] == "desktop"

    def test_client_without_a_source_keeps_the_record(self, clean_env):
        """Older desktops and the bot-room plumbing never send ``source``; they must keep the
        record they created rather than be reset to the env-resolved default."""
        session = {"source": "desktop"}
        for omitted in (None, "", "   "):
            self._adopt()(session, omitted)
            assert session["source"] == "desktop"

    def test_unchanged_source_is_left_alone(self):
        session = {"source": "desktop"}
        self._adopt()(session, "desktop")
        assert session["source"] == "desktop"

    def test_adopted_source_drives_the_next_agent_build(self, clean_env):
        """The end-to-end point: what a rebuild reads back must be the reattaching surface.

        Without the adopt, ``_session_source`` returns the stale value and the rebuilt agent is
        pinned to it — with no env var set, an omitted source resolves to "tui" (see
        ``_resolve_session_platform``), which is exactly how a desktop session came back as a
        terminal one after a dropped socket.
        """
        _srv = _reload_resolver()
        session = {"source": "tui"}
        self._adopt()(session, "desktop")
        assert _srv._session_source(session) == "desktop"


