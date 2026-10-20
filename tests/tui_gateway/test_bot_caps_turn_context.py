"""A Bot Chat capability rebuild runs inside the live turn's session context and must not
wipe it: set/clear_session_vars is non-nestable, so the rebuild's clear used to leave the
rest of the turn (start_chat, secret prompts) with an empty HERMES_UI_SESSION_ID (#136877)."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("rebuild_raises", [False, True])
def test_capability_rebuild_restores_turn_session_context(monkeypatch, rebuild_raises):
    from gateway import session_context
    from tui_gateway import server

    def make_agent(*_args, **_kwargs):
        if rebuild_raises:
            raise RuntimeError("rebuild failed")
        return SimpleNamespace(_session_db=None, _owns_session_db=False)

    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_config_model_target", lambda: "default-a")
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "tools.bot_mode_probe.capability_fingerprint", lambda _home: "after"
    )
    session = {
        "agent": SimpleNamespace(_session_title_hint="Bot Chat"),
        "session_key": "durable-key",
        "bot_caps_seen": "before",
        "cwd": "/tmp",
        "source": "desktop",
    }
    monkeypatch.setitem(server._sessions, "sid-1", session)
    tokens = session_context.set_session_vars(
        source="desktop",
        session_key="durable-key",
        session_id="durable-key",
        ui_session_id="sid-1",
    )
    try:
        server._sync_bot_capabilities("sid-1", session)
        assert session_context.get_session_env("HERMES_UI_SESSION_ID") == "sid-1"
        assert session_context.get_session_env("HERMES_SESSION_KEY") == "durable-key"
        assert session_context.get_session_env("HERMES_SESSION_SOURCE") == "desktop"
    finally:
        session_context.clear_session_vars(tokens)
