"""Session-scoped /model pin must survive an agent rebuild (#127449).

A composer /model pick lives in ``session["model_override"]``. Any rebuild
(``_sync_bot_capabilities`` roster change, compression rotation, deferred /
failed build restart) funnels through ``_rebuild_session_agent`` — which must
forward the live pin to ``_make_agent`` instead of letting the replacement
resolve its model from config. ``/new`` (``_reset_session_agent``) pops the
pin first, so it stays a real conversation boundary.
"""

import types

import tui_gateway.server as server

PIN = {
    "model": "stealth/space-bunny-alpha",
    "provider": "openrouter",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "sk-test",
    "api_mode": "chat_completions",
}


def _stub_build(monkeypatch):
    seen = {}

    def fake_make_agent(sid, key, **kwargs):
        seen.update(kwargs)
        agent = types.SimpleNamespace(model="cfg/model", provider="nous")
        agent._session_db = None
        agent._owns_session_db = False
        return agent

    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_bind_build_profile_scopes", lambda _home: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("cfg/model", "nous"))
    monkeypatch.setattr(server, "_transfer_db_to_agent", lambda _a, _db: False)
    return seen


def _live_session(**extra):
    old = types.SimpleNamespace(model="stealth/space-bunny-alpha", provider="openrouter")
    old._session_db = None
    old._owns_session_db = False
    session = {"agent": old, "session_key": "session-key", "profile_home": None}
    session.update(extra)
    return session


def test_rebuild_forwards_live_model_pin(monkeypatch):
    seen = _stub_build(monkeypatch)
    session = _live_session(model_override=dict(PIN))

    server._rebuild_session_agent("sid", session, session_id="session-key")

    assert seen.get("model_override") == PIN


def test_rebuild_explicit_kwarg_wins_over_live_pin(monkeypatch):
    seen = _stub_build(monkeypatch)
    session = _live_session(model_override=dict(PIN))
    explicit = {"model": "other/model"}

    server._rebuild_session_agent(
        "sid", session, session_id="session-key", model_override=explicit
    )

    assert seen.get("model_override") == explicit


def test_rebuild_without_pin_sends_no_override(monkeypatch):
    # /new pops the pin before rebuilding: nothing to forward.
    seen = _stub_build(monkeypatch)
    session = _live_session()

    server._rebuild_session_agent("sid", session, session_id="session-key")

    assert "model_override" not in seen
