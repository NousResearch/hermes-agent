"""A Desktop/TUI model switch must survive a rebuild even after the messaging gateway synced a route.

The gateway's after-turn sync writes the route it ran only into the nested ``model_config.gateway_runtime``,
and ``session_gateway_runtime`` reads that shape FIRST. The TUI switch persisted only the top-level keys, so
a session first used from Discord on the profile default (opencode-go) and then switched in Desktop to
grok-4.7 on xai-oauth resumed after a websocket reattach as grok-4.7 on opencode-go: the turn succeeded and
was billed to the wrong provider (#143157). A committed switch now pins both shapes, like
``update_session_model`` (#79536); the later writer wins in either order.
"""

import types

import pytest

from gateway.run_turn_prepare import GatewayTurnPrepareMixin
from hermes_state import SessionDB
from tui_gateway import server

OPENCODE_GO = {"provider": "opencode-go", "base_url": "https://opencode.ai/zen/go/v1", "api_mode": "chat_completions"}
XAI_OAUTH = {"provider": "xai-oauth", "base_url": "https://api.x.ai/v1", "api_mode": "codex_responses"}


def _agent(db, model, route):
    agent = types.SimpleNamespace(model=model, reasoning_config=None, service_tier=None, _session_db=db, **route)

    def switch_model(*, new_model, new_provider, base_url, api_mode, **_kw):
        agent.model, agent.provider, agent.base_url, agent.api_mode = new_model, new_provider, base_url, api_mode

    agent.switch_model = switch_model
    return agent


def _gateway_turn(db, session_id, agent):
    """The real after-turn sync a messaging-gateway turn runs."""
    runner = types.SimpleNamespace(_session_db=types.SimpleNamespace(_db=db))
    GatewayTurnPrepareMixin._sync_session_model_from_agent(runner, session_id, agent)


def _desktop_switch(monkeypatch, db, session_id, model, route, *, snapshot=None):
    """The real TUI switch commit, with the UI side effects stubbed out."""
    for name in ("_restart_slash_worker", "_persist_live_session_system_prompt", "_append_model_switch_marker",
                 "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    session = {"agent": _agent(db, "deepseek-v4.1-flash", OPENCODE_GO), "session_key": session_id}
    result = types.SimpleNamespace(new_model=model, target_provider=route["provider"], api_key=None,
                                   base_url=route["base_url"], api_mode=route["api_mode"])
    server._commit_agent_switch("sid", session, session["agent"], result, "deepseek-v4.1-flash", snapshot)


def _resumed_route(db, session_id):
    override = server._stored_session_runtime_overrides(db.get_session(session_id))["model_override"]
    return override["model"], override["provider"]


@pytest.fixture
def db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="discord", model="deepseek-v4.1-flash")
    return db


def test_desktop_switch_after_gateway_sync_resumes_on_picked_provider(monkeypatch, db):
    _gateway_turn(db, "s1", _agent(db, "deepseek-v4.1-flash", OPENCODE_GO))
    _desktop_switch(monkeypatch, db, "s1", "grok-4.7", XAI_OAUTH)

    assert _resumed_route(db, "s1") == ("grok-4.7", "xai-oauth")


def test_gateway_sync_after_desktop_switch_still_wins(monkeypatch, db):
    _desktop_switch(monkeypatch, db, "s1", "grok-4.7", XAI_OAUTH)
    _gateway_turn(db, "s1", _agent(db, "deepseek-v4.1-flash", OPENCODE_GO))

    assert _resumed_route(db, "s1") == ("deepseek-v4.1-flash", "opencode-go")


def test_one_turn_switch_does_not_pin_route(monkeypatch, db):
    _gateway_turn(db, "s1", _agent(db, "deepseek-v4.1-flash", OPENCODE_GO))
    _desktop_switch(monkeypatch, db, "s1", "grok-4.7", XAI_OAUTH, snapshot={"model": "deepseek-v4.1-flash"})

    assert SessionDB.session_gateway_runtime(db.get_session("s1"))["provider"] == "opencode-go"
