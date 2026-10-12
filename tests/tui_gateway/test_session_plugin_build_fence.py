"""Construction and native session commands reserve the same exact idle session."""

from __future__ import annotations

import threading

import pytest

from tests.tui_gateway.test_plugin_session_slash import (
    LiveAgent, _flag_call, _override, _result, _session,
)

pytest_plugins = ("tests.tui_gateway.test_plugin_session_slash",)


@pytest.fixture
def cold_build(gateway, monkeypatch):
    server, db, _home = gateway
    sid, key = "sid-build-fence", "build-fence"
    session = _session(server, db, sid, key, None)
    session["agent_ready"] = threading.Event()
    entered, release = threading.Event(), threading.Event()
    made = []

    def make_agent(_sid, session_key, **_kwargs):
        agent = LiveAgent(db, session_key)
        made.append(agent)
        entered.set()
        assert release.wait(5), "blocked construction was not released"
        return agent

    from tui_gateway import entry
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_bind_build_profile_scopes", lambda _home: None)
    monkeypatch.setattr(server, "_deferred_build_agent_kwargs", lambda *_a: {})
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_session_todo_state", lambda _session: None)
    monkeypatch.setattr(server, "_config_model_target", dict)
    monkeypatch.setattr(server, "_wire_session_agent", lambda *_a: False)
    monkeypatch.setattr(server, "_announce_built_agent", lambda *_a: None)
    try:
        yield server, db, sid, session, entered, release, made
    finally:
        release.set()
        if thread := session.get("_agent_build_thread"):
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize("method", ["slash.exec", "command.dispatch"])
@pytest.mark.parametrize("arg", ["on", "status"])
def test_build_admission_rejects_read_and_write_leases_until_ready(cold_build, method, arg):
    server, db, sid, session, entered, release, made = cold_build
    before = db.get_session(session["session_key"])
    server._start_agent_build(sid, session)
    assert entered.wait(2), "construction did not enter"
    assert session["agent_build_started"] and not session["agent_ready"].is_set()
    response = _flag_call(server, method, sid, arg)
    assert response["error"]["code"] == 4009
    assert session.get("_session_control_inflight") is None and session["agent"] is None
    assert db.get_session(session["session_key"]) == before
    release.set()
    assert session["agent_ready"].wait(2), "construction did not complete"
    assert session["agent"] is made[0]
    assert "Session flag" in _result(_flag_call(server, method, sid, arg))["output"]


def test_control_lease_prevents_build_and_attach_then_release_allows_build(cold_build):
    server, db, sid, session, entered, release, made = cold_build
    target, error = server._claim_live_plugin_session_control(sid, session)
    assert error is None and target is not None
    try:
        server._start_agent_build(sid, session)
        assert not session.get("agent_build_started")
        assert "_agent_build_thread" not in session and not made
        assert not server._attach_built_agent(sid, session, LiveAgent(db, session["session_key"]))
        assert session["agent"] is None and session.get("_session_control_inflight") is target.token
    finally:
        assert server._release_live_plugin_session_control(target)
    server._start_agent_build(sid, session)
    assert entered.wait(2), "released session did not start construction"
    release.set()
    assert session["agent_ready"].wait(2)
    assert session["agent"] is made[0] and session.get("_session_control_inflight") is None
    assert "saved: ON" in _result(_flag_call(server, "slash.exec", sid, "on"))["output"]
    assert _override(db.get_session(session["session_key"])) is True


def test_attach_rejects_replaced_registry_without_mutating_either_record(cold_build):
    server, db, sid, original, _entered, _release, _made = cold_build
    replacement = _session(server, db, sid, "replacement-build-fence", None)
    token = replacement["_session_control_inflight"] = object()
    assert not server._attach_built_agent(sid, original, LiveAgent(db, original["session_key"]))
    assert original["agent"] is None and replacement["agent"] is None
    assert replacement["_session_control_inflight"] is token


@pytest.mark.parametrize("path", ["start", "attach"])
def test_legacy_minimal_record_initializes_one_shared_history_lock(cold_build, path):
    server, db, sid, session, entered, release, made = cold_build
    session.pop("history_lock")
    if path == "start":
        server._start_agent_build(sid, session)
        assert entered.wait(2)
        shared_lock = session["history_lock"]
        release.set()
        assert session["agent_ready"].wait(2)
        assert session["agent"] is made[0] and session["history_lock"] is shared_lock
    else:
        agent = LiveAgent(db, session["session_key"])
        assert server._attach_built_agent(sid, session, agent)
        assert session["agent"] is agent and session["history_lock"] is not None
