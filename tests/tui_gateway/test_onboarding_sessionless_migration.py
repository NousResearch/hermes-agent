"""Sessionless setup retires legacy chat records without rewriting chat history."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from agent import onboarding_record
from agent.onboarding_flow import End, Facts, new_record
from agent.onboarding_session import onboarding_chat_toolsets
from hermes_cli.onboarding_run import set_run
from tests.waiting import wait_until
from tui_gateway import server, server_requests


@pytest.fixture
def db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_db", None)
    db = server._get_db()
    db.create_session("onb", "desktop")
    db.patch_session_model_config("onb", {
        onboarding_record.KEY: new_record(Facts(os="macos"), run_id="legacy"),
    })
    set_run(True)
    return db


def _call(method: str, params: dict | None = None) -> dict:
    response = server.handle_request({"jsonrpc": "2.0", "id": 3, "method": method, "params": {"sessionless": True, **(params or {})}})
    assert "result" in response, response
    return response["result"]


def test_state_retires_legacy_records_without_rewriting_compressed_history(db, monkeypatch):
    events = []
    monkeypatch.setattr(server, "_broadcast_global_event", lambda kind, payload: events.append((kind, payload)))
    config = Path(server._hermes_home) / "config.yaml"
    config_before = config.read_bytes()
    db.append_message("onb", "user", "Keep my conversation")
    db.patch_session_model_config("onb", {"codex_thread_id": "segment-only"})
    db.publish_compression_child(
        parent_session_id="onb", child_session_id="child", source="desktop",
        messages=[{"role": "user", "content": "summary"}], require_compression_lease=False,
    )
    assert onboarding_record.load(db, "child") == onboarding_record.load(db, "onb")
    assert db.get_session_model_config_value("child", "codex_thread_id") is None
    history = {sid: db.get_messages(sid) for sid in ("onb", "child")}
    toolsets = onboarding_chat_toolsets(["web"], db, "child")

    classic = server.handle_request({"jsonrpc": "2.0", "id": 4, "method": "onboarding.state", "params": {}})
    assert "result" in classic, classic
    assert onboarding_record.active_session_id(db) == "child"
    assert {sid: db.get_messages(sid) for sid in history} == history

    assert _call("onboarding.state")["run"] is True
    assert onboarding_record.active_session_id(db) is None
    assert {sid: db.get_messages(sid) for sid in history} == history
    assert all(db.get_session(sid) is not None for sid in history)
    ended = [("onboarding.ended", {"run_id": "legacy", "stored_session_id": "child", "outcome": "replaced"})]
    assert events == ended
    assert onboarding_chat_toolsets(["web"], db, "child") == toolsets
    assert db.get_session_model_config_value("onb", "codex_thread_id") == "segment-only"
    assert config.read_bytes() == config_before

    # Repeated reads and a late legacy agent completion cannot finish the new guide.
    onboarding_record.advance(db, "child", End())
    assert _call("onboarding.state")["run"] is True
    assert {sid: db.get_messages(sid) for sid in history} == history
    assert events == ended
    assert config.read_bytes() == config_before


def test_retirement_marks_already_ended_records_without_reannouncing_or_closing_guide(db, monkeypatch):
    onboarding_record.advance(db, "onb", End("skipped"))
    set_run(True, sessionless=True, step="model", restart=True)
    config = Path(server._hermes_home) / "config.yaml"
    config_before = config.read_bytes()
    events = []
    monkeypatch.setattr(server, "_broadcast_global_event", lambda kind, payload: events.append((kind, payload)))

    for _ in range(2):
        assert _call("onboarding.state")["run"] is True
        record = onboarding_record.load(db, "onb")
        assert record["sessionless"] is True
        assert record["ended_by"] == "skipped"
    assert events == []
    assert config.read_bytes() == config_before


def test_state_withdraws_a_waiting_legacy_card(db, monkeypatch):
    sid, frames, events = "ui-onb", [], []
    monkeypatch.setitem(server._sessions, sid, {"session_key": "onb", "source": "desktop"})
    monkeypatch.setattr(server_requests, "_answerable", lambda _sid: True)
    monkeypatch.setattr(server_requests, "_write", frames.append)
    monkeypatch.setattr(server_requests, "_emit", lambda kind, _sid, payload: events.append((kind, payload)))
    reply = []
    waiter = threading.Thread(target=lambda: reply.append(server._setup_beat_request(
        sid, {"beat": "theme", "card": {"picker": "accent"}})), daemon=True)
    waiter.start()
    try:
        wait_until(lambda: server_requests.pending_kind(sid) == "setup_beat")
        assert _call("onboarding.state")["run"] is True
        waiter.join(timeout=5)
        assert reply == [None]
        assert events == [("request.cancel", {"id": frames[0]["id"], "method": "setup_beat", "reason": "resolved"})]
        assert onboarding_record.active_session_id(db) is None
    finally:
        server._cancel_session_requests(sid, "resolved", method="setup_beat")
        waiter.join(timeout=5)


def test_state_withdraws_a_legacy_card_waiting_in_a_compute_host_child(db, monkeypatch):
    forwarded = []

    class Supervisor:
        def respond(self, sid, params, *, timeout=15.0):
            forwarded.append((sid, dict(params)))
            return {"type": "respond.ack", "response": {"result": {"cancelled": 1}}}

    sid = "ui-onb"
    monkeypatch.setitem(server._sessions, sid, {
        "session_key": "onb", "agent": None, "source": "desktop",
        "agent_ready": threading.Event(), "_compute_host_active": True, "history_lock": threading.Lock(),
    })
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: Supervisor())
    monkeypatch.setattr(server, "write_json", lambda _message: True)
    server._relay_compute_host_rpc({
        "jsonrpc": "2.0", "id": "srq-host", "method": "setup_beat",
        "params": {"session_id": sid, "beat": "theme", "card": {"picker": "accent"}},
    })

    assert _call("onboarding.state")["run"] is True
    assert forwarded == [(sid, {"cancel": {"method": "setup_beat", "reason": "resolved"}})]
    assert onboarding_record.active_session_id(db) is None


def test_initial_latch_keeps_peer_progress_and_completion(db):
    fresh = _call("onboarding.state")
    assert fresh["started"] is False
    started = _call("onboarding.set_run", {"run": True})
    assert started["started"] is True and started["step"] == "preferences"

    _call("onboarding.set_run", {"run": True, "step": "model"})
    # The window that saw `fresh` above latches after another window advances.
    resumed = _call("onboarding.set_run", {"run": True})
    assert resumed["run"] is True and resumed["step"] == "model"

    _call("onboarding.set_run", {"run": False, "mark_profile_offered": True})
    completed = _call("onboarding.set_run", {"run": True})
    assert completed["run"] is False and completed["step"] == "model"
    restarted = _call("onboarding.set_run", {"run": True, "step": "preferences", "restart": True})
    assert restarted["run"] is True and restarted["step"] == "preferences"
