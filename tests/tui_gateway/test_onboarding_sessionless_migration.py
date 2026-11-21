"""Sessionless setup retires legacy chat records without rewriting chat history."""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path

import pytest

from agent import onboarding_record
from session_rules.onboarding_flow import End, Facts, new_record
from agent.onboarding_session import onboarding_chat_toolsets
from hermes_cli.onboarding_run import set_run
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
from persistence.conversation.sessions import NewSession
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
    # Each test is a fresh backend process: the migration latch starts open.
    monkeypatch.setattr(server, "_sessionless_setup_migrated", False)
    db = server._get_db()
    db.sessions.create(NewSession(session_id="onb", source="desktop"))
    db.sessions.patch_model_config("onb", {
        onboarding_record.KEY: new_record(Facts(os="macos"), run_id="legacy"),
    })
    set_run(True)
    return db


def _call(method: str, params: dict | None = None) -> dict:
    response = server.handle_request({"jsonrpc": "2.0", "id": 3, "method": method, "params": {"sessionless": True, **(params or {})}})
    assert "result" in response, response
    return response["result"]


def _history(db, session_ids) -> dict:
    return {sid: db.messages.read_transcript(sid, TranscriptRead(view=TranscriptView.ALL)) for sid in session_ids}


def test_state_retires_legacy_records_without_rewriting_compressed_history(db, monkeypatch):
    events = []
    monkeypatch.setattr(server, "_broadcast_global_event", lambda kind, payload: events.append((kind, payload)))
    config = Path(server._hermes_home) / "config.yaml"
    config_before = config.read_bytes()
    db.messages.append("onb", NewMessage(role="user", content="Keep my conversation"))
    db.sessions.patch_model_config("onb", {"codex_thread_id": "segment-only"})
    db.lineage.publish_compression_child("onb", CompressionChild(session_id="child", source="desktop"),
                                         [NewMessage(role="user", content="summary")],
                                         require_compression_lease=False)
    assert onboarding_record.load(db, "child", persistence=db) == onboarding_record.load(db, "onb", persistence=db)
    assert db.sessions.get_model_config_value("child", "codex_thread_id") is None
    history = _history(db, ("onb", "child"))
    assert all(history.values())
    toolsets = onboarding_chat_toolsets(["web"], db, "child", persistence=db)

    classic = server.handle_request({"jsonrpc": "2.0", "id": 4, "method": "onboarding.state", "params": {}})
    assert "result" in classic, classic
    assert onboarding_record.active_session_id(db, persistence=db) == "child"
    assert _history(db, history) == history

    assert _call("onboarding.state")["run"] is True
    assert onboarding_record.active_session_id(db, persistence=db) is None
    assert _history(db, history) == history
    assert all(db.sessions.get(sid) is not None for sid in history)
    ended = [("onboarding.ended", {"run_id": "legacy", "stored_session_id": "child", "outcome": "replaced"})]
    assert events == ended
    assert onboarding_chat_toolsets(["web"], db, "child", persistence=db) == toolsets
    assert db.sessions.get_model_config_value("onb", "codex_thread_id") == "segment-only"
    assert config.read_bytes() == config_before

    # Repeated reads and a late legacy agent completion cannot finish the new guide.
    onboarding_record.advance(db, "child", End(), persistence=db)
    assert _call("onboarding.state")["run"] is True
    assert _history(db, history) == history
    assert events == ended
    assert config.read_bytes() == config_before


def test_retirement_marks_already_ended_records_without_reannouncing_or_closing_guide(db, monkeypatch):
    onboarding_record.advance(db, "onb", End("skipped"), persistence=db)
    set_run(True, sessionless=True, step="model", restart=True)
    config = Path(server._hermes_home) / "config.yaml"
    config_before = config.read_bytes()
    events = []
    monkeypatch.setattr(server, "_broadcast_global_event", lambda kind, payload: events.append((kind, payload)))

    for _ in range(2):
        assert _call("onboarding.state")["run"] is True
        record = onboarding_record.load(db, "onb", persistence=db)
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
        assert onboarding_record.active_session_id(db, persistence=db) is None
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
    assert onboarding_record.active_session_id(db, persistence=db) is None


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


def test_state_takes_the_store_writer_only_while_legacy_runs_remain(db, monkeypatch):
    writers = []
    real = server._profile_db

    @contextlib.contextmanager
    def recording(params=None, *, writer=False):
        writers.append(writer)
        with real(params, writer=writer) as handle:
            yield handle

    monkeypatch.setattr(server, "_profile_db", recording)
    for _ in range(3):
        assert _call("onboarding.state")["run"] is True
    assert writers.count(True) == 1
    assert onboarding_record.active_session_id(db, persistence=db) is None

    # A new backend process finds the legacy run already retired and stays a reader.
    monkeypatch.setattr(server, "_sessionless_setup_migrated", False)
    writers.clear()
    assert _call("onboarding.state")["run"] is True
    assert True not in writers
