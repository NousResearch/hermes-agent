"""Edges of the idle-released Desktop lease: every refusal answers 4090, changes nothing, and never clears
another turn's ``running`` or leaks a transcript claim.

Real SQLite SessionDB, real active-session registry, real reaper and real JSON-RPC handlers. Interleavings
are pinned at production seams (the refusal ``error`` frame, the supervisor's late-handler registration).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from hermes_cli.active_sessions import try_acquire_active_session
from hermes_state import SessionDB
from tui_gateway import server


class _LiveSocket:
    _closed = False

    def write(self, frame):
        return True


class _Agent:
    def __init__(self, sid):
        self.session_id, self.model, self.provider, self.base_url, self.api_key = sid, "old/model", "prov", "", ""
        self.reasoning_config = None
        self.switched = []

    def clear_interrupt(self):
        return None

    def switch_model(self, **kw):
        self.switched.append(kw["new_model"])
        self.model = kw["new_model"]


def _make(monkeypatch, sid):
    from hermes_constants import get_hermes_home
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    db = SessionDB(db_path=get_hermes_home() / "state.db")
    monkeypatch.setattr(server, "_db", db)
    db.create_session(sid, source="desktop")
    for n in range(3):
        db.append_message(sid, "user", f"question {n}")
        db.append_message(sid, "assistant", f"answer {n}")
    ready = threading.Event()
    ready.set()
    long_ago = time.time() - 3600
    record = {
        "session_key": sid, "source": "desktop", "history_version": 0, "history_lock": threading.Lock(),
        "history": db.get_messages_as_conversation(sid, include_row_ids=True), "agent_ready": ready,
        "running": False, "created_at": long_ago, "last_active": long_ago, "transport": _LiveSocket(),
        "agent": _Agent(sid), "attached_images": [], "cols": 80}
    server._sessions[sid] = record
    assert server._ensure_active_session_slot(sid, record) is None
    return db, record


@pytest.fixture
def chat(monkeypatch):
    """Idle Desktop record still holding its lease (nobody else around)."""
    sid = "idle-edge-chat"
    db, record = _make(monkeypatch, sid)
    leases = []

    def compete():
        lease, _ = try_acquire_active_session(session_id=sid, surface="telegram", config={})
        if lease is not None:
            leases.append(lease)
        return lease
    try:
        yield SimpleNamespace(sid=sid, record=record, db=db, compete=compete)
    finally:
        for lease in leases:
            lease.release()
        server._release_active_session_slot(record)
        server._sessions.pop(sid, None)
        db.close()


@pytest.fixture
def taken(chat):
    """The real reaper released the idle lease; Telegram claimed the chat and extended it."""
    server._reap_idle_sessions()
    assert chat.record.get("active_session_lease") is None
    lease = chat.compete()
    assert lease is not None
    chat.db.append_message(chat.sid, "user", "telegram message")
    chat.db.append_message(chat.sid, "assistant", "telegram answer")
    chat.lease = lease
    return chat


def _transcript(chat):
    return [m["content"] for m in chat.db.get_messages_as_conversation(chat.sid)]


# ── SessionNotOwned on the compute-host paths answers 4090 (Greptile 4569456266)
def _compute_host(monkeypatch):
    sent = []
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: True)
    monkeypatch.setattr(server, "_send_compute_host_control", lambda sid, **kw: sent.append(kw) or {"type": "control.ack"})
    return sent


def test_g1_compute_host_session_compress_answers_4090(taken, monkeypatch):
    sent = _compute_host(monkeypatch)
    resp = server.handle_request({"id": "r", "method": "session.compress", "params": {"session_id": taken.sid}})
    err = resp.get("error") or {}
    assert {"code": err.get("code"), "reason": (err.get("data") or {}).get("reason"), "sent": sent} == {
        "code": 4090, "reason": "SESSION_NOT_OWNED", "sent": []}, err


def test_g2_compute_host_command_dispatch_compress_answers_4090(taken, monkeypatch):
    sent = _compute_host(monkeypatch)
    resp = server.handle_request({"id": "r", "method": "command.dispatch",
                                  "params": {"session_id": taken.sid, "name": "compress", "arg": ""}})
    err = resp.get("error") or {}
    assert {"code": err.get("code"), "sent": sent} == {"code": 4090, "sent": []}, resp


def test_g3_compute_host_slash_exec_model_answers_4090(taken, monkeypatch):
    sent = _compute_host(monkeypatch)

    class _Worker:
        def run(self, cmd):
            return "model switched (worker)"

        def pop_seed(self):
            return ""
    monkeypatch.setattr(server, "_slash_worker_for", lambda rid, sid, session: (_Worker(), None))
    resp = server.handle_request({"id": "r", "method": "slash.exec",
                                  "params": {"session_id": taken.sid, "command": "/model other/model"}})
    err = resp.get("error") or {}
    assert {"code": err.get("code"), "result": resp.get("result"), "sent": sent} == {
        "code": 4090, "result": None, "sent": []}, resp
