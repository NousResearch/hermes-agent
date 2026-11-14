"""Off-turn transcript writers on an idle-released Desktop record (#146735) refuse a chat another surface owns.

The model-switch marker (``config.set model`` / ``/model``) and a manual ``/compress`` used to write into the
new owner's transcript: only submit and rewind re-claimed the lease. Real SQLite store, real registry, real
reaper tick and real JSON-RPC handlers; only the model resolver and the summarizer are stubbed.
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
    _closed = False  # the Desktop app is still connected

    def write(self, frame):
        return True


class _Agent:
    def __init__(self, sid: str):
        self.session_id, self.model, self.provider, self.base_url, self.api_key = sid, "old/model", "prov", "", ""
        self.reasoning_config = None
        self.switched = []

    def clear_interrupt(self):
        return None

    def switch_model(self, **kw):
        self.switched.append(kw["new_model"])
        self.model = kw["new_model"]


@pytest.fixture
def taken_chat(monkeypatch):
    """An idle Desktop record the real reaper released, then Telegram claimed and extended."""
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    db = SessionDB(db_path=get_hermes_home() / "state.db")
    monkeypatch.setattr(server, "_db", db)
    sid = "taken-chat"
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
    server._reap_idle_sessions()  # the real idle tick releases the lease
    lease, refusal = try_acquire_active_session(session_id=sid, surface="telegram", config={})
    assert lease is not None, refusal
    db.append_message(sid, "user", "telegram message")
    db.append_message(sid, "assistant", "telegram answer")
    try:
        yield SimpleNamespace(sid=sid, record=record, db=db)
    finally:
        lease.release()
        server._release_active_session_slot(record)
        server._sessions.pop(sid, None)
        db.close()


_OWNER_TRANSCRIPT = [
    "question 0", "answer 0", "question 1", "answer 1", "question 2", "answer 2", "telegram message", "telegram answer"]


def _outcome(response: dict, chat) -> dict:
    error = response.get("error") or {}
    return {"code": error.get("code"), "reason": (error.get("data") or {}).get("reason"),
            "transcript": [m["content"] for m in chat.db.get_messages_as_conversation(chat.sid)]}


def test_model_switch_never_writes_its_marker_into_the_new_owners_transcript(taken_chat, monkeypatch):
    from hermes_cli.model_switch import ModelSwitchResult

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: ModelSwitchResult(
        success=True, new_model="other/model", target_provider="prov"))
    monkeypatch.setattr(server, "_expensive_model_confirm", lambda *a, **k: None)
    monkeypatch.setattr(server, "_merge_preflight_warning", lambda *a, **k: None)

    response = server.handle_request({"id": "r", "method": "config.set", "params": {
        "session_id": taken_chat.sid, "key": "model", "value": "other/model --provider prov --session"}})

    agent = taken_chat.record["agent"]
    assert {**_outcome(response, taken_chat), "switched": agent.switched, "pinned": "model_override" in taken_chat.record} == {
        "code": 4090, "reason": "SESSION_NOT_OWNED", "transcript": _OWNER_TRANSCRIPT, "switched": [], "pinned": False}


def test_manual_compress_never_starts_on_the_new_owners_transcript(taken_chat, monkeypatch):
    summarized = []

    def summarizer(agent, history, request, **kw):  # the seam: a real one would archive and rewrite rows
        summarized.append(len(history))
        raise AssertionError("compression ran on a chat another surface owns")

    monkeypatch.setattr("agent.conversation_compression_manual.compress_now", summarizer)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: False)

    response = server.handle_request({"id": "r", "method": "session.compress", "params": {"session_id": taken_chat.sid}})

    assert {**_outcome(response, taken_chat), "summarized": summarized, "running": taken_chat.record["running"]} == {
        "code": 4090, "reason": "SESSION_NOT_OWNED", "transcript": _OWNER_TRANSCRIPT, "summarized": [], "running": False}
