"""A bot delivery holds the writer fence, not the user's next desktop message."""

import contextlib
import threading
import types

import pytest

from hermes_cli import active_sessions as registry
from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.contracts.prompt_voice import PromptSubmitResult

_START_SESSION_WORK = server._start_session_work

@pytest.fixture
def chat(monkeypatch, tmp_path):
    home = tmp_path / "profile"
    db = SessionDB(db_path=home / "state.db")
    db.create_session("chat", source="desktop", model="test-model")
    agent = types.SimpleNamespace(session_id="chat")
    ready = threading.Event()
    ready.set()
    session = {"agent": agent, "agent_ready": ready, "session_key": "chat", "source": "desktop",
               "history": [], "history_lock": threading.RLock(), "history_version": 0,
               "running": False, "transport": None, "attached_images": [], "cols": 80}
    events, workers, dispatched = [], [], []
    monkeypatch.setattr(server, "_sessions", {"sid": session})
    monkeypatch.setattr(server, "_session_db", lambda s: contextlib.nullcontext(db))
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda key, **kw: registry.try_acquire_active_session(
        session_id=key, surface=kw["surface"], config={}, registry_home=home,
        metadata={"live_session_id": kw["live_session_id"]}))
    monkeypatch.setattr(server, "current_transport", lambda: None)
    monkeypatch.setattr(server, "_start_session_work", lambda fn, **kw: workers.append(fn) or types.SimpleNamespace(is_alive=lambda: True))
    monkeypatch.setattr(server, "_emit", lambda kind, sid, data: events.append((kind, data)))
    monkeypatch.setattr(server, "_emit_session_info_for_session", lambda *a: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a: False)
    monkeypatch.setattr(server, "_wire_session_agent", lambda *a: None)

    def run(rid, sid, s, text, **kwargs):
        dispatched.append((text, list(s["history"]), kwargs))
        s["running"] = False
        server._drain_queued_prompt(rid, sid, s)

    monkeypatch.setattr(server, "_run_prompt_submit", run)
    def hold(surface="bot_delivery", key="chat"):
        lease, error = registry.try_acquire_active_session(
            session_id=key, surface=surface, config={}, registry_home=home,
            metadata={"live_session_id": "delivery"})
        assert lease is not None and error is None
        return lease

    yield types.SimpleNamespace(home=home, db=db, session=session, events=events, workers=workers,
                                dispatched=dispatched, hold=hold)
    server._release_active_session_slot(session)
    db.close()


def submit(text="next", **params):
    return server._methods["prompt.submit"]("r1", {"session_id": "sid", "text": text, **params})


def test_desktop_send_queues_without_writing_while_delivery_owns_chat(chat):
    lease = chat.hold()
    try:
        result = submit()
        assert result["result"] == {"status": "queued"}
        PromptSubmitResult.model_validate(result["result"])
        assert chat.dispatched == []
        assert len(chat.workers) == 1
        assert chat.session["queued_prompt"]["text"] == "next"
        assert chat.db.get_messages("chat") == []
        assert chat.session.get("active_session_lease") is None
        assert chat.events == [("notice", {"message": "This bot is finishing a task; your message will send next."})]
    finally:
        lease.release()


def test_delivery_release_drains_fifo_and_adopts_completed_history(chat):
    lease = chat.hold()
    submit("first")
    chat.session["attached_images"] = ["image.png"]
    submit("second", display_kind="hidden")
    submit("second")  # a separate occurrence is not a duplicate of the earlier send
    assert len(chat.workers) == 1
    chat.db.append_message("chat", "user", "teammate request", display_metadata={"test_sender": "peer"})
    chat.db.append_message("chat", "assistant", "completed task")
    chat.db.end_session("chat", "cli_exit")
    lease.release()
    chat.workers[0]()
    assert [turn[0] for turn in chat.dispatched] == ["first", "second", "second"]
    assert chat.dispatched[0][1][-1]["content"] == "completed task"
    assert chat.dispatched[0][1][0]["display_metadata"]["test_sender"] == "peer"
    assert chat.dispatched[1][2]["image_paths"] == ["image.png"]
    assert chat.dispatched[1][2]["display_kind"] == "hidden"
    assert not chat.session.get("_bot_delivery_wait")
    assert chat.db.get_session("chat")["ended_at"] is None
    assert not any(kind in ("error", "message.complete") for kind, _ in chat.events)


def test_wait_worker_retries_the_real_fence_before_dispatch(chat, monkeypatch):
    lease = chat.hold()
    submit()
    calls = []
    original = server._ensure_active_session_slot
    def acquire(*args):
        result = original(*args)
        calls.append(result)
        if len(calls) == 2:
            lease.release()
        return result
    monkeypatch.setattr(server, "_ensure_active_session_slot", acquire)
    monkeypatch.setattr(server, "_BOT_DELIVERY_WAIT_SECONDS", 0)
    chat.workers[0]()
    assert len(calls) == 3
    assert all(r.reason == registry.SESSION_NOT_OWNED and r.owner_surface == "bot_delivery" for r in calls[:2])
    assert calls[2] is None
    assert len(chat.dispatched) == 1


def test_cancelled_wait_does_not_dispatch_or_release_delivery_lease(chat):
    lease = chat.hold()
    submit()
    chat.session["_queued_prompt_generation"] = 1
    chat.session["queued_prompt"] = None
    chat.workers[0]()
    assert chat.dispatched == []
    assert registry.active_session_registry_snapshot(registry_home=chat.home)[0]["lease_id"] == lease.lease_id
    lease.release()


def test_human_cli_owner_and_non_desktop_submit_still_refuse(chat):
    lease = chat.hold("cli")
    try:
        result = submit()
        assert result["error"]["data"]["reason"] == registry.SESSION_NOT_OWNED
        assert chat.workers == []
    finally:
        lease.release()
    lease = chat.hold()
    chat.session["source"] = "tui"
    try:
        assert submit()["error"]["data"]["reason"] == registry.SESSION_NOT_OWNED
        assert chat.workers == []
    finally:
        lease.release()


def test_wait_follows_compression_tip_and_never_claims_parent(chat, monkeypatch):
    lease = chat.hold()
    submit()
    chat.db.create_session("tip", source="desktop", model="test-model")
    monkeypatch.setattr(chat.db, "resolve_resume_session_id", lambda key: "tip")
    lease.release()
    tip_lease = chat.hold(key="tip")
    calls = []
    original = server._ensure_active_session_slot
    def acquire(sid, session):
        calls.append(session["session_key"])
        result = original(sid, session)
        if result is not None:
            tip_lease.release()
        return result
    monkeypatch.setattr(server, "_ensure_active_session_slot", acquire)
    monkeypatch.setattr(server, "_BOT_DELIVERY_WAIT_SECONDS", 0)
    chat.workers[0]()
    assert calls == ["tip", "tip"]
    assert chat.session["agent"].session_id == "tip"
    assert len(chat.dispatched) == 1


def test_dead_delivery_pid_is_reclaimed_by_desktop(chat, monkeypatch):
    lease = chat.hold()
    path = registry._state_path(chat.home)
    entries = registry._read_entries(path)
    entries[0]["pid"] = 99999999
    registry._write_entries(path, entries)
    assert server._ensure_active_session_slot("sid", chat.session) is None
    assert chat.session["active_session_lease"].lease_id != lease.lease_id
    assert chat.workers == []


def test_real_wait_thread_dispatches_after_release(chat, monkeypatch):
    lease = chat.hold()
    monkeypatch.setattr(server, "_start_session_work", _START_SESSION_WORK)
    try:
        assert submit()["result"]["status"] == "queued"
        worker = chat.session["_run_thread"]
        assert worker.is_alive()
        assert chat.dispatched == []
        lease.release()
        worker.join(5)
        assert not worker.is_alive()
        assert [turn[0] for turn in chat.dispatched] == ["next"]
    finally:
        lease.release()


def test_stop_only_cancels_desktop_queue_not_delivery_delegations(chat, monkeypatch):
    import tools.async_delegation as delegation
    import tools.approval as approval
    calls = []
    monkeypatch.setattr(delegation, "interrupt_for_session", lambda **kw: calls.append("delegation"))
    monkeypatch.setattr(approval, "resolve_gateway_approval", lambda *a, **kw: calls.append("approval"))
    lease = chat.hold()
    submit()
    server._interrupt_session_turn("sid", chat.session)
    chat.workers[0]()
    assert calls == []
    assert chat.dispatched == []
    assert registry.active_session_registry_snapshot(registry_home=chat.home)[0]["lease_id"] == lease.lease_id
    lease.release()


def test_disconnected_wait_is_not_reaped(chat, monkeypatch):
    timers = []
    class Timer:
        def __init__(self, delay, callback):
            self.callback = callback
            timers.append(self)
        def start(self):
            pass
        def cancel(self):
            pass
    monkeypatch.setattr(server.threading, "Timer", Timer)
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *a: False)
    lease = chat.hold()
    submit()
    chat.session["transport"] = server._detached_ws_transport
    server._schedule_ws_orphan_reap("sid")
    timers[0].callback()
    assert server._sessions["sid"] is chat.session
    assert len(timers) == 2
    lease.release()
