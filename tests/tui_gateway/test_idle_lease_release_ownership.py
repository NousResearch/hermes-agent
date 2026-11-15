"""An idle-released Desktop lease (#146735) never lets this record write a transcript another surface owns.

Real SQLite store, real active-session registry, real reaper tick and real JSON-RPC handlers; the
competing owner is a separate registry lease (writers differ by pid AND live_session_id).
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


@pytest.fixture
def resident(monkeypatch):
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    db = SessionDB(db_path=get_hermes_home() / "state.db")
    monkeypatch.setattr(server, "_db", db)
    records, competitors = [], []

    def add(sid: str) -> dict:
        db.create_session(sid, source="desktop")
        ready = threading.Event()
        ready.set()
        long_ago = time.time() - 3600
        record = {
            "session_key": sid, "source": "desktop", "history": [], "history_version": 0,
            "history_lock": threading.Lock(), "agent_ready": ready, "running": False,
            "created_at": long_ago, "last_active": long_ago, "transport": _LiveSocket(),
            "agent": SimpleNamespace(session_id=sid, clear_interrupt=lambda: None),
            "attached_images": [], "cols": 80}
        server._sessions[sid] = record
        records.append(record)
        assert server._ensure_active_session_slot(sid, record) is None
        return record

    def compete(sid: str):
        lease, refusal = try_acquire_active_session(session_id=sid, surface="telegram", config={})
        if lease is not None:
            competitors.append(lease)
        return lease, refusal

    yield SimpleNamespace(add=add, compete=compete, db=db)
    for lease in competitors:
        lease.release()
    for record in records:
        if thread := record.get("_run_thread"):
            thread.join(10)
        server._release_active_session_slot(record)
        server._sessions.pop(record["session_key"], None)
    db.close()


def _seed(db, sid: str) -> list[int]:
    ids = []
    for n in range(3):
        ids.append(db.append_message(sid, "user", f"question {n}"))
        db.append_message(sid, "assistant", f"answer {n}")
    return ids


def _active_contents(db, sid: str) -> list[str]:
    return [m["content"] for m in db.get_messages_as_conversation(sid)]


@pytest.mark.parametrize("path", ["prompt.submit rewind", "session.undo"])
def test_idle_released_record_never_rewinds_the_new_owners_transcript(resident, monkeypatch, path):
    sid = path.replace(" ", "-").replace(".", "-")
    record = resident.add(sid)
    ids = _seed(resident.db, sid)
    record["history"] = resident.db.get_messages_as_conversation(sid, include_row_ids=True)

    def other_surface_takes_the_chat(*, writes: bool = True):
        server._reap_idle_sessions()  # the real idle tick releases the lease
        lease, _refusal = resident.compete(sid)
        assert lease is not None, "precondition: the idle release freed the chat"
        if writes:
            resident.db.append_message(sid, "user", "telegram message")
            resident.db.append_message(sid, "assistant", "telegram answer")

    if path == "session.undo":
        # The new owner has claimed the chat but not written yet: the warm history still matches the store.
        other_surface_takes_the_chat(writes=False)
        request = {"method": "session.undo", "params": {"session_id": sid}}
    else:
        # prompt.submit checks ownership, then expands the replayed text, then takes history_lock to apply
        # the cut: schedule the reaper tick and the competing claim in exactly that window.
        expand = server._expand_skill_invocation_for_replay

        def expand_after_tick(text, key):
            other_surface_takes_the_chat()
            return expand(text, key)

        monkeypatch.setattr(server, "_expand_skill_invocation_for_replay", expand_after_tick)
        request = {"method": "prompt.submit", "params": {
            "session_id": sid, "text": "edited", "truncate_before_row_id": ids[1],
            "truncate_before_user_ordinal": 1, "confirm_truncate": True, "confirm_deep_truncate": True}}

    response = server.handle_request({"id": "r", **request})
    if thread := record.get("_run_thread"):
        thread.join(10)

    owner_rows = ["telegram message", "telegram answer"] if path != "session.undo" else []
    assert {"refused": "error" in response, "transcript": _active_contents(resident.db, sid)} == {
        "refused": True,
        "transcript": ["question 0", "answer 0", "question 1", "answer 1", "question 2", "answer 2", *owner_rows]}


@pytest.mark.parametrize("work", ["loop", "heartbeat", "process_notification"])
def test_idle_release_keeps_the_lease_of_a_record_driving_local_wakeups(resident, work):
    from hermes_cli.heartbeat import HeartbeatManager
    from hermes_cli.loops import LoopManager
    from tools.process_registry import process_registry
    from tools.process_registry_awaiting import release, take_ready

    sid = f"wakeup-{work}"
    resident.add(sid)
    manager = None
    if work == "loop":
        manager = LoopManager(sid)
    elif work == "heartbeat":
        manager = HeartbeatManager(sid)
    if manager is not None:
        manager.set("check the build", interval_seconds=1800)
    else:
        process_registry.completion_queue.put(
            {"type": "process_exit", "owner_task_id": sid, "session_id": "proc-1"})
    try:
        server._reap_idle_sessions()
        lease, refusal = resident.compete(sid)
        assert (lease, refusal is not None) == (None, True), "a pending local wakeup lost its chat"
    finally:
        if manager is not None:
            manager.clear()
        else:
            release(take_ready(process_registry.completion_queue, 0))


def test_a_refused_loop_wakeup_rolls_its_tick_back(resident):
    """A due tick whose turn is refused at admission (another surface owns the chat) never settles: it is
    abandoned back to the countdown instead of leaving the loop awaiting a response forever."""
    from hermes_cli.loops import LoopManager, save_loop

    sid = "refused-loop"
    record = resident.add(sid)
    server._release_active_session_slot(record)
    lease, _refusal = resident.compete(sid)
    assert lease is not None
    manager = LoopManager(sid)
    manager.set("check the build", interval_seconds=1800)
    manager.state.next_due_at = time.time() - 1
    save_loop(sid, manager.state)
    try:
        server._maybe_fire_tui_loop_tick(sid, record)
        state = LoopManager(sid).state
        assert {"awaiting_response": state.awaiting_response, "ticks_fired": state.ticks_fired,
                "running": record["running"]} == {"awaiting_response": False, "ticks_fired": 0, "running": False}
    finally:
        manager.clear()
