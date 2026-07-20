"""Regression coverage for #64484 — durable-restored delegation completions
must never be adopted by a session that cannot positively prove ownership.

Layers under test:
1. ``restore_undelivered_completions`` stamps every restored event with
   ``restored=True`` (in-memory only).
2. ``ProcessRegistry.drain_notifications`` with NO filter (legacy
   consume-everything CLI path) re-queues restored events instead of
   consuming them.
3. Same-process (non-restored) keyless events keep the legacy behavior.
4. An owner with a matching session_key still receives its restored event.
"""

import json
import queue

from tools.process_registry import ProcessRegistry


def _make_registry():
    reg = ProcessRegistry.__new__(ProcessRegistry)
    import threading

    reg._running = {}
    reg._finished = {}
    reg._lock = threading.Lock()
    reg.completion_queue = queue.Queue()
    reg._completion_consumed = set()
    reg._poll_observed = set()
    return reg


def _delegation_event(session_key="", restored=False, delegation_id="d1"):
    evt = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": session_key,
        "origin_ui_session_id": "",
        "goal": "secret goal",
        "status": "success",
        "summary": "SECRET RESULT",
        "api_calls": 3,
        "duration_seconds": 1.5,
        "dispatched_at": 1.0,
        "completed_at": 2.0,
    }
    if restored:
        evt["restored"] = True
    return evt


def test_restore_stamps_restored_flag(tmp_path, monkeypatch):
    """Every durable completion re-enqueued at startup carries restored=True."""
    import tools.async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "async_delegations.db")
    record = {
        "delegation_id": "d-old",
        "goal": "old goal",
        "context": None,
        "toolsets": None,
        "role": "leaf",
        "model": "m",
        "session_key": "OLD_SESSION_A",
        "origin_ui_session_id": "",
        "parent_session_id": "OLD_SESSION_A",
        "status": "running",
        "dispatched_at": 1.0,
        "completed_at": None,
        "interrupt_fn": None,
    }
    ad._persist_dispatch(record)
    evt = _delegation_event(session_key="OLD_SESSION_A", delegation_id="d-old")
    ad._persist_completion(evt, {"summary": "SECRET RESULT"})

    q = queue.Queue()
    restored = ad.restore_undelivered_completions(q)
    assert restored == 1
    got = q.get_nowait()
    assert got["restored"] is True
    assert got["session_key"] == "OLD_SESSION_A"

    # The stamp is in-memory only — the durable payload is unchanged.
    with ad._connect() as conn:
        row = conn.execute(
            "SELECT event_json FROM async_delegations WHERE delegation_id='d-old'"
        ).fetchone()
    assert "restored" not in json.loads(row[0])


def test_unfiltered_drain_never_consumes_restored_events():
    """The legacy consume-everything branch must fail closed on restored events."""
    reg = _make_registry()
    reg.completion_queue.put(_delegation_event(session_key="DEAD_SESSION", restored=True))

    results = reg.drain_notifications()  # no filter — legacy CLI post-turn shape

    assert results == []
    # Still queued for its real owner.
    assert reg.completion_queue.qsize() == 1
    assert reg.completion_queue.get_nowait()["session_key"] == "DEAD_SESSION"


def test_unfiltered_drain_keeps_legacy_behavior_for_same_process_events():
    """Non-restored keyless events (created by this process) are still consumed."""
    reg = _make_registry()
    reg.completion_queue.put(_delegation_event(session_key=""))

    results = reg.drain_notifications()

    assert len(results) == 1
    assert results[0][0]["delegation_id"] == "d1"
    assert reg.completion_queue.empty()


def test_owner_session_key_drain_consumes_restored_event():
    """The owning session (key match) still receives its restored completion."""
    reg = _make_registry()
    reg.completion_queue.put(_delegation_event(session_key="OWNER", restored=True))

    results = reg.drain_notifications(session_key="OWNER")

    assert len(results) == 1
    assert results[0][0]["session_key"] == "OWNER"
    assert reg.completion_queue.empty()


def test_foreign_session_key_drain_requeues_restored_event():
    """A different session's keyed drain must not claim the restored event."""
    reg = _make_registry()
    reg.completion_queue.put(_delegation_event(session_key="OWNER", restored=True))

    results = reg.drain_notifications(session_key="SOMEONE_ELSE")

    assert results == []
    assert reg.completion_queue.qsize() == 1


def test_owns_event_callback_beats_restored_flag():
    """A positive-proof ownership callback consumes restored events it owns."""
    reg = _make_registry()
    reg.completion_queue.put(_delegation_event(session_key="OWNER", restored=True))

    results = reg.drain_notifications(
        owns_event=lambda e: e.get("session_key") == "OWNER"
    )

    assert len(results) == 1
    assert reg.completion_queue.empty()
