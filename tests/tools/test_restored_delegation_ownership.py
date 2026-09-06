"""Regression coverage for #64484 — durable-restored delegation completions
must never be adopted by a session that cannot positively prove ownership.

Fixture timestamps are recent (now-based): restore_undelivered_completions
terminally drops pending completions older than _MAX_COMPLETION_REPLAY_AGE_S,
so epoch-era toy timestamps would exercise the staleness cap instead of the
restored-flag contract under test here.

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
import time

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
        "dispatched_at": time.time() - 2.0,
        "completed_at": time.time() - 1.0,
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
        "dispatched_at": time.time() - 2.0,
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


def test_owns_event_callback_beats_restored_flag():
    """A positive-proof ownership callback consumes restored events it owns."""
    reg = _make_registry()
    reg.completion_queue.put(_delegation_event(session_key="OWNER", restored=True))

    results = reg.drain_notifications(
        owns_event=lambda e: e.get("session_key") == "OWNER"
    )

    assert len(results) == 1
    assert reg.completion_queue.empty()


def test_foreign_owner_callback_requeues_restored_event():
    """A foreign owner filter cannot consume a restored completion."""
    reg = _make_registry()
    event = _delegation_event(session_key="OWNER", restored=True)
    reg.completion_queue.put(event)

    results = reg.drain_notifications(
        session_key="FOREIGN",
        owns_event=lambda candidate: candidate.get("session_key") == "FOREIGN",
    )

    assert results == []
    assert reg.completion_queue.get_nowait() == event
    assert reg.completion_queue.empty()


def test_absent_owner_filter_requeues_restored_event():
    """A legacy unfiltered drain cannot adopt a restored completion."""
    reg = _make_registry()
    event = _delegation_event(session_key="OWNER", restored=True)
    reg.completion_queue.put(event)

    results = reg.drain_notifications()

    assert results == []
    assert reg.completion_queue.get_nowait() == event
    assert reg.completion_queue.empty()


def test_restored_owner_callback_exception_fails_closed():
    """A broken owner check leaves the restored completion recoverable."""
    reg = _make_registry()
    event = _delegation_event(session_key="OWNER", restored=True)
    reg.completion_queue.put(event)

    def broken(_event):
        raise RuntimeError("synthetic ownership failure")

    results = reg.drain_notifications(owns_event=broken)

    assert results == []
    assert reg.completion_queue.get_nowait() == event
    assert reg.completion_queue.empty()


def test_matching_owner_claims_and_acknowledges_restored_event_once(
    tmp_path, monkeypatch,
):
    """The verified owner can claim and acknowledge a restored row once."""
    import tools.async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "async_delegations.db")
    event = _delegation_event(session_key="OWNER", delegation_id="d-owner")
    ad._persist_dispatch({
        "delegation_id": event["delegation_id"],
        "session_key": event["session_key"],
        "origin_ui_session_id": "",
        "parent_session_id": "OWNER",
        "dispatched_at": event["dispatched_at"],
    })
    ad._persist_completion(event, {
        "status": "success",
        "summary": event["summary"],
    })

    restored_queue = queue.Queue()
    assert ad.restore_undelivered_completions(restored_queue) == 1
    restored = restored_queue.get_nowait()
    assert restored["restored"] is True

    reg = _make_registry()
    reg.completion_queue.put(restored)
    results = reg.drain_notifications(
        owns_event=lambda candidate: candidate.get("session_key") == "OWNER",
    )
    assert len(results) == 1

    claim_id = ad.claim_event_delivery(restored, "owner-consumer")
    assert claim_id
    ad.complete_event_delivery(restored, claim_id)
    assert ad.get_durable_delegation("d-owner")["delivery_state"] == "delivered"
    assert ad.claim_event_delivery(restored, "foreign-after-ack") is None
