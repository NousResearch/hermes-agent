"""Same-process heartbeats must not be delivered out of order (#135295).

The poller snapshot-drains the shared completion queue and requeues events one by one
while the session is busy; between two requeues (the 0.25 s back-off window) the
heartbeat producer keeps enqueueing, so a newer beat lands in front of a stale one and
front-to-back delivery spends a model turn on outdated output first. A stale beat's
payload is an output delta a newer beat already supersedes, so the delivery boundary
drops it (per-process ``seq`` monotonic gate) instead of delivering it late.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server


@pytest.fixture(autouse=True)
def _fresh_seq_gate():
    """The seq gate is process-global module state (like the queue itself); reset per test."""
    server._heartbeat_last_delivered_seq.clear()
    yield
    server._heartbeat_last_delivered_seq.clear()


@pytest.fixture
def surface(monkeypatch):
    submits: list = []
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda evt, consumer: "claimed")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a, **k: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda rid, sid, session, text, **kw: submits.append(text))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_owns_notification_event", lambda sid, session, evt: True)
    return submits


def _session() -> dict:
    return {"history_lock": threading.RLock(), "running": False, "history": [], "agent": None,
            "profile_home": None}


def _registry() -> SimpleNamespace:
    return SimpleNamespace(completion_queue=queue.Queue(), is_completion_consumed=lambda session_id: False)


def _beat(seq: int, process: str = "proc_hb") -> dict:
    return {"type": "heartbeat", "session_id": process, "seq": seq, "elapsed": 60.0, "interval": 60,
            "command": "npm test", "output": f"beat {seq}\n"}


_FMT = lambda evt: f"{evt.get('session_id')}-{evt.get('seq')}"


def _snapshot(registry) -> list:
    """One poller dequeue: the whole visible queue in one pass (session_notifications.py
    poller loop: ``queue.get`` then ``get_nowait`` until empty)."""
    events = []
    while not registry.completion_queue.empty():
        events.append(registry.completion_queue.get_nowait())
    return events


def _deliver_idle(server_fn, session, registry, emitted, submits):
    """The session is idle: drain front to back; every beat claims a turn that ends at once."""
    delivered: list[str] = []
    for evt in _snapshot(registry):
        submits.clear()
        assert server_fn("sid", session, evt, emitted, registry, _FMT, None, owned=False) is True
        server._notif_release_turn(session)
        delivered.extend(submits)
    return delivered


@pytest.mark.parametrize("owned", [False, True])
def test_stale_heartbeat_requeued_behind_newer_seq_is_dropped_at_delivery(surface, owned):
    """A beat requeued behind a newer beat of the same process is dropped, not delivered late.

    Deterministic replay of the poller rounds that produce the reporter's inversion: both
    requeues run while a user turn is live and the producer enqueues *inside* the per-event
    back-off window between them (time-shifted here between the two handles). ``owned``
    covers both real callers of the delivery boundary (False = the per-session poller,
    True = the post-turn safety net drain)."""
    server_fn = server._notif_handle_event
    session, registry, emitted = _session(), _registry(), set()

    assert server._notif_claim_turn(session) is True  # a user turn is live throughout

    # Round 1: snapshot sees only beat 1; busy → requeue.
    registry.completion_queue.put(_beat(1))
    batch = _snapshot(registry)
    assert [evt["seq"] for evt in batch] == [1]
    for evt in batch:
        assert server_fn("sid", session, evt, emitted, registry, _FMT, None, owned=owned) is True
    # The back-off window after the requeue: the producer queues beat 2.
    registry.completion_queue.put(_beat(2))

    # Round 2: snapshot sees [1, 2]; busy requeues run one by one — and the producer's
    # beat 3 lands inside the window BETWEEN the two requeues.
    batch = _snapshot(registry)
    assert [evt["seq"] for evt in batch] == [1, 2]
    assert server_fn("sid", session, batch[0], emitted, registry, _FMT, None, owned=owned) is True
    registry.completion_queue.put(_beat(3))
    assert server_fn("sid", session, batch[1], emitted, registry, _FMT, None, owned=owned) is True

    assert [evt["seq"] for evt in list(registry.completion_queue.queue)] == [1, 3, 2], \
        "precondition: the window insertion put the newer beat ahead of the stale one"

    # The user turn ends; delivery runs front to back.
    server._notif_release_turn(session)
    delivered = _deliver_idle(server_fn, session, registry, emitted, surface)
    assert delivered == ["proc_hb-1", "proc_hb-3"], \
        f"stale beat 2 must be dropped at the delivery boundary, got {delivered}"


def test_fresh_heartbeats_still_deliver_in_order_across_processes(surface):
    """Behavior guard: the monotonic gate is per PROCESS, not global — independent seq spaces
    never suppress each other, and fresh beats still deliver in queue order."""
    server_fn = server._notif_handle_event
    session, registry, emitted = _session(), _registry(), set()
    delivered: list[str] = []
    for evt in (_beat(1, "proc_a"), _beat(2, "proc_a"), _beat(1, "proc_b")):
        surface.clear()
        assert server_fn("sid", session, evt, emitted, registry, _FMT, None) is True
        server._notif_release_turn(session)
        delivered.extend(surface)
    assert delivered == ["proc_a-1", "proc_a-2", "proc_b-1"], delivered


def test_duplicate_seq_is_delivered_once_even_across_independent_emitted_sets(surface):
    """Poller and post-turn safety net each keep their own ``emitted`` set, so a beat requeued
    by one and later drained by the other passes the dedup gate a second time. The seq gate
    must still drop it: delivery is monotonic per process, and ``seq == last`` is stale too."""
    server_fn = server._notif_handle_event
    session, registry = _session(), _registry()
    outcomes: list[list[str]] = []
    for emitted in (set(), set()):
        surface.clear()
        assert server_fn("sid", session, _beat(2), emitted, registry, _FMT, None) is True
        server._notif_release_turn(session)
        outcomes.append(list(surface))
    assert outcomes == [["proc_hb-2"], []], \
        f"same seq through an independent emitted set must be dropped the second time: {outcomes}"
