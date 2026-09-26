"""A notification that claims the session's turn must hand it back whenever no turn runs.

The poller claims the durable event before reserving the idle session turn. A competing claim
or unreadable ledger must leave ``running`` false: otherwise a busy session is exempt from the
reaper and never polls its bot mailbox again. Dispatch failures must release the reservation too.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server

DELEGATION = {"type": "async_delegation", "delegation_id": "deleg-1", "session_key": "stored"}


def _claimed_session() -> dict:
    session = {"history_lock": threading.RLock(), "running": False, "history": []}
    assert server._notif_claim_turn(session) is True
    return session


def _no_turn(monkeypatch) -> list:
    started: list = []
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: started.append(a))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    return started


@pytest.mark.parametrize("claim", [None, sqlite3.OperationalError("database is locked")],
                         ids=["row-held-by-another-consumer", "ledger-unreadable"])
def test_a_lost_delivery_claim_hands_the_turn_back(monkeypatch, claim):
    """A gateway sharing this home claims the durable row before it verifies the target, so the
    poller holding the live copy of the same event gets ``None`` — after it already took the turn."""
    def _claim(evt, consumer):
        if isinstance(claim, Exception):
            raise claim
        return claim

    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", _claim)
    started = _no_turn(monkeypatch)
    session = {"history_lock": threading.RLock(), "running": False, "history": []}
    registry = SimpleNamespace(completion_queue=queue.Queue(), completion_routing_lock=threading.RLock())
    server._notif_handle_ready("sid", session, [dict(DELEGATION)], set(), registry,
                               lambda evt: "text", None, owned=True)

    assert session["running"] is False
    assert started == []
    assert server._notif_claim_turn(session) is True, "the session must be claimable again"


@pytest.mark.parametrize("fail_at", ["claim", "render"])
def test_a_completion_batch_that_cannot_be_prepared_hands_the_turn_back(monkeypatch, fail_at):
    events = [{"type": "completion", "session_id": "proc_a"}, {"type": "completion", "session_id": "proc_b"}]
    released: list = []
    claims = iter(["claim-a", sqlite3.OperationalError("database is locked")] if fail_at == "claim"
                  else ["claim-a", "claim-b"])

    def _claim(evt, consumer):
        value = next(claims)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", _claim)
    monkeypatch.setattr("tools.async_delegation.release_event_delivery", lambda evt, c: released.append(c))
    if fail_at == "render":
        monkeypatch.setattr("tools.process_registry_notifications.ProcessNotificationBatch.render",
                            lambda self, registry: (_ for _ in ()).throw(ValueError("bad payload")))
    started = _no_turn(monkeypatch)
    session = {"history_lock": threading.RLock(), "running": False, "history": []}

    server._notif_dispatch_completions("sid", session, [(e, "t") for e in events],
                                       SimpleNamespace(completion_queue=queue.Queue()), None)

    assert session["running"] is False and started == []
    # Claims already taken go back too, or those completions are lost to every consumer for 300 s.
    assert released == (["claim-a"] if fail_at == "claim" else ["claim-a", "claim-b"])


def test_a_loop_wakeup_whose_send_cannot_start_hands_the_turn_back(monkeypatch):
    """The /loop slash wakeup re-claims the turn for a command that resolves to a prompt, then runs
    the send under ``except Exception: pass`` — which swallowed the only signal that no turn started."""
    monkeypatch.setitem(server._methods, "command.dispatch",
                        lambda rid, params: {"result": {"type": "send", "message": "run the skill"}})
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no free worker")))
    ticks: list = []
    mgr = SimpleNamespace(abandon_tick=lambda: ticks.append("abandoned"),
                          complete_tick=lambda text: ticks.append("completed") or {})
    session = _claimed_session()

    server._notif_slash_loop_tick("rid", "sid", session, mgr, "/skill go")

    assert session["running"] is False
    assert ticks == ["completed"]


def test_the_poller_thread_survives_a_dispatch_that_raises(monkeypatch):
    """The poller is the session's only path to notifications, /loop, /heartbeat and its bot
    mailbox; an exception out of one event's dispatch used to end the thread for good."""
    events: queue.Queue = queue.Queue()
    events.put({"type": "completion", "session_id": "proc_a"})
    monkeypatch.setattr("tools.process_registry.process_registry", SimpleNamespace(
        completion_queue=events, completion_routing_lock=threading.RLock(),
        is_completion_consumed=lambda sid: False))
    for name in ("_poll_bot_live_delivery_guarded", "_maybe_fire_tui_loop_tick",
                 "_maybe_fire_tui_heartbeat_tick", "_notif_poll_kanban"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    stop = threading.Event()
    handled: list = []

    def _handle_ready(sid, session, ready, emitted, registry, fmt, deferred, **kwargs):
        if deferred is not None:  # the post-stop drain
            return
        handled.append(len(ready))
        if len(handled) == 1:
            events.put({"type": "completion", "session_id": "proc_b"})
            raise RuntimeError("one bad event")
        stop.set()

    monkeypatch.setattr(server, "_notif_handle_ready", _handle_ready)
    session = {"history_lock": threading.RLock(), "running": False, "history": []}
    worker = threading.Thread(target=server._notification_poller_scoped_loop, args=(stop, "sid", session), daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert handled == [1, 1], "the second event must still be dispatched after the first one raised"
