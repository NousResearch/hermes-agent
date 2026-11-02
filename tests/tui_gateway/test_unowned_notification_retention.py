"""An addressed notification whose owner session is not live must survive until that owner returns.

Every TUI/Desktop/dashboard session's poller drains ONE process-wide completion queue. A session can end
(explicit close, a reap) while a ``notify_on_complete`` process or an async delegation it started is still
running, and ``resume`` brings the very same ``session_key`` back. Whichever poller drained the event first
used to drop it (#118653, #129071), and the closing session's own shutdown drain lost what it could not
route (#140718). The event is parked instead (bounded), and only its owner's poller claims it.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from types import SimpleNamespace

import pytest

from tui_gateway import notification_park, server

OWNER_KEY = "owner-session-key"
OTHER_KEY = "unrelated-session-key"


def _session(key: str) -> dict:
    return {"session_key": key, "history_lock": threading.RLock(), "running": False, "history": [],
            "agent": None, "profile_home": "", "_finalized": False}


def _completion(process_id: str = "proc_owner1", key: str = OWNER_KEY) -> dict:
    return {"type": "completion", "session_id": process_id, "session_key": key,
            "task_id": f"session:{key}", "owner_task_id": key,
            "command": "make slow-thing", "exit_code": 0, "completion_reason": "exited", "output": "done"}


def _delegation(delegation_id: str = "deleg_owner1", **extra) -> dict:
    return {"type": "async_delegation", "delegation_id": delegation_id, "session_key": OWNER_KEY,
            "origin_ui_session_id": "closed-ui-sid", "results": [], **extra}


def _registry() -> SimpleNamespace:
    return SimpleNamespace(completion_queue=queue.Queue(), is_completion_consumed=lambda _sid: False,
                           restore_completions=lambda: 0)


def _parked() -> list:
    return notification_park.parked()


class _Batch:
    def __init__(self, events):
        self.events = events

    def render(self, registry):
        return "batch"

    def display_text(self, registry):
        return "batch"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    """The park, the session registry and the lineage store are process-global: pin all three."""
    monkeypatch.setattr(notification_park, "_parked", [])
    monkeypatch.setattr(server, "_sessions", {})

    @contextlib.contextmanager
    def _fake_session_db(_session_):
        yield SimpleNamespace(resolve_resume_session_id=lambda key: key)

    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda evt, consumer: "claim")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda evt, claim: None)
    monkeypatch.setattr("tools.process_registry_notifications.ProcessNotificationBatch", _Batch)
    for name in ("_maybe_fire_tui_loop_tick", "_maybe_fire_tui_heartbeat_tick", "_notif_poll_kanban"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr("tools.async_delegation.maybe_sweep_orphaned_completions", lambda q: 0)


@pytest.mark.parametrize("make_event", [_completion, _delegation], ids=["notify_on_complete", "async_delegation"])
def test_resumed_owner_gets_what_a_foreign_poller_could_not_deliver(monkeypatch, make_event):
    """The foreign poller neither delivers the event nor puts it back in the shared queue (that spins every
    live poller); a foreign session never claims it; the resumed owner's real poller loop delivers it once."""
    evt = make_event()
    other = _session(OTHER_KEY)
    monkeypatch.setattr(server, "_sessions", {"other-sid": other})
    registry = _registry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    started: list = []
    stop = threading.Event()

    def _submit(rid, sid, session, text, **kwargs):
        started.append(sid)
        stop.set()
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", _submit)

    server._notif_handle_ready("other-sid", other, [evt], set(), registry, lambda _evt: "text", None)
    assert registry.completion_queue.empty()
    assert started == [] and _parked() == [evt]
    assert notification_park.claim(lambda evt: server._session_owns_notification_event("other-sid", other, evt)) == []

    owner = _session(OWNER_KEY)
    monkeypatch.setattr(server, "_sessions", {"other-sid": other, "resumed-sid": owner})
    worker = threading.Thread(target=server._notification_poller_scoped_loop,
                              args=(stop, "resumed-sid", owner), daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert started == ["resumed-sid"]
    assert _parked() == []


def test_the_finalizing_owners_drain_parks_instead_of_dropping(monkeypatch):
    """The closing session's own shutdown drain runs with the session already ``_finalized``, so it no longer
    proves ownership of its own events; they must still reach the resumed successor."""
    owner = _session(OWNER_KEY)
    owner["_finalized"] = True
    evt = _completion()
    monkeypatch.setattr(server, "_sessions", {"owner-sid": owner})
    registry = _registry()

    server._notif_handle_ready("owner-sid", owner, [evt], set(), registry, lambda _evt: "text", [])

    assert registry.completion_queue.empty()
    assert _parked() == [evt]


def test_a_delegation_the_park_lets_go_of_goes_back_to_the_ledger(monkeypatch):
    """Bounded retention must not turn into silent loss: a parked delegation past its window hands its offer
    back to the durable ledger, and a startup replay (its owner process is dead, so the orphan sweep re-offers
    it) is never parked at all."""
    returned: list = []
    monkeypatch.setattr("tools.async_delegation.return_completion_offer", returned.append)
    monkeypatch.setattr(notification_park, "RETENTION_SECONDS", 0.0)
    live = _delegation()
    assert notification_park.park(live) is True
    owner = _session(OWNER_KEY)

    assert notification_park.claim(lambda evt: server._session_owns_notification_event("owner-sid", owner, evt)) == []
    assert returned == [live] and _parked() == []
    assert notification_park.park(_delegation("deleg_replayed", restored=True)) is False


def test_shutdown_drain_hands_back_what_it_could_not_route(monkeypatch):
    """When routing one drained event raises, the events after it (here one owned by another live session)
    are the drain's only in-memory copies and must go back to the shared queue (#140718)."""
    owner, other = _session(OWNER_KEY), _session(OTHER_KEY)
    monkeypatch.setattr(server, "_sessions", {"owner-sid": owner, "other-sid": other})
    registry = _registry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)

    def _format(evt):
        if evt.get("session_id") == "proc_broken":
            raise RuntimeError("unformattable")
        return "text"

    monkeypatch.setattr("tools.process_registry_notifications.format_process_notification", _format)
    broken, foreign = _completion("proc_broken"), _completion("proc_foreign", key=OTHER_KEY)
    registry.completion_queue.put(broken)
    registry.completion_queue.put(foreign)
    stop = threading.Event()
    stop.set()  # straight to the shutdown drain

    with contextlib.suppress(Exception):
        server._notification_poller_scoped_loop(stop, "owner-sid", owner)

    assert list(registry.completion_queue.queue) == [foreign]
