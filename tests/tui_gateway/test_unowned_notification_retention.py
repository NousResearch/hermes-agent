"""An addressed notification whose owner session is not live must survive until that owner returns.

Every TUI/Desktop/dashboard session's poller drains ONE process-wide completion queue. A session can end
(explicit close, a reap) while a ``notify_on_complete`` process or an async delegation it started is still
running, and ``resume`` brings the very same ``session_key`` back. Whichever poller drained the event first
used to drop it (#118653, #129071), and the closing session's own shutdown drain lost what it could not
route (#140718). The event is parked instead (bounded), and only its owner's poller claims it.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import queue
import threading
import time
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
    """The park, its held-copy marks, the session registry and the lineage store are process-global: pin them."""
    monkeypatch.setattr(notification_park, "_parked", [])
    monkeypatch.setattr("tools.async_delegation._held_copies", collections.Counter())
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
    from tools import async_delegation
    assert sum(async_delegation._held_copies.values()) == (evt["type"] == "async_delegation")
    assert notification_park.claim(lambda evt: server._session_owns_notification_event("other-sid", other, evt)) == []

    owner = _session(OWNER_KEY)
    monkeypatch.setattr(server, "_sessions", {"other-sid": other, "resumed-sid": owner})
    worker = threading.Thread(target=server._notification_poller_scoped_loop,
                              args=(stop, "resumed-sid", owner), daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert started == ["resumed-sid"]
    assert _parked() == []
    from tools import async_delegation
    assert not async_delegation._held_copies  # the claimed copy's park mark ended once it was routed


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


def test_a_delegation_the_park_lets_go_of_is_a_loud_loss_with_its_offer_cleared(monkeypatch, caplog):
    """Bounded retention must not turn into silent loss: a parked delegation past its window is logged at WARNING
    (as an unparked drop is) and its sweep offer is cleared; a startup replay (its owner process is dead, so the
    orphan sweep re-offers it) is never parked at all."""
    caplog.set_level("WARNING", logger=notification_park.__name__)
    returned: list = []
    monkeypatch.setattr("tools.async_delegation.return_completion_offer", returned.append)
    monkeypatch.setattr(notification_park, "RETENTION_SECONDS", 0.0)
    live = _delegation()
    assert notification_park.park(live) is True
    owner = _session(OWNER_KEY)

    assert notification_park.claim(lambda evt: server._session_owns_notification_event("owner-sid", owner, evt)) == []
    assert returned == [live] and _parked() == []
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert notification_park.park(_delegation("deleg_replayed", restored=True)) is False


def test_a_result_displaces_progress_notices_instead_of_being_refused(monkeypatch):
    """A closed session's watch_patterns process can match more often than its per-owner slots allow before it
    exits: the completion the user waits for must still be parked (the oldest progress notice yields), and a
    full park of other owners' progress notices must not refuse it either."""
    other = _session(OTHER_KEY)
    registry = _registry()
    watches = [{**_completion(), "type": "watch_match", "output": f"match {i}"}
               for i in range(notification_park.MAX_PARKED_PER_KEY)]
    done = _completion()

    server._notif_handle_ready("other-sid", other, [*watches, done], set(), registry, lambda _evt: "text", None)

    assert _parked() == [*watches[1:], done]
    assert registry.completion_queue.empty()
    monkeypatch.setattr(notification_park, "MAX_PARKED", len(_parked()))
    deleg = _delegation(session_key="third-session-key")
    assert notification_park.park(deleg) is True and deleg in _parked() and watches[1] not in _parked()
    assert notification_park.park({**watches[0], "session_key": "third-session-key"}) is False


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


def test_the_orphan_sweep_counts_a_parked_delegation_as_live_until_the_park_lets_go(tmp_path, monkeypatch, caplog):
    """The park is out of the ledger sweep's sight: without the held-copy mark, a live delegation parked past a
    claim lease reads as lost, so the sweep warns and queues a duplicate every lease. While parked it is never
    re-offered; once the park lets go of it (TTL) the sweep re-offers the still-pending row."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tests.waiting import wait_until
    from tools import async_delegation as ad
    from tools.process_registry_awaiting import release, take_ready

    home = tmp_path / "home"
    home.mkdir()
    registry = _registry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    ad._reset_for_tests()
    token = set_hermes_home_override(str(home))
    try:
        handle = ad.dispatch_async_delegation(
            goal="mine", context=None, toolsets=None, role="leaf", model="m", session_key=OWNER_KEY,
            runner=lambda: {"status": "completed", "summary": "done"})
        wait_until(lambda: ad.active_count() == 0, timeout=10, what="delegation finished")
        other, sweep_q = _session(OTHER_KEY), queue.Queue()
        monkeypatch.setattr(server, "_sessions", {"other-sid": other})
        ready = take_ready(registry.completion_queue, 0)  # a foreign poller drains the owner's result...
        server._notif_handle_ready("other-sid", other, ready, set(), registry, lambda _evt: "text", None)
        release(ready)  # ...and parks it
        assert [evt["delegation_id"] for evt in _parked()] == [handle["delegation_id"]]
        lease = time.time() + ad._CLAIM_LEASE_S + 1
        caplog.set_level(logging.WARNING, logger="tools.async_delegation")

        assert ad.sweep_orphaned_completions(sweep_q, now=lease) == 0
        assert sweep_q.empty() and not [r for r in caplog.records if "Re-offering" in r.getMessage()]

        monkeypatch.setattr(notification_park, "time", SimpleNamespace(monotonic=lambda: float("inf")))  # past the TTL
        assert notification_park.claim(lambda _evt: False) == [] and _parked() == []
        assert ad.sweep_orphaned_completions(sweep_q, now=lease + ad._CLAIM_LEASE_S) == 1
        assert [evt["delegation_id"] for evt in _drain(sweep_q)] == [handle["delegation_id"]]
    finally:
        reset_hermes_home_override(token)
        ad._reset_for_tests()


@pytest.mark.parametrize("evt", [
    {"type": "async_delegation", "delegation_id": "deleg_gw1", "session_key": "agent:main:telegram:dm:12345", "results": []},
    {"type": "async_delegation", "delegation_id": "deleg_gw2", "session_key": "agent:main:local:dm:local-abc", "results": []},
    {"type": "async_delegation", "delegation_id": "deleg_gw3", "session_key": "agent:trader:discord:channel:99", "results": [], "restored": True},
    {"type": "async_delegation", "delegation_id": "deleg_gw4", "session_key": "agent:main:telegram:dm:12345", "task_failure_notice": True, "results": [{"task_index": 0, "status": "failed"}]},
    {"type": "watch_match", "session_id": "proc_gw1", "session_key": "agent:main:telegram:dm:12345", "pattern": "ERR", "output": "boom"},
    {"type": "watch_disabled", "session_id": "proc_gw1", "session_key": "agent:main:telegram:dm:12345", "message": "capped"},
    {"type": "heartbeat", "session_id": "proc_gw1", "session_key": "agent:main:telegram:dm:12345", "seq": 1, "output": "tick"},
    {"type": "async_delegation", "delegation_id": "deleg_api1", "session_key": "raw-api-sid", "origin_session_id": "raw-api-sid", "results": []},
], ids=["telegram_dm", "local_dm", "restored_named_profile", "task_failure_notice", "watch_match", "watch_disabled", "heartbeat", "api_server_origin"])
def test_embedded_tui_poller_leaves_gateway_owned_events_on_the_shared_queue(monkeypatch, evt):
    """When an in-process GatewayRunner is live, embedded TUI pollers must requeue gateway-routed events
    instead of parking them in ``notification_park`` (which starves ``_async_delegation_watcher`` for 600s
    and silently drops ``restored=True`` orphan-sweep replays)."""
    import sys
    from tools import async_delegation

    other = _session(OTHER_KEY)
    monkeypatch.setattr(server, "_sessions", {"other-sid": other})
    monkeypatch.setattr(server, "time", SimpleNamespace(sleep=lambda _s: None, time=time.time, monotonic=time.monotonic))
    registry = _registry()
    runner = SimpleNamespace(_running=True, _gateway_loop=SimpleNamespace(is_running=lambda: True))
    fake_run_mod = SimpleNamespace(_gateway_runner_ref=lambda: runner)
    monkeypatch.setitem(sys.modules, "gateway.run", fake_run_mod)

    server._notif_handle_ready("other-sid", other, [evt], set(), registry, lambda _evt: "text", None)

    assert _parked() == []
    assert not async_delegation._held_copies
    assert _drain(registry.completion_queue) == [evt]

    # During shutdown drain (deferred list provided), the event is also handed back to the queue, never parked.
    deferred: list = []
    server._notif_handle_ready("other-sid", other, [evt], set(), registry, lambda _evt: "text", deferred)
    assert _parked() == []
    assert deferred == [evt]


def test_resumed_tui_session_does_not_steal_in_flight_api_server_delegation(monkeypatch):
    """An api_server delegation carries ``origin_session_id`` (no ``origin_ui_session_id``) and must be
    persisted as a delivery row by the gateway without starting a model wake, even if a TUI tab later
    resumes the same raw session id (poller and post-turn ``drained_for`` paths alike)."""
    import sys
    from tools.process_registry import ProcessRegistry
    from tools.process_registry_awaiting import drained_for

    raw_sid = "raw-api-sid"
    resumed = _session(raw_sid)
    monkeypatch.setattr(server, "_sessions", {"tui-sid": resumed})
    monkeypatch.setattr(server, "time", SimpleNamespace(sleep=lambda _s: None, time=time.time, monotonic=time.monotonic))
    runner = SimpleNamespace(_running=True, _gateway_loop=SimpleNamespace(is_running=lambda: True))
    monkeypatch.setitem(sys.modules, "gateway.run", SimpleNamespace(_gateway_runner_ref=lambda: runner))

    api_evt = {
        "type": "async_delegation", "delegation_id": "deleg_api_resumed",
        "session_key": raw_sid, "origin_session_id": raw_sid, "results": [],
    }
    registry = _registry()
    registry.completion_queue.put(api_evt)

    owns = lambda e: ProcessRegistry._owns_event(
        e, raw_sid, lambda evt: server._session_owns_notification_event("tui-sid", resumed, evt), True)
    with drained_for(registry, owns) as mine:
        assert mine == []
    assert _drain(registry.completion_queue) == [api_evt]

    server._notif_handle_ready("tui-sid", resumed, [api_evt], set(), registry, lambda _evt: "text", None)
    assert _parked() == []
    assert _drain(registry.completion_queue) == [api_evt]


def test_inactive_gateway_runner_does_not_claim_unowned_events(monkeypatch):
    """A stopped GatewayRunner (or process completion not drained from completion_queue by the gateway)
    must not cause TUI pollers to requeue-spin unowned events."""
    import sys

    other = _session(OTHER_KEY)
    monkeypatch.setattr(server, "_sessions", {"other-sid": other})
    registry = _registry()
    stopped_runner = SimpleNamespace(_running=False, _gateway_loop=SimpleNamespace(is_running=lambda: False))
    monkeypatch.setitem(sys.modules, "gateway.run", SimpleNamespace(_gateway_runner_ref=lambda: stopped_runner))

    deleg = _delegation("deleg_stopped", session_key="agent:main:telegram:dm:12345", origin_ui_session_id="")
    server._notif_handle_ready("other-sid", other, [deleg], set(), registry, lambda _evt: "text", None)
    assert registry.completion_queue.empty()
    assert _parked() == [deleg]

    notification_park.unhold(_parked())
    monkeypatch.setattr(notification_park, "_parked", [])
    live_runner = SimpleNamespace(_running=True, _gateway_loop=SimpleNamespace(is_running=lambda: True))
    monkeypatch.setitem(sys.modules, "gateway.run", SimpleNamespace(_gateway_runner_ref=lambda: live_runner))
    proc_comp = _completion("proc_gw", key="agent:main:telegram:dm:12345")
    server._notif_handle_ready("other-sid", other, [proc_comp], set(), registry, lambda _evt: "text", None)
    assert registry.completion_queue.empty()
    assert _parked() == [proc_comp]


def _drain(q) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out
