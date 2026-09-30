"""A completion addressed to a session that is not live must survive until its owner claims it.

The session layer reaps a session whose client is gone (``ws_orphan_reap``, eviction) while a
``notify_on_complete`` process it started is still running, and ``resume`` brings the very same
``session_key`` back. Whichever *foreign* poller drained the shared queue first used to discard that
completion outright, so the wake the terminal tool promises never arrived. The event is parked
(bounded) instead, and the owner's poller claims it on its first pass after it is live again.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server

OWNER_KEY = "owner-session-key"
OTHER_KEY = "unrelated-session-key"


def _session(key: str) -> dict:
    return {"session_key": key, "history_lock": threading.RLock(), "running": False, "history": [],
            "agent": None, "profile_home": "", "_finalized": False}


def _completion(process_id: str = "proc_owner1") -> dict:
    return {"type": "completion", "session_id": process_id, "session_key": OWNER_KEY,
            "task_id": f"session:{OWNER_KEY}", "owner_task_id": OWNER_KEY,
            "command": "make slow-thing", "exit_code": 0, "completion_reason": "exited",
            "output": "done"}


def _registry() -> SimpleNamespace:
    return SimpleNamespace(completion_queue=queue.Queue(), is_completion_consumed=lambda _sid: False,
                           restore_completions=lambda: 0)


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    """The park list, the session registry and the lineage store are process-global: pin all three."""
    monkeypatch.setattr(server, "_unowned_parked", [], raising=False)
    monkeypatch.setattr(server, "_sessions", {})

    @contextlib.contextmanager
    def _fake_session_db(_session_):
        yield SimpleNamespace(resolve_resume_session_id=lambda key: key)

    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    yield


def test_a_foreign_poller_parks_a_reaped_owners_completion(monkeypatch):
    """The foreign drain must neither deliver it to the wrong session nor push it back into the
    shared queue: a foreign event left in the queue keeps ``queue.get`` returning immediately, so
    every live poller spins until its owner returns or the process ends."""
    other, evt = _session(OTHER_KEY), _completion()
    monkeypatch.setattr(server, "_sessions", {"other-sid": other})
    registry = _registry()

    server._notif_handle_ready("other-sid", other, [evt], set(), registry, lambda _evt: "text", None)

    assert registry.completion_queue.empty(), "a parked event must not be requeued into the hot queue"
    assert [parked for _deadline, parked in server._unowned_parked] == [evt]


def test_a_foreign_session_never_claims_a_parked_event(monkeypatch):
    evt = _completion()
    assert server._park_unowned_notification(evt) is True
    other = _session(OTHER_KEY)
    monkeypatch.setattr(server, "_sessions", {"other-sid": other})

    assert server._claim_parked_notifications("other-sid", other) == []
    assert [parked for _deadline, parked in server._unowned_parked] == [evt]


def test_the_resumed_owner_still_gets_its_completion(monkeypatch):
    """End of the story: the owner resumes, its real poller claims the parked event and starts the
    wake turn. Without the park the event was already gone by this point."""
    evt = _completion()
    other = _session(OTHER_KEY)
    monkeypatch.setattr(server, "_sessions", {"other-sid": other})
    registry = _registry()
    server._notif_handle_ready("other-sid", other, [evt], set(), registry, lambda _evt: "text", None)
    assert registry.completion_queue.empty()

    owner = _session(OWNER_KEY)
    monkeypatch.setattr(server, "_sessions", {"owner-sid": owner})
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    for name in ("_poll_bot_live_delivery_guarded", "_maybe_fire_tui_loop_tick",
                 "_maybe_fire_tui_heartbeat_tick", "_notif_poll_kanban"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda evt, consumer: "claim")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda evt, claim: None)
    monkeypatch.setattr("tools.process_registry_notifications.ProcessNotificationBatch", _Batch)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    started: list = []
    stop = threading.Event()

    def _submit(rid, sid, session, text, **kwargs):
        started.append(text)
        stop.set()
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", _submit)
    worker = threading.Thread(target=server._notification_poller_scoped_loop,
                              args=(stop, "owner-sid", owner), daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert started, "the resumed owner's poller must deliver the parked completion"
    assert server._unowned_parked == []


class _Batch:
    def __init__(self, events):
        self.events = events

    def render(self, registry):
        return "batch"

    def display_text(self, registry):
        return "batch"


def test_a_parked_event_past_its_ttl_is_dropped(monkeypatch):
    """A client that never returns must not pin memory: expired entries drop on the next scan."""
    monkeypatch.setattr(server, "_UNOWNED_RETENTION_SECONDS", 0.0)
    assert server._park_unowned_notification(_completion()) is True
    owner = _session(OWNER_KEY)
    monkeypatch.setattr(server, "_sessions", {"owner-sid": owner})

    assert server._claim_parked_notifications("owner-sid", owner) == []
    assert server._unowned_parked == []


def test_a_full_park_list_refuses_new_entries(monkeypatch):
    """The park is capped; past the cap the caller drops, exactly as it did before the park existed."""
    monkeypatch.setattr(server, "_UNOWNED_RETAINED_MAX", 1)
    kept = _completion("proc_kept")
    assert server._park_unowned_notification(kept) is True
    assert server._park_unowned_notification(_completion("proc_refused")) is False
    assert [parked for _deadline, parked in server._unowned_parked] == [kept]
