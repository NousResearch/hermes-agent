"""Accepted FIFO sends survive exhausted turn capacity, then dispatch once in order.

Uses the real registry, durable queued rows, queue drain, and inline/compute-host
admission. Only the model-worker / compute-child process boundaries are held.
"""

import copy
import threading
import types

import pytest

from hermes_cli import active_sessions
from hermes_state import SessionDB
from tui_gateway import server


def make_poller_tick(monkeypatch):
    """Run the production poller target for one empty-queue tick, without a thread or wait."""
    from tools import async_delegation, process_registry_awaiting
    from tools.process_registry import process_registry
    from tui_gateway import notification_park

    targets = []

    class HeldThread:
        def __init__(self, *, target, args, **kwargs):
            targets.append((target, args))

        def start(self):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(server, "threading", types.SimpleNamespace(**{**vars(threading), "Thread": HeldThread}))
    monkeypatch.setattr(server, "time", types.SimpleNamespace(**{**vars(server.time), "monotonic": lambda: 5.0}))
    monkeypatch.setattr(server, "_notification_pollers", [])
    monkeypatch.setattr(server, "_wire_desktop_sinks", lambda: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_loop_tick", lambda *args: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_heartbeat_tick", lambda *args: None)
    monkeypatch.setattr(server, "_notif_poll_kanban", lambda *args: None)
    monkeypatch.setattr(process_registry, "restore_completions", lambda: None)
    monkeypatch.setattr(async_delegation, "maybe_sweep_orphaned_completions", lambda queue: None)
    monkeypatch.setattr(notification_park, "claim", lambda owns: [])
    monkeypatch.setattr(notification_park, "unhold", lambda parked: None)
    monkeypatch.setattr(process_registry_awaiting, "release", lambda ready: None)

    def tick(sid, session, *, stop=None, before=None):
        if stop is None:
            stop = server._start_notification_poller(sid, session)
            session["_notif_stop"] = stop
        target, args = targets[-1]
        assert args == (stop, sid, session)
        if before is not None:
            before(stop)

        def take_ready(queue, timeout):
            if timeout:
                stop.set()
            return []

        monkeypatch.setattr(process_registry_awaiting, "take_ready", take_ready)
        target(*args)
        stop.clear()
        return stop

    return tick


@pytest.fixture
def poller_tick(monkeypatch):
    return make_poller_tick(monkeypatch)


@pytest.mark.parametrize("execution", ["inline", "compute_host"])
@pytest.mark.parametrize("initial_refusal", ["capacity", "registry_unavailable", "legacy_unavailable"])
def test_accepted_fifo_recovers_after_capacity_refusal(
    monkeypatch, tmp_path, execution, initial_refusal, poller_tick
):
    cfg = {"max_concurrent_sessions": 1}
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(server, "_canonical_ledger_refusal", lambda session: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_transport_is_dead", lambda transport: False)
    attached = []
    monkeypatch.setattr(server, "_attach_session_transport", lambda session, transport: attached.append(transport))
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda session, cfg=None: execution == "compute_host")
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    workers, frames = [], []
    monkeypatch.setattr(server, "_start_session_work", lambda target, **kwargs: workers.append(target) or object())
    supervisor = types.SimpleNamespace(submit_turn=lambda frame, **kwargs: frames.append(frame))
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *args: supervisor)
    created = server.handle_request({
        "id": "create", "method": "session.create", "params": {"source": "desktop"}})
    assert "result" in created, created
    sid = created["result"]["session_id"]
    session = server._sessions[sid]
    session["agent"] = types.SimpleNamespace(clear_interrupt=lambda: None)
    session["history"] = [{"role": "user", "content": "original context"}]
    blocker = None
    try:
        assert server._ensure_active_session_slot(sid, session) is None
        blocker, refusal = active_sessions.try_acquire_active_session(
            session_id="other-running-turn", surface="gateway:telegram", config=cfg)
        assert refusal is None and blocker is not None
        session["running"] = True
        server._ensure_session_db_row(session)
        for number in (1, 2):
            # Media keeps Q1/Q2 distinct under the established coalescing policy.
            session["attached_images"] = [f"q{number}.png"]
            accepted = server._handle_busy_submit(
                f"accept-{number}", sid, session, f"Q{number}", f"socket-{number}", queued=True,
                turn_author={"id": f"author-{number}", "kind": "human"}, display_kind="user")
            assert accepted["result"]["status"] == "queued"
        session["running"] = False
        head, tail = session["queued_prompt"], session["queued_prompts"][0]
        accepted_envelopes = copy.deepcopy([head, tail])
        key = session["session_key"]
        durable_before = db.get_messages_as_conversation(key, include_inactive=True, include_row_ids=True)
        with monkeypatch.context() as initial:
            if initial_refusal != "capacity":
                refusal = (active_sessions.ActiveSessionRefusal(
                    "registry temporarily unavailable", active_sessions.SESSION_COORDINATION_UNAVAILABLE)
                    if initial_refusal == "registry_unavailable" else server._SESSION_OWNERSHIP_UNAVAILABLE)
                initial.setattr(server, "_ensure_active_session_slot", lambda *args: refusal)
            assert server._drain_queued_prompt("settled-request", sid, session) is True
        assert session.get("_queued_prompt_capacity_wait"), "accepted work needs a retry even on registry outage"
        # The bug removed Q1 inline, and recursively removed Q2 on the host path.
        assert session.get("queued_prompt") is head
        assert session.get("queued_prompts") == [tail]
        assert [head, tail] == accepted_envelopes
        assert not session["running"] and not workers and not frames and not attached
        assert db.get_messages_as_conversation(key, include_inactive=True, include_row_ids=True) == durable_before
        # Repeated scheduler ticks under the same refusal neither mutate rows nor
        # churn IDs, enter the executor, or advance the FIFO.
        stop = poller_tick(sid, session)
        poller_tick(sid, session, stop=stop)
        assert session["queued_prompt"] is head
        assert not workers and not frames and not attached
        assert [head, tail] == accepted_envelopes
        assert db.get_messages_as_conversation(key, include_inactive=True, include_row_ids=True) == durable_before
        # An idle-but-capacity-waiting queue is still busy for newly arriving
        # sends, even when the user's configured busy mode normally steers.
        session["attached_images"] = ["q3.png"]
        newest = server.handle_request({"id": "accept-3", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "Q3"}})
        assert newest["result"]["status"] == "queued", newest
        assert session["queued_prompt"] is head
        assert [entry["text"] for entry in session["queued_prompts"]] == ["Q2", "Q3"]
        assert not workers and not frames
        blocker.release()
        blocker = None
        poller_tick(sid, session, stop=stop)
        assert session["running"] and session["queued_prompt"] is tail
        assert len(workers if execution == "inline" else frames) == 1
        assert attached == ["socket-1"]
        poller_tick(sid, session, stop=stop)
        assert len(workers if execution == "inline" else frames) == 1
        if execution == "compute_host":
            assert frames[0]["text"] == "Q1"
            assert frames[0]["attached_images"] == ["q1.png"]
            assert frames[0]["history"] == [{"role": "user", "content": "original context"}]
            assert frames[0]["session_key"] == key
        # Settlement of Q1 drives the existing drain of Q2, not another timer.
        server._notif_release_turn(session)
        assert server._drain_queued_prompt("settled-request", sid, session) is True
        assert session["running"] and session["queued_prompt"]["text"] == "Q3"
        assert len(workers if execution == "inline" else frames) == 2
        assert attached == ["socket-1", "socket-2"]
        if execution == "compute_host":
            assert [frame["text"] for frame in frames] == ["Q1", "Q2"]
            assert frames[1]["attached_images"] == ["q2.png"]
            assert frames[0]["request_id"] != frames[1]["request_id"]
        server._notif_release_turn(session)
        assert server._drain_queued_prompt("settled-request", sid, session) is True
        assert not session.get("queued_prompt")
        assert len(workers if execution == "inline" else frames) == 3
        if execution == "compute_host":
            assert [frame["text"] for frame in frames] == ["Q1", "Q2", "Q3"]
            assert frames[2]["attached_images"] == ["q3.png"]
        server._notif_release_turn(session)
        assert server._drain_queued_prompt("settled-request", sid, session) is False
        assert server._retry_capacity_queued_prompt(sid, session) is False
        assert len(workers if execution == "inline" else frames) == 3
        active_rows = db.get_messages_as_conversation(key, include_row_ids=True)
        if execution == "inline":
            assert [row["content"] for row in active_rows] == ["Q1", "Q2", "Q3"]
            assert len({row["_row_id"] for row in active_rows}) == 3
        else:
            # Compute-host child (held at the process boundary here) owns persistence.
            assert active_rows == []
    finally:
        if blocker is not None:
            blocker.release()
        server._release_active_session_slot(session)
        server._sessions.pop(sid, None)
        db.close()


@pytest.mark.parametrize("invalidated", ["stop", "generation", "close", "finalized"])
def test_capacity_retry_does_not_resurrect_invalidated_work(monkeypatch, invalidated):
    session = {"history_lock": threading.RLock(), "running": False,
               "queued_prompt": {"text": "Q1"}, "_queued_prompt_generation": 3,
               "_queued_prompt_capacity_wait": {"rid": "original", "generation": 3}}
    if invalidated == "stop":
        session["queued_prompt"] = None
        session["_queued_prompt_generation"] = 4
    elif invalidated == "generation":
        session["_queued_prompt_generation"] = 4
    else:
        session["_closing" if invalidated == "close" else "_finalized"] = True
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *args: pytest.fail("invalidated send was dispatched"))
    assert server._retry_capacity_queued_prompt("test", session) is False
    assert "_queued_prompt_capacity_wait" not in session


def test_stale_capacity_marker_does_not_requeue_a_new_idle_send(monkeypatch):
    session = {"history_lock": threading.RLock(), "running": False,
               "queued_prompt": None, "_queued_prompt_generation": 4,
               "_queued_prompt_capacity_wait": {"rid": "stopped", "generation": 3}}
    assert server._handle_busy_submit("new-send", "test", session, "new text", None, queued=True) is None
    assert "_queued_prompt_capacity_wait" not in session


@pytest.mark.parametrize("invalidated", ["stop_event", "poller_replaced", "session_replaced", "stop", "generation", "close", "finalized"])
def test_real_poller_rejects_stale_queue_ownership(monkeypatch, poller_tick, invalidated):
    sid = "queued-stale-owner"
    head = {"text": "Q1"}
    session = {"history_lock": threading.RLock(), "running": False, "queued_prompt": head,
               "_queued_prompt_generation": 3,
               "_queued_prompt_capacity_wait": {"rid": "accepted", "generation": 3}}
    monkeypatch.setitem(server._sessions, sid, session)
    admissions = []
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *args: admissions.append(args) or "blocked")

    invalidations = {
        "stop_event": lambda stop: stop.set(),
        "poller_replaced": lambda stop: session.__setitem__("_notif_stop", threading.Event()),
        "session_replaced": lambda stop: monkeypatch.setitem(server._sessions, sid, {}),
        "stop": lambda stop: session.__setitem__("_turn_cancel_requested", True),
        "generation": lambda stop: session.__setitem__("_queued_prompt_generation", 4),
        "close": lambda stop: session.__setitem__("_closing", True),
        "finalized": lambda stop: session.__setitem__("_finalized", True),
    }

    poller_tick(sid, session, before=invalidations[invalidated])
    assert admissions == []
    assert session["queued_prompt"] is head and not session["running"]


def test_retry_snapshot_cannot_dispatch_a_replacement_generation(monkeypatch):
    from contextlib import contextmanager

    session = {"history_lock": threading.RLock(), "running": False,
               "queued_prompt": {"text": "old"}, "_queued_prompt_generation": 3,
               "_queued_prompt_capacity_wait": {"rid": "old", "generation": 3}}
    replacement = {"text": "new"}

    @contextmanager
    def replace_queue(_session):
        session.update(queued_prompt=replacement, _queued_prompt_generation=4,
                       _queued_prompt_capacity_wait={"rid": "new", "generation": 4})
        yield

    monkeypatch.setattr(server, "_session_profile_runtime_scope", replace_queue)
    admissions = []
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *args: admissions.append(args) or "blocked")
    assert server._retry_capacity_queued_prompt("queued", session) is False
    assert admissions == [] and session["queued_prompt"] is replacement
    assert session["_queued_prompt_capacity_wait"] == {"rid": "new", "generation": 4}


@pytest.mark.parametrize("cancelled", ["stop", "close", "finalized"])
def test_cancel_after_queue_claim_never_restores_cancelled_envelope(monkeypatch, cancelled):
    session = {"history_lock": threading.RLock(), "running": False,
               "queued_prompt": {"text": "cancelled"}, "_queued_prompt_generation": 3}
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *args: None)
    monkeypatch.setattr(server, "_activate_active_session_slot", lambda *args: None)
    releases, dispatched = [], []
    monkeypatch.setattr(server, "_release_hosted_room_turn_slot", lambda s: releases.append(s))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *args, **kwargs: dispatched.append(args))

    def cancel_after_claim(_session):
        assert session["running"] and session["queued_prompt"] is None
        session["_queued_prompt_generation"] = 4
        session["_turn_cancel_requested" if cancelled == "stop" else
                "_closing" if cancelled == "close" else "_finalized"] = True
        return False

    monkeypatch.setattr(server, "_session_uses_compute_host", cancel_after_claim)
    assert server._drain_queued_prompt("old", "queued", session) is True
    assert session["queued_prompt"] is None
    assert not session["running"] and not dispatched and releases == [session]


@pytest.mark.parametrize("invalidated", ["stop_event", "poller_replaced", "session_replaced"])
def test_poller_ownership_is_rechecked_after_capacity_claim(monkeypatch, poller_tick, invalidated):
    sid = "queued-poller-replaced"
    head = {"text": "accepted"}
    stop = threading.Event()
    session = {"history_lock": threading.RLock(), "running": False, "queued_prompt": head,
               "_queued_prompt_generation": 3, "_notif_stop": stop,
               "_queued_prompt_capacity_wait": {"rid": "accepted", "generation": 3}}
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *args: None)
    monkeypatch.setattr(server, "_activate_active_session_slot", lambda *args: None)
    monkeypatch.setattr(server, "_release_hosted_room_turn_slot", lambda *args: None)
    monkeypatch.setattr(server, "_prepare_queued_user_rows_for_turn", lambda *args: None)
    dispatched = []
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *args, **kwargs: dispatched.append(args))

    def replace_owner(_session):
        if invalidated == "stop_event":
            stop.set()
        elif invalidated == "poller_replaced":
            session["_notif_stop"] = threading.Event()
        else:
            monkeypatch.setitem(server._sessions, sid, {})
        return False

    monkeypatch.setattr(server, "_session_uses_compute_host", replace_owner)
    assert server._retry_capacity_queued_prompt(sid, session, stop_event=stop) is True
    assert not dispatched and not session["running"]
    assert session["queued_prompt"] is head
    assert session["_queued_prompt_capacity_wait"]["generation"] == 3
    # A fresh owning poller can still make progress: restoration must keep its retry marker.
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *args: False)
    current_stop = poller_tick(sid, session)
    poller_tick(sid, session, stop=current_stop)
    assert len(dispatched) == 1 and session["running"] and session["queued_prompt"] is None


def test_refused_reanchored_queue_replaces_its_stale_retry_marker(monkeypatch):
    head = {"text": "accepted"}
    session = {"history_lock": threading.RLock(), "running": False, "queued_prompt": head,
               "_queued_prompt_generation": 4,
               "_queued_prompt_capacity_wait": {"rid": "old-generation", "generation": 3}}
    admissions = []
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *args: admissions.append(args) or "blocked")
    assert server._drain_queued_prompt("reanchored", "queued", session) is True
    assert session["_queued_prompt_capacity_wait"] == {"rid": "reanchored", "generation": 4}
    assert server._retry_capacity_queued_prompt("queued", session) is True
    assert len(admissions) == 2 and session["queued_prompt"] is head and not session["running"]


def test_notification_cannot_overtake_a_capacity_waiting_fifo():
    session = {"history_lock": threading.RLock(), "running": False,
               "queued_prompt": {"text": "accepted"}, "_queued_prompt_generation": 3,
               "_queued_prompt_capacity_wait": {"rid": "old", "generation": 3}}
    assert server._notif_claim_turn(session) is False
    assert not session["running"]
