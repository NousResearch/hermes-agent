"""``groups.changed``: the change watcher's push for an open Group Chat.

The room view re-reads its log delta and ``groups.state`` on this signal and keeps a slow poll as the
backstop, so the signal must move whenever either answer would: a log append from any process (a cron post,
the CLI, another gateway), a task transition with no append (queued -> running), and status only this process
holds (a pending approval, the tool a member is in). It must not move on a store write that changes neither
(a lease renewal), and it names the rooms that moved so a view of another room does not re-read.
Real temp stores, real ``hosted_rooms`` / ``hosted_room_driver`` writes; only the broadcast is captured.
"""

import sqlite3
import threading
from contextlib import closing
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms as rooms
from tui_gateway import hosted_room_member_activity as activity
from tui_gateway import hosted_room_service
from tui_gateway import server


class _Service:
    """A room service as the watcher sees one: its store, its live status, its wake."""

    def __init__(self, db_path, live=None):
        self.db_path, self.live, self.wakes = db_path, live or {}, 0

    def live_status_sig(self):
        return {"": (True,), **self.live}

    def wakeup(self):
        self.wakes += 1


@pytest.fixture()
def room_watch(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("display: {}\n")
    shared = tmp_path / "shared-state.db"
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_cfg_cache", None)
    for name, value in (("_change_sigs", {}), ("_change_checked_at", {}), ("_change_broadcast_at", {}),
                        ("_room_store_sig_cache", {}), ("_groups_woken_heads", {})):
        monkeypatch.setattr(server, name, value)
    monkeypatch.setattr(rooms, "default_db_path", lambda: shared)
    # Every other watch is someone else's contract; keep the captured stream to this one.
    monkeypatch.setattr(server, "_CHANGE_WATCHES", {"groups.changed": server._CHANGE_WATCHES["groups.changed"]})
    services = []
    monkeypatch.setattr(hosted_room_service, "live_room_services", lambda: list(services))
    events = []
    monkeypatch.setattr(server, "_broadcast_global_event", lambda ev, payload=None: events.append((ev, payload)))
    for room_id in ("room-1", "room-2"):
        rooms.create_room(shared, room_id=room_id, name="Room", members=[{"profile": "ops", "handle": "ops"}],
                          authority_gateway_id="gateway-a", now=10)
    clock = iter(range(0, 10_000, 10))

    def tick():
        """One watcher pass (broadcast + driver wake), far enough apart that no interval or floor applies."""
        server._broadcast_watched_changes(now=float(next(clock)))
        server._wake_room_drivers()
        return [{k: v for k, v in payload.items() if k != "broadcast_id"} for ev, payload in events if ev == "groups.changed"]

    tick()  # first sighting seeds silently
    return shared, tick, services, events


def _post(db, n, room_id="room-1"):
    return rooms.append_event(
        db, room_id=room_id, event_id=f"post-{room_id}-{n}", kind="message.user", actor={"kind": "user", "id": "u"},
        payload={"text": f"post {n}", "thread_id": "t"}, authority_gateway_id="gateway-a", authority_epoch=1)


def test_an_append_from_another_process_names_its_room_once(room_watch):
    db, tick, _services, _events = room_watch
    assert tick() == []
    _post(db, 1)  # a cron job's post: written straight to the store, no RPC on this process
    assert tick() == [{"room_ids": ["room-1"]}]
    assert tick() == [{"room_ids": ["room-1"]}], "nothing moved since: no second broadcast"


def test_a_task_transition_with_no_append_broadcasts_but_a_lease_does_not(room_watch):
    """queued -> running changes ``groups.state`` (member_activity) but appends nothing to the log."""
    db, tick, _services, _events = room_watch
    seq = _post(db, 1)["seq"]
    task = driver.admit_task(
        db, driver.TaskIdentity(room_id="room-1", task_id="t1", thread_id="t", turn_id="t1"),
        payload={"target_profile": "ops", "prompt": "go", "source_event_seq": seq}, clock=lambda: 20.0)
    before = len(tick())
    lease = driver.acquire_lease(db, room_id="room-1", gateway_id="gateway-a", authority_epoch=1,
                                 process_generation="p1", ttl_seconds=30, clock=lambda: 30.0)
    lease = driver.renew_lease(db, lease, ttl_seconds=30, clock=lambda: 31.0)
    assert len(tick()) == before, "taking or renewing a lease changes nothing a room view shows"
    attempt = driver.start_task(db, task["identity"], lease, expected_cancel_generation=0, clock=lambda: 32.0)
    assert tick()[before:] == [{"room_ids": ["room-1"]}]
    driver.requeue_not_admitted_task(db, attempt, clock=lambda: 33.0)
    assert tick()[before + 1:] == [{"room_ids": ["room-1"]}], "running -> queued, nothing appended"


def test_process_local_status_broadcasts_without_any_store_write(room_watch):
    """An approval prompt or a member entering a tool writes nothing to the store."""
    db, tick, services, _events = room_watch
    service = _Service(db)
    services.append(service)
    assert tick() == [{}], "a worker appearing is process-wide: every view re-reads"
    service.live = {"room-2": (True, ("action", "ops", "approve"))}
    assert tick()[1:] == [{"room_ids": ["room-2"]}]
    service.live = {}
    assert tick()[2:] == [{"room_ids": ["room-2"]}], "the prompt was answered"


def test_a_burst_inside_the_floor_is_held_and_then_names_every_room(room_watch):
    db, _tick, _services, events = room_watch
    events.clear()
    server._broadcast_watched_changes(now=100.0)
    _post(db, 1, "room-1")
    server._broadcast_watched_changes(now=100.5)
    _post(db, 1, "room-2")
    server._broadcast_watched_changes(now=101.0)  # inside the 1 s floor of the last broadcast: held
    assert len(events) == 1
    server._broadcast_watched_changes(now=101.6)
    assert [payload["room_ids"] for _ev, payload in events] == [["room-1"], ["room-2"]]
    ids = [payload["broadcast_id"] for _ev, payload in events]
    assert len(set(ids)) == 2 and all(ids), "each broadcast has its own id"


def test_only_the_store_whose_log_grew_wakes_its_driver(room_watch, tmp_path):
    """A cron post must reach the driver now, not at its 5 s idle poll; a status move or another store's
    append must not wake it (each wake re-schedules every busy room)."""
    db, tick, services, _events = room_watch
    owner, other = _Service(db), _Service(tmp_path / "home" / "state.db")
    services += [owner, other]
    tick()
    _post(db, 1)
    tick()
    assert (owner.wakes, other.wakes) == (1, 0)
    owner.live = {"room-1": (True, ("tool", "t1", "terminal"))}
    tick()
    assert owner.wakes == 1, "status-only moves do not wake the driver"


def test_room_status_signature_reads_the_live_service(tmp_path):
    """The real service registers itself and signs what only it holds: approvals, and a preparation's stall
    and last error."""
    db = tmp_path / "state.db"
    rooms.create_room(db, room_id="room-1", name="Room", members=[{"profile": "ops", "handle": "ops"}],
                      authority_gateway_id="gateway-a", now=10)
    service = hosted_room_service.HostedRoomService(
        SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock()), db_path=db)
    assert service in hosted_room_service.live_room_services()
    quiet = service.live_status_sig()
    service._set_pending_action("room-1", "ops", {"kind": "approval", "task_id": "t1", "request_id": "r1"})
    asked = service.live_status_sig()
    assert asked["room-1"] != quiet.get("room-1")
    service._prepare_stalls.failed("room-1", RuntimeError("member profile deleted"))
    erred = service.live_status_sig()
    assert erred != asked, "the room's preparation failed"
    service._prepare_stalls.failed("room-1", RuntimeError("member home unreachable"))
    assert service.live_status_sig() != erred, "the room's last error changed"
    erred = service.live_status_sig()
    service._prepare_stalls.failed("room-1", RuntimeError("member home unreachable"))
    assert service.live_status_sig()["room-1"] != erred["room-1"], "the room's Retry row appeared"


def test_member_progress_is_signed_once_for_the_process(room_watch):
    """Member progress is one process-wide map that every room service's status reads, so the watcher signs it
    once: the tool a member is in, a recovered attempt the member confirms running (#149655's
    ``task_is_active``), and progress to 30 s (a frame alone moves nothing)."""
    _db, tick, _services, _events = room_watch
    info = {"task_id": "t9", "execution_generation": 1, "progress_at": 60.0}
    seen = len(tick())
    try:
        activity.record_task_progress("room-1", "t9", 1, {**info, "tool": "terminal"})
        assert tick()[seen:] == [{"room_ids": ["room-1"]}], "a member entered a tool"
        activity.record_task_progress("room-1", "t9", 1, {**info, "tool": "terminal", "progress_at": 61.0})
        assert tick()[seen + 1:] == [], "a progress tick alone is not a change"
        activity.record_task_progress("room-1", "t9", 1, {**info, "tool": "terminal", "progress_at": 95.0})
        assert tick()[seen + 1:] == [{"room_ids": ["room-1"]}], "progress 30 s later moves the quiet timer"
        activity.record_task_progress("room-1", "t9", 1, {**info, "progress_at": 95.0, "tool": "terminal", "active": True})
        assert activity.task_is_active("room-1", "t9", 1)
        assert tick()[seen + 2:] == [{"room_ids": ["room-1"]}], "a recovered attempt confirmed running"
        activity.record_task_progress("room-1", "t9", 1, {**info, "progress_at": 95.0, "tool": "terminal", "active": False})
        assert tick()[seen + 3:] == [{"room_ids": ["room-1"]}], "and no longer confirmed"
        activity.record_task_progress("room-1", "t9", 1, {**info, "progress_at": 95.0, "tool": "browser", "active": False})
        assert tick()[seen + 4:] == [{"room_ids": ["room-1"]}], "the member moved to another tool"
        activity.record_task_progress("room-1", "t9", 1, {**info, "progress_at": float("inf"), "tool": "browser"})
        assert activity.task_progress("room-1", "t9", 1)["progress_at"] == 95.0, "a non-finite clock is not adopted"
        assert tick()[seen + 5:] == [], "and the watch pass still finishes"
    finally:
        activity.forget_task_progress("room-1", "t9", 1)


def test_a_failing_driver_wake_does_not_end_the_watcher_thread(monkeypatch):
    """Every push (skin, sessions, groups, ...) runs on one daemon thread; a wake that raises (a service
    registry changing size mid-copy) must cost one wake, not the thread."""
    passes = []

    def boom():
        passes.append(1)
        if len(passes) >= 3:
            raise SystemExit  # end the test's loop only after two raising wakes were survived
        raise RuntimeError("Set changed size during iteration")

    started = []
    monkeypatch.setattr(server, "_skin_watcher_started", False)
    monkeypatch.setattr(server, "_note_skin_broadcast", lambda: None)
    monkeypatch.setattr(server, "_broadcast_skin_if_changed", lambda: None)
    monkeypatch.setattr(server, "_broadcast_watched_changes", lambda: None)
    monkeypatch.setattr(server, "_wake_room_drivers", boom)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)
    monkeypatch.setattr(server.threading, "Thread", lambda target, **_k: SimpleNamespace(
        start=lambda: started.append(target)))
    server._ensure_skin_watcher()
    with pytest.raises(SystemExit):
        started[0]()
    assert len(passes) == 3


def test_live_status_never_waits_for_the_policy_lock(tmp_path):
    """prepare_room/stop_room hold _policy_lock across store writes (10 s busy timeout) and peer calls; the
    watcher thread every push shares must read live status without waiting for it."""
    db = tmp_path / "state.db"
    rooms.create_room(db, room_id="room-1", name="Room", members=[{"profile": "ops", "handle": "ops"}],
                      authority_gateway_id="gateway-a", now=10)
    service = hosted_room_service.HostedRoomService(
        SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock()), db_path=db)
    service._set_pending_action("room-1", "ops", {"kind": "approval", "task_id": "t1", "request_id": "r1"})
    held, done, result = threading.Event(), threading.Event(), {}

    def hold():
        with service._policy_lock:
            held.set()
            done.wait(5)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    held.wait(5)
    reader = threading.Thread(target=lambda: result.setdefault("sig", service.live_status_sig()), daemon=True)
    reader.start()
    reader.join(2.0)  # tests/AGENTS.md: wall-clock bounds >= 2 s; the holder waits 5 s
    waited = reader.is_alive()  # read while the lock is still held
    done.set()
    holder.join(5)
    reader.join(5)
    assert not waited, "live_status_sig waited on _policy_lock"
    assert any(row[0] == "action" for row in result["sig"]["room-1"])


def test_status_reads_every_row_it_splits_into_activity_and_retry():
    """status() reads only the driver's shown rows, then splits them by these sets: a status missing from the
    driver's set would drop out of member activity and Retry with no error."""
    shown = set(driver.SHOWN_TASK_STATUSES)
    assert {*hosted_room_service._LIVE_STATUSES, *hosted_room_service._RETRYABLE_STATUSES} == shown
    assert set(hosted_room_service._STOPPABLE_STATUSES) == shown


def test_a_room_service_lists_approval_cards_before_retry(tmp_path):
    """groups.state's action order is the Desktop's card order. Both room kinds list the stall and approval
    cards first and the per-attempt Retry after them (a gateway-owned room always did; reading the rows in
    one snapshot does not change it)."""
    db = tmp_path / "state.db"
    rooms.create_room(db, room_id="room-1", name="Room", members=[{"profile": "ops", "handle": "ops"}],
                      authority_gateway_id="gateway-a", now=10)
    identity = driver.TaskIdentity("room-1", "task-1", "thread", "turn")
    payload = {"target_profile": "ops", "target_member_id": "ops", "source_event_seq": 1, "prompt": "p"}
    driver.admit_task(db, identity, payload=payload, clock=lambda: 10)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("UPDATE hosted_room_driver_tasks SET status='deferred' WHERE task_id='task-1'")
    service = hosted_room_service.HostedRoomService(
        SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock()), db_path=db)
    with closing(sqlite3.connect(db)) as conn:
        generation, = conn.execute(
            "SELECT execution_generation FROM hosted_room_driver_tasks WHERE task_id='task-1'").fetchone()
    # The approval names that attempt: status() offers only an approval for an attempt still approvable.
    service._set_pending_action("room-1", "ops", {"kind": "approval", "task_id": "task-1",
                                                  "execution_generation": generation, "request_id": "r9"})
    assert [action["kind"] for action in service.status("room-1")["pending_actions"]] == ["approval", "retry"]


def test_room_change_rows_read_live_rows_through_the_status_index_and_skip_disbanded_rooms(tmp_path):
    """The watcher re-reads a store on every commit to it (a streaming turn's state.db: about twice a second),
    so each room's read must touch only its live rows: never the retained terminal tasks (up to 2048 a room)
    or a disbanded room's tombstone (up to 512 a store)."""
    db = tmp_path / "state.db"
    for room_id in ("live", "gone"):
        rooms.create_room(db, room_id=room_id, name="Room", members=[{"profile": "ops", "handle": "ops"}],
                          authority_gateway_id="gateway-a", now=10)
    driver.admit_task(db, driver.TaskIdentity("live", "t1", "thread", "turn"),
                      payload={"target_profile": "ops", "prompt": "p", "source_event_seq": 1}, clock=lambda: 10)
    rooms.disband_room(db, room_id="gone", expected_gateway_id="gateway-a", expected_epoch=1, now=20)
    queries = []
    with closing(sqlite3.connect(db)) as conn:
        conn.set_trace_callback(queries.append)
        rows = driver.room_change_rows(conn)
        conn.set_trace_callback(None)
        statuses = driver.SHOWN_TASK_STATUSES
        task_reads = [sql for sql in queries if "FROM hosted_room_driver_tasks" in sql]
        plans = [row[-1] for sql in task_reads for row in conn.execute("EXPLAIN QUERY PLAN " + sql)]
    assert set(rows) == {"live"}
    assert rows["live"].tasks == (("t1", "queued", 0, 0),)
    assert len(task_reads) == 1 and len(statuses) > 1
    assert plans and all("idx_hosted_room_driver_tasks_status" in plan for plan in plans), plans


def test_a_failed_read_keeps_a_good_answer_but_not_a_fallback(tmp_path, monkeypatch):
    """The shared DB-signature cache: a busy read after a good answer keeps it (answer -> fallback -> answer
    would broadcast twice), but after a fallback a newer fallback replaces it, so an unreadable store that keeps
    moving still moves its signature (main's sessions digest rule)."""
    db = tmp_path / "state.db"
    db.write_bytes(b"")
    cache, mtimes, answers = {}, iter((1, 2, 3, 4)), iter(("good", RuntimeError("busy")))
    monkeypatch.setattr(server, "_newest_mtime_ns", lambda _paths: next(mtimes))
    monkeypatch.setattr(server, "_settled_mtime", lambda mtime: mtime)

    def read(_conn, _mtime):
        answer = next(answers, RuntimeError("busy"))
        if isinstance(answer, Exception):
            raise answer
        return answer, False

    def sig():
        return server._db_content_sig(db, cache, read, fallback=lambda mtime: ("fallback", mtime))

    assert sig() == "good"
    assert sig() == "good", "a busy read after a good answer keeps it"
    cache[str(db)] = (2, ("fallback", 2), True)
    assert sig() == ("fallback", 3)
    assert sig() == ("fallback", 4), "after a fallback, a newer one replaces it"


def test_finished_attempts_fencing_entries_move_no_room(room_watch):
    """Each finished attempt leaves an empty progress entry that only fences a late write; the 512-entry LRU
    evicting it changes nothing a room shows, so it must not name that room."""
    _db, tick, _services, _events = room_watch
    for n in range(3):
        activity.forget_task_progress("room-2", f"done-{n}", 1)
    seen = len(tick())
    try:
        activity.forget_task_progress("room-2", "done-0", 2)  # an entry replaced, as an eviction would
        activity.forget_task_progress("room-2", "done-9", 1)
        assert tick()[seen:] == []
        activity.record_task_progress("room-2", "live-1", 1, {"task_id": "live-1", "execution_generation": 1, "active": True})
        assert tick()[seen:] == [{"room_ids": ["room-2"]}], "a live confirmation before any frame is not a fencing entry"
    finally:
        for n in (0, 1, 2, 9):
            activity.forget_task_progress("room-2", f"done-{n}", 5)
        activity.forget_task_progress("room-2", "live-1", 5)


def test_a_store_only_a_room_service_names_is_watched(room_watch, tmp_path):
    """A multiplexing gateway's secondary authority keeps its rooms in its own home's state.db, which is
    neither the watcher home's store nor the shared one: the service registry is how the watcher finds it."""
    _shared, tick, services, _events = room_watch
    store = tmp_path / "secondary" / "state.db"
    store.parent.mkdir()
    rooms.create_room(store, room_id="room-9", name="Room", members=[{"profile": "ops", "handle": "ops"}],
                      authority_gateway_id="gateway-a", now=10)
    service = _Service(store)
    services.append(service)
    seen = len(tick())
    _post(store, 1, "room-9")
    assert tick()[seen:] == [{"room_ids": ["room-9"]}]
    assert service.wakes == 1


def test_a_room_service_starting_or_stopping_moves_its_live_signature(tmp_path):
    """groups.state drops driver_status while the room service is not running (session_group_controls._group),
    and no store write marks that: the worker's run state is the part of the signature that moves."""
    service = hosted_room_service.HostedRoomService(
        SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock()), db_path=tmp_path / "state.db")
    idle = service.live_status_sig()
    service.start()
    try:
        running = service.live_status_sig()
    finally:
        assert service.stop(timeout=5)
    assert running[""] != idle[""]
    assert service.live_status_sig() == idle
