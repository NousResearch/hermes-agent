"""Runtime tests for the hosted-room session adapter."""

from __future__ import annotations

import hashlib
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import psutil

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from gateway import status as gateway_status
from tui_gateway import hosted_room_driver as driver_runtime
from tui_gateway.hosted_room_driver import (
    MAX_TERMINAL_TEXT_BYTES,
    ROOM_SESSION_SOURCE,
    HostedRoomBinding,
    HostedRoomRuntime,
    _owner_process_state,
    room_session_title,
)


ROOM_ID = "room-1"
PROFILE = "ops"
BINDING = HostedRoomBinding(
    room_id=ROOM_ID,
    gateway_id="gateway-a",
    authority_epoch=1,
)


@pytest.mark.parametrize("error_type", [psutil.NoSuchProcess, psutil.ZombieProcess])
def test_owner_process_state_reports_confirmed_terminal_process_as_dead(
    monkeypatch,
    error_type,
):
    def process_probe(pid: int):
        raise error_type(pid)

    monkeypatch.setattr(psutil, "Process", process_probe)
    monkeypatch.setattr(
        gateway_status,
        "get_process_start_time",
        lambda _pid: pytest.fail("terminal process must not need a fingerprint"),
    )

    assert _owner_process_state(4242, 7001) == "dead"


def test_owner_process_state_reports_zombie_status_as_dead(monkeypatch):
    class ZombieProcess:
        def status(self):
            return psutil.STATUS_ZOMBIE

    monkeypatch.setattr(psutil, "Process", lambda _pid: ZombieProcess())
    monkeypatch.setattr(
        gateway_status,
        "get_process_start_time",
        lambda _pid: pytest.fail("zombie process must not need a fingerprint"),
    )

    assert _owner_process_state(4242, 7001) == "dead"


@pytest.mark.parametrize(
    "probe_error",
    [psutil.AccessDenied(4242), RuntimeError("process probe failed")],
)
def test_owner_process_state_keeps_process_probe_errors_unknown(
    monkeypatch,
    probe_error,
):
    def process_probe(_pid: int):
        raise probe_error

    monkeypatch.setattr(psutil, "Process", process_probe)
    monkeypatch.setattr(
        psutil,
        "pid_exists",
        lambda _pid: pytest.fail("pid_exists is not authoritative for owner death"),
    )

    assert _owner_process_state(4242, 7001) == "unknown"


@pytest.mark.parametrize(
    ("current_start", "expected"),
    [(None, "unknown"), (7001, "alive"), (7002, "dead")],
)
def test_owner_process_state_uses_confirmed_start_fingerprint(
    monkeypatch,
    current_start,
    expected,
):
    class RunningProcess:
        def status(self):
            return "running"

    monkeypatch.setattr(psutil, "Process", lambda _pid: RunningProcess())
    monkeypatch.setattr(
        psutil,
        "pid_exists",
        lambda _pid: pytest.fail("pid_exists is not authoritative for owner death"),
    )
    monkeypatch.setattr(
        gateway_status,
        "get_process_start_time",
        lambda _pid: current_start,
    )

    assert _owner_process_state(4242, 7001) == expected


def test_owner_process_state_keeps_fingerprint_errors_unknown(monkeypatch):
    class RunningProcess:
        def status(self):
            return "running"

    def fingerprint_probe(_pid: int):
        raise OSError("fingerprint unavailable")

    monkeypatch.setattr(psutil, "Process", lambda _pid: RunningProcess())
    monkeypatch.setattr(gateway_status, "get_process_start_time", fingerprint_probe)

    assert _owner_process_state(4242, 7001) == "unknown"


class RecordingTurnLocks:
    """Record the profile lock and expose ownership to the fake RPC."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.local = threading.local()

    @contextmanager
    def __call__(self, profile: str):
        self.events.append(("lock-enter", profile))
        self.local.profile = profile
        try:
            yield
        finally:
            self.events.append(("lock-exit", profile))
            self.local.profile = None

    def held_for(self, profile: str) -> bool:
        return getattr(self.local, "profile", None) == profile


class FakeSessionRPC:
    """Normalized in-memory session adapter with no model or network."""

    def __init__(
        self,
        *,
        auto_complete: bool = True,
        required_lock: RecordingTurnLocks | None = None,
    ) -> None:
        self.auto_complete = auto_complete
        self.required_lock = required_lock
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sessions: dict[tuple[str, str], dict[str, Any]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.submitted = threading.Event()
        self.on_interrupt = None
        self.on_info = None
        self.history_failures = 0
        self._next_id = 1
        self._lock = threading.Lock()

    def _assert_lock(self, profile: str) -> None:
        if self.required_lock is not None:
            assert self.required_lock.held_for(profile)

    def add_session(
        self,
        *,
        profile: str = PROFILE,
        title: str = room_session_title(ROOM_ID),
        active: bool = False,
        task_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        with self._lock:
            session_id = f"session-{self._next_id}"
            self._next_id += 1
            session = {"session_id": session_id, "title": title}
            self.sessions[(profile, title)] = session
            self.states[session_id] = {
                "active": active,
                "task_id": task_id,
                "task": None,
                "execution_generation": None,
                "history": list(history or []),
                "on_terminal": None,
                "pending_approval": None,
            }
            return session_id

    def complete(
        self,
        task_id: str,
        *,
        content: str = "Finished once.",
        status: str = "settled",
    ) -> None:
        callback = None
        receipt = None
        with self._lock:
            for session_id, session_state in self.states.items():
                if session_state["task_id"] != task_id:
                    continue
                receipt = {
                    "role": "assistant",
                    "task_id": task_id,
                    "execution_generation": session_state["execution_generation"],
                    "status": status,
                    "message_id": f"reply:{task_id}",
                    "content": content,
                }
                session_state["history"].append(receipt)
                session_state["active"] = False
                callback = session_state.get("on_terminal")
                self.calls.append(("complete", {"session_id": session_id}))
                break
        if receipt is None:
            raise AssertionError(f"no active session for {task_id}")
        if callback is not None:
            callback({
                "status": status,
                "settlement_id": receipt["message_id"],
                "message_id": receipt["message_id"],
                "text": content,
            })

    def resolve_exact(self, *, profile: str, title: str, source: str):
        self._assert_lock(profile)
        params = {"profile": profile, "title": title, "source": source}
        self.calls.append(("resolve_exact", params))
        with self._lock:
            session = self.sessions.get((profile, title))
            return dict(session) if session is not None else None

    def create(self, *, profile: str, title: str, source: str):
        self._assert_lock(profile)
        params = {"profile": profile, "title": title, "source": source}
        self.calls.append(("create", params))
        session_id = self.add_session(profile=profile, title=title)
        return {"session_id": session_id, "title": title}

    def resume(self, *, profile: str, session_id: str, source: str):
        self._assert_lock(profile)
        params = {
            "profile": profile,
            "session_id": session_id,
            "source": source,
        }
        self.calls.append(("resume", params))
        return {"session_id": session_id}

    def submit(
        self,
        *,
        profile: str,
        session_id: str,
        prompt: str,
        source: str,
        task: state.TaskIdentity,
        execution_generation: int,
        on_terminal,
    ):
        self._assert_lock(profile)
        params = {
            "profile": profile,
            "session_id": session_id,
            "prompt": prompt,
            "source": source,
            "task": task,
            "execution_generation": execution_generation,
            "on_terminal": on_terminal,
        }
        self.calls.append(("submit", params))
        with self._lock:
            self.states[session_id]["active"] = True
            self.states[session_id]["task_id"] = task.task_id
            self.states[session_id]["task"] = task
            self.states[session_id]["execution_generation"] = execution_generation
            self.states[session_id]["on_terminal"] = on_terminal
        self.submitted.set()
        if self.auto_complete:
            self.complete(task.task_id)
        return {"accepted": True}

    def history(self, *, profile: str, session_id: str, source: str):
        self._assert_lock(profile)
        params = {
            "profile": profile,
            "session_id": session_id,
            "source": source,
        }
        self.calls.append(("history", params))
        if self.history_failures > 0:
            self.history_failures -= 1
            raise RuntimeError("transient history read failed")
        with self._lock:
            return [dict(message) for message in self.states[session_id]["history"]]

    def info(self, *, profile: str, session_id: str, source: str):
        self._assert_lock(profile)
        params = {
            "profile": profile,
            "session_id": session_id,
            "source": source,
        }
        self.calls.append(("info", params))
        with self._lock:
            session_state = self.states[session_id]
            result = {
                "active": session_state["active"],
                "task_id": session_state["task_id"],
            }
            if session_state.get("pending_approval"):
                result["status"] = "waiting_for_approval"
                result["pending_approval"] = dict(session_state["pending_approval"])
        if self.on_info is not None:
            self.on_info()
        return result

    def interrupt_admitted(
        self,
        *,
        task: state.TaskIdentity,
        execution_generation: int,
        source: str,
    ):
        params = {
            "task": task,
            "execution_generation": execution_generation,
            "source": source,
            "expected_task_id": task.task_id,
        }
        with self._lock:
            matches = [
                (session_id, current)
                for session_id, current in self.states.items()
                if current.get("task") == task
                and current.get("execution_generation") == execution_generation
            ]
            if len(matches) > 1:
                raise RuntimeError("ambiguous admitted hosted room task")
            if not matches:
                self.calls.append(("interrupt_absent", params))
                return {"found": False, "active": False, "interrupted": False}
            session_id, current = matches[0]
            params["session_id"] = session_id
            if not current["active"] or current["task_id"] != task.task_id:
                self.calls.append(("interrupt_skipped", params))
                return {"found": True, "active": False, "interrupted": False}
            current["active"] = False
        self.calls.append(("interrupt", params))
        if self.on_interrupt is not None:
            self.on_interrupt()
        return {"found": True, "active": True, "interrupted": True}


class SelectiveCompletionRPC(FakeSessionRPC):
    """Keep selected local profiles running while peers complete normally."""

    def __init__(self, *, waiting_profiles: set[str]) -> None:
        super().__init__()
        self.waiting_profiles = waiting_profiles
        self._submit_mode_lock = threading.Lock()

    def submit(self, **kwargs):
        with self._submit_mode_lock:
            original = self.auto_complete
            self.auto_complete = kwargs["profile"] not in self.waiting_profiles
            try:
                return super().submit(**kwargs)
            finally:
                self.auto_complete = original


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    hosted_rooms.create_room(
        path,
        room_id=ROOM_ID,
        name="Release room",
        members=[{"profile": PROFILE, "handle": PROFILE}],
        authority_gateway_id=BINDING.gateway_id,
        now=time.time(),
    )
    return path


def _identity(task_id: str = "task-1") -> state.TaskIdentity:
    return state.TaskIdentity(
        room_id=ROOM_ID,
        task_id=task_id,
        thread_id="thread-1",
        turn_id=f"turn-{task_id}",
    )


def _admit(
    db: Path,
    identity: state.TaskIdentity,
    *,
    prompt: str = "Inspect the release candidate.",
) -> None:
    state.admit_task(
        db,
        identity,
        payload={
            "target_profile": PROFILE,
            "prompt": prompt,
            "source_event_seq": 1,
        },
        clock=time.time,
    )


def _runtime(
    db: Path,
    rpc: FakeSessionRPC,
    locks: RecordingTurnLocks | None = None,
    **kwargs,
) -> HostedRoomRuntime:
    return HostedRoomRuntime(
        db_path=db,
        rooms=[BINDING],
        rpc=rpc,
        turn_lock=locks or RecordingTurnLocks(),
        lease_ttl_seconds=kwargs.pop("lease_ttl_seconds", 0.4),
        poll_interval_seconds=kwargs.pop("poll_interval_seconds", 0.01),
        **kwargs,
    )


def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def test_runtime_uses_unique_process_generation(db: Path):
    first = _runtime(db, FakeSessionRPC())
    second = _runtime(db, FakeSessionRPC())

    assert first.process_generation != second.process_generation
    assert len(first.process_generation) == 32


@pytest.mark.parametrize("value", [0, True])
def test_room_concurrency_bound_must_be_a_positive_integer(db: Path, value):
    with pytest.raises(ValueError, match="max_concurrent_rooms"):
        _runtime(db, FakeSessionRPC(), max_concurrent_rooms=value)


def test_waiting_room_does_not_block_an_independent_local_room(tmp_path: Path):
    db = tmp_path / "state.db"
    bindings = [
        HostedRoomBinding("room-waiting", "gateway-a", 1),
        HostedRoomBinding("room-healthy", "gateway-a", 1),
    ]
    identities = [
        state.TaskIdentity("room-waiting", "task-waiting", "thread-a", "turn-a"),
        state.TaskIdentity("room-healthy", "task-healthy", "thread-b", "turn-b"),
    ]
    profiles = ["profile-waiting", "profile-healthy"]
    for binding, identity, profile in zip(bindings, identities, profiles):
        hosted_rooms.create_room(
            db,
            room_id=binding.room_id,
            name=binding.room_id,
            members=[{"profile": profile, "handle": profile}],
            authority_gateway_id=binding.gateway_id,
            now=time.time(),
        )
        state.admit_task(
            db,
            identity,
            payload={
                "target_profile": profile,
                "prompt": f"Run {binding.room_id}.",
                "source_event_seq": 1,
            },
            clock=time.time,
        )

    rpc = SelectiveCompletionRPC(waiting_profiles={"profile-waiting"})
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=bindings,
        rpc=rpc,
        turn_lock=RecordingTurnLocks(),
        lease_ttl_seconds=0.4,
        poll_interval_seconds=0.01,
        max_concurrent_rooms=2,
    )

    runtime.start()
    _wait_for(lambda: state.get_task(db, identities[1])["status"] == "settled")
    _wait_for(lambda: state.get_task(db, identities[0])["status"] == "running")
    assert state.get_task(db, identities[0])["status"] == "running"
    _wait_for(
        lambda: len(runtime.status()["current_tasks"]) == 1,
        timeout=5.0,
    )
    assert len(runtime.status()["current_tasks"]) == 1
    assert runtime.stop(timeout=1.0)


def test_rotated_bounded_scheduler_eventually_runs_later_room(tmp_path: Path):
    db = tmp_path / "state.db"
    bindings = [
        HostedRoomBinding(f"room-{index}", "gateway-a", 1) for index in range(1, 4)
    ]
    for binding in bindings:
        hosted_rooms.create_room(
            db,
            room_id=binding.room_id,
            name=binding.room_id,
            members=[{"profile": PROFILE, "handle": PROFILE}],
            authority_gateway_id=binding.gateway_id,
            now=time.time(),
        )
    identity = state.TaskIdentity(
        "room-3",
        "task-room-3",
        "thread-room-3",
        "turn-room-3",
    )
    state.admit_task(
        db,
        identity,
        payload={
            "target_profile": PROFILE,
            "prompt": "Run the later room.",
            "source_event_seq": 1,
        },
        clock=time.time,
    )
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=bindings,
        rpc=FakeSessionRPC(),
        turn_lock=RecordingTurnLocks(),
        lease_ttl_seconds=0.4,
        poll_interval_seconds=0.01,
        max_concurrent_rooms=2,
    )

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=1.0)


def test_queued_task_routes_profile_and_credentials_without_overrides(db: Path):
    identity = _identity()
    _admit(db, identity, prompt="Use the configured profile credentials.")
    rpc = FakeSessionRPC()
    runtime = _runtime(db, rpc)

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=1.0)

    create = next(params for method, params in rpc.calls if method == "create")
    submit = next(params for method, params in rpc.calls if method == "submit")
    assert create == {
        "profile": PROFILE,
        "title": f"Group: {ROOM_ID}",
        "source": ROOM_SESSION_SOURCE,
    }
    assert submit["profile"] == PROFILE
    assert submit["source"] == ROOM_SESSION_SOURCE
    assert submit["prompt"] == "Use the configured profile credentials."
    assert "model" not in create | submit
    assert "provider" not in create | submit
    assert state.get_task(db, identity)["result"]["text"] == "Finished once."


def test_profile_deleted_after_admission_defers_without_session_fallback(db: Path):
    deleted_profile = "deleted"
    unavailable = state.TaskIdentity(
        room_id=ROOM_ID,
        task_id="task-unavailable",
        thread_id="thread-1",
        turn_id="turn-unavailable",
    )
    healthy = state.TaskIdentity(
        room_id=ROOM_ID,
        task_id="task-healthy",
        thread_id="thread-1",
        turn_id="turn-healthy",
    )
    state.admit_task(
        db,
        unavailable,
        payload={
            "target_profile": deleted_profile,
            "prompt": "Do not fall back to the launch profile.",
            "source_event_seq": 1,
        },
        clock=time.time,
    )
    state.admit_task(
        db,
        healthy,
        payload={
            "target_profile": PROFILE,
            "prompt": "Continue with the healthy member.",
            "source_event_seq": 2,
        },
        clock=time.time,
    )
    available_profiles = {deleted_profile, PROFILE}
    rpc = FakeSessionRPC()
    published: list[dict[str, Any]] = []
    runtime = _runtime(
        db,
        rpc,
        profile_available=lambda profile: profile in available_profiles,
        publish_terminal=lambda _binding, task: published.append(dict(task)),
    )

    available_profiles.remove(deleted_profile)
    runtime._process_room(BINDING)

    deferred = state.get_task(db, unavailable)
    assert deferred["status"] == "deferred"
    assert deferred["result"] == {
        "reason": "member_unavailable",
        "retryable": True,
    }
    assert state.get_task(db, healthy)["status"] == "settled"
    profile_calls = [
        params["profile"]
        for method, params in rpc.calls
        if method in {"resolve_exact", "create", "resume", "submit"}
    ]
    assert profile_calls == [PROFILE, PROFILE, PROFILE]
    assert [task["status"] for task in published] == ["deferred", "settled"]


def test_worker_settles_without_any_client_transport(db: Path):
    identity = _identity()
    _admit(db, identity)
    runtime = _runtime(db, FakeSessionRPC())

    runtime.start()
    _wait_for(
        lambda: (
            state.get_task(db, identity)["status"] == "settled"
            and runtime.status()["cycles"] >= 1
        )
    )

    assert runtime.status()["running"] is True
    assert runtime.status()["cycles"] >= 1
    assert runtime.stop(timeout=1.0)


def test_policy_hooks_prepare_and_publish_terminal_idempotently(db: Path):
    identity = _identity()
    _admit(db, identity)
    prepared = []
    published = []
    runtime = _runtime(
        db,
        FakeSessionRPC(),
        prepare_room=lambda binding: prepared.append(binding.room_id),
        publish_terminal=lambda binding, task: published.append((
            binding.room_id,
            task["identity"].task_id,
            task["status"],
        )),
    )

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=1.0)

    assert prepared
    assert published == [(ROOM_ID, identity.task_id, "settled")]


def test_existing_canonical_session_is_resumed_not_duplicated(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC()
    session_id = rpc.add_session()
    runtime = _runtime(db, rpc)

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=1.0)

    assert not [call for call in rpc.calls if call[0] == "create"]
    resume = next(params for method, params in rpc.calls if method == "resume")
    assert resume == {
        "profile": PROFILE,
        "session_id": session_id,
        "source": ROOM_SESSION_SOURCE,
    }


def test_long_room_id_reuses_bounded_collision_resistant_session_title(
    tmp_path: Path,
):
    assert room_session_title("room-1") == "Group: room-1"
    room_id = "r" * 127 + "a"
    sibling_room_id = "r" * 127 + "b"
    title = room_session_title(room_id)

    assert len(title) == 100
    assert title.endswith(hashlib.sha256(room_id.encode("utf-8")).hexdigest())
    assert title == room_session_title(room_id)
    assert title != room_session_title(sibling_room_id)

    db = tmp_path / "state.db"
    binding = HostedRoomBinding(room_id, "gateway-a", 1)
    hosted_rooms.create_room(
        db,
        room_id=room_id,
        name="Long identity room",
        members=[{"profile": PROFILE, "handle": PROFILE}],
        authority_gateway_id=binding.gateway_id,
        now=time.time(),
    )
    rpc = FakeSessionRPC()
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=[binding],
        rpc=rpc,
        turn_lock=RecordingTurnLocks(),
        lease_ttl_seconds=0.4,
        poll_interval_seconds=0.01,
    )

    first = state.TaskIdentity(room_id, "task-1", "thread-1", "turn-1")
    _admit(db, first, prompt="First turn.")
    runtime._process_room(binding)

    second = state.TaskIdentity(room_id, "task-2", "thread-1", "turn-2")
    _admit(db, second, prompt="Second turn.")
    runtime._process_room(binding)

    creates = [params for method, params in rpc.calls if method == "create"]
    resumes = [params for method, params in rpc.calls if method == "resume"]
    assert [params["title"] for params in creates] == [title]
    assert len(resumes) == 1
    assert len(rpc.sessions) == 1


def test_local_crash_recovery_keeps_ambiguous_history_explicit_without_resume(
    db: Path,
):
    identity = _identity()
    now = [100.0]

    def clock():
        return now[0]

    old_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-process",
        ttl_seconds=0.2,
        clock=clock,
    )
    _admit(db, identity)
    state.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    rpc = FakeSessionRPC(auto_complete=False)
    rpc.add_session(
        task_id=identity.task_id,
        history=[
            {
                "role": "assistant",
                "task_id": identity.task_id,
                "execution_generation": 1,
                "status": "settled",
                "message_id": "reply:recovered",
                "content": "Recovered durable answer.",
            }
        ],
    )
    now[0] = 101.0
    runtime = _runtime(
        db,
        rpc,
        clock=clock,
        indeterminate_defer_seconds=5,
    )

    runtime._process_room(BINDING)

    recovered = state.get_task(db, identity)
    assert recovered["status"] == "indeterminate"
    assert recovered["result"] is None
    assert not [call for call in rpc.calls if call[0] == "history"]
    assert [call for call in rpc.calls if call[0] == "info"]
    assert not [call for call in rpc.calls if call[0] == "resume"]
    assert not [call for call in rpc.calls if call[0] == "submit"]


def test_expired_local_attempt_defers_without_hydrating_or_resubmitting(db: Path):
    identity = _identity()
    now = [100.0]

    def clock():
        return now[0]

    old_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-process",
        ttl_seconds=0.2,
        clock=clock,
    )
    _admit(db, identity)
    state.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    rpc = FakeSessionRPC(auto_complete=False)
    rpc.add_session(
        task_id=identity.task_id,
        history=[
            {
                "role": "assistant",
                "task_id": identity.task_id,
                "execution_generation": 1,
                "status": "settled",
                "message_id": "reply:expired-recovered",
                "content": "Recovered after lease expiry.",
            }
        ],
    )
    now[0] = 101.0
    runtime = _runtime(
        db,
        rpc,
        clock=clock,
        indeterminate_defer_seconds=0.5,
    )

    runtime._process_room(BINDING)
    now[0] = 102.0
    runtime._process_room(BINDING)

    recovered = state.get_task(db, identity)
    assert recovered["status"] == "deferred"
    assert recovered["result"] == {
        "reason": "member_unavailable",
        "retryable": True,
    }
    assert not [call for call in rpc.calls if call[0] == "history"]
    assert not [call for call in rpc.calls if call[0] == "resume"]
    assert not [call for call in rpc.calls if call[0] == "submit"]


def test_oversized_terminal_reply_is_bounded_without_waiting_for_deadline(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc, turn_timeout_seconds=30)

    runtime.start()
    assert rpc.submitted.wait(timeout=2.0)
    rpc.complete(
        identity.task_id,
        content="é" * (MAX_TERMINAL_TEXT_BYTES + 100),
    )
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=1.0)

    result = state.get_task(db, identity)["result"]
    assert result["truncated"] is True
    assert len(result["text"].encode("utf-8")) <= MAX_TERMINAL_TEXT_BYTES
    assert result["text"].endswith("share the full result as a file.]")


def test_turn_deadline_stops_exact_attempt_and_publishes_durable_failure(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    published = []
    runtime = _runtime(
        db,
        rpc,
        active_poll_interval_seconds=0.01,
        turn_timeout_seconds=0.05,
        publish_terminal=lambda _binding, task: published.append(task),
    )

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "failed")
    assert runtime.stop(timeout=1.0)

    failed = state.get_task(db, identity)
    assert failed["result"] == {
        "error": (
            "This Group Chat turn exceeded its configured time limit and was stopped."
        ),
        "reason_code": "turn_deadline_exceeded",
        "timeout_seconds": 0.05,
    }
    assert failed["cancel_id"] == "deadline:1"
    assert [call for call in rpc.calls if call[0] == "interrupt"]
    assert [task["status"] for task in published] == ["failed"]


def test_deadline_releases_worker_capacity_for_later_room(tmp_path: Path):
    db = tmp_path / "state.db"
    bindings = [
        HostedRoomBinding("room-stuck", "gateway-a", 1),
        HostedRoomBinding("room-healthy", "gateway-a", 1),
    ]
    identities = [
        state.TaskIdentity("room-stuck", "task-stuck", "thread-a", "turn-a"),
        state.TaskIdentity("room-healthy", "task-healthy", "thread-b", "turn-b"),
    ]
    for binding, identity in zip(bindings, identities):
        hosted_rooms.create_room(
            db,
            room_id=binding.room_id,
            name=binding.room_id,
            members=[{"profile": PROFILE, "handle": PROFILE}],
            authority_gateway_id=binding.gateway_id,
        )
        state.admit_task(
            db,
            identity,
            payload={
                "target_profile": PROFILE,
                "prompt": f"Run {binding.room_id}.",
                "source_event_seq": 1,
            },
            clock=time.time,
        )

    class FirstRoomStallsRPC(FakeSessionRPC):
        def submit(self, **kwargs):
            result = super().submit(**kwargs)
            if kwargs["task"].room_id == "room-healthy":
                self.complete(kwargs["task"].task_id)
            return result

    rpc = FirstRoomStallsRPC(auto_complete=False)
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=bindings,
        rpc=rpc,
        turn_lock=RecordingTurnLocks(),
        lease_ttl_seconds=0.4,
        poll_interval_seconds=0.02,
        active_poll_interval_seconds=0.01,
        turn_timeout_seconds=0.05,
        max_concurrent_rooms=1,
    )

    runtime.start()
    _wait_for(lambda: state.get_task(db, identities[0])["status"] == "failed")
    _wait_for(lambda: state.get_task(db, identities[1])["status"] == "settled")
    assert runtime.stop(timeout=1.0)

    assert state.get_task(db, identities[0])["result"]["reason_code"] == (
        "turn_deadline_exceeded"
    )
    assert state.get_task(db, identities[1])["status"] == "settled"


def test_retry_ignores_late_receipt_from_prior_execution_generation(db: Path):
    identity = _identity()
    now = [100.0]

    def clock():
        return now[0]

    old_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-process",
        ttl_seconds=0.2,
        clock=clock,
    )
    _admit(db, identity)
    old_attempt = state.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    now[0] = 101.0
    current_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="manual-recovery",
        ttl_seconds=30,
        clock=clock,
    )
    state.recover_room(db, current_lease, clock=clock)
    state.requeue_indeterminate_task(
        db,
        identity,
        current_lease,
        expected_execution_generation=old_attempt.execution_generation,
        expected_cancel_generation=old_attempt.cancel_generation,
        clock=clock,
    )
    state.release_lease(db, current_lease, clock=clock)
    rpc = FakeSessionRPC(auto_complete=False)
    rpc.add_session(
        task_id=identity.task_id,
        history=[
            {
                "role": "assistant",
                "task_id": identity.task_id,
                "execution_generation": old_attempt.execution_generation,
                "status": "settled",
                "message_id": "reply:late-old-attempt",
                "content": "Late old result.",
            }
        ],
    )
    runtime = _runtime(db, rpc, clock=clock)

    runtime.start()
    assert rpc.submitted.wait(1.0)
    time.sleep(0.04)
    assert runtime.stop(timeout=1.0)

    task = state.get_task(db, identity)
    assert task["status"] == "running"
    assert task["execution_generation"] == old_attempt.execution_generation + 1


def test_active_recovered_turn_is_never_resubmitted(db: Path):
    identity = _identity()
    old_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-process",
        ttl_seconds=10,
        clock=time.time,
    )
    _admit(db, identity)
    state.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=time.time,
    )
    rpc = FakeSessionRPC(auto_complete=False)
    rpc.add_session(active=True, task_id=identity.task_id)
    runtime = _runtime(db, rpc)

    runtime.start()
    time.sleep(0.08)
    assert runtime.stop(timeout=1.0)

    assert state.get_task(db, identity)["status"] == "running"
    assert not [call for call in rpc.calls if call[0] == "submit"]


def test_ambiguous_recovery_remains_indeterminate(db: Path):
    identity = _identity()
    now = [100.0]

    def clock():
        return now[0]

    old_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-process",
        ttl_seconds=0.2,
        clock=clock,
    )
    _admit(db, identity)
    state.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    rpc = FakeSessionRPC(auto_complete=False)
    rpc.add_session(active=False, task_id=identity.task_id)
    now[0] = 101.0
    runtime = _runtime(db, rpc, clock=clock)

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "indeterminate")
    assert runtime.stop(timeout=1.0)

    assert not [call for call in rpc.calls if call[0] == "submit"]


def test_restart_harvests_private_durable_terminal_receipt(db: Path):
    identity = _identity()
    now = [100.0]

    def clock():
        return now[0]

    old_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-process",
        ttl_seconds=0.2,
        clock=clock,
    )
    _admit(db, identity)
    attempt = state.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    state.record_terminal_receipt(
        db,
        identity,
        execution_generation=attempt.execution_generation,
        settlement_id="reply-task-1-1",
        status="settled",
        result={"message_id": "reply-task-1-1", "text": "durable answer"},
        clock=clock,
    )
    rpc = FakeSessionRPC(auto_complete=False)
    rpc.add_session(active=False, task_id=identity.task_id)
    now[0] = 101.0
    published = []
    runtime = _runtime(
        db,
        rpc,
        clock=clock,
        publish_terminal=lambda _binding, task: published.append(task),
    )

    runtime._process_room(BINDING)

    settled = state.get_task(db, identity)
    assert settled["status"] == "settled"
    assert settled["result"]["text"] == "durable answer"
    assert [task["status"] for task in published] == ["settled"]
    assert not [call for call in rpc.calls if call[0] == "submit"]


def test_offline_member_defers_then_healthy_task_runs_and_retry_is_fenced(
    db: Path,
):
    now = [100.0]

    def clock():
        return now[0]

    first = _identity("task-offline")
    second = state.TaskIdentity(
        room_id=ROOM_ID,
        task_id="task-healthy",
        thread_id="thread-1",
        turn_id="turn-task-healthy",
    )
    state.admit_task(
        db,
        first,
        payload={
            "target_profile": PROFILE,
            "prompt": "Try the offline member.",
            "source_event_seq": 1,
        },
        clock=clock,
    )
    old_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-process",
        ttl_seconds=1,
        clock=clock,
    )
    old_attempt = state.start_task(
        db,
        first,
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    state.admit_task(
        db,
        second,
        payload={
            "target_profile": PROFILE,
            "prompt": "Continue with the healthy member.",
            "source_event_seq": 2,
        },
        clock=clock,
    )
    now[0] = 102.0
    rpc = FakeSessionRPC()
    published = []
    runtime = _runtime(
        db,
        rpc,
        clock=clock,
        lease_ttl_seconds=30,
        indeterminate_defer_seconds=5,
        publish_terminal=lambda _binding, task: published.append(task),
    )

    runtime._process_room(BINDING)
    assert state.get_task(db, first)["status"] == "indeterminate"
    assert state.get_task(db, second)["status"] == "queued"

    now[0] = 108.0
    runtime._process_room(BINDING)
    assert state.get_task(db, first)["status"] == "deferred"
    assert state.get_task(db, second)["status"] == "settled"
    assert [task["status"] for task in published] == ["deferred", "settled"]
    assert ROOM_ID not in runtime.status()["blocked_rooms"]

    requeued = runtime.retry_indeterminate(first)
    assert requeued["status"] == "queued"
    lease = runtime._leases[ROOM_ID]
    retry_attempt = state.start_task(
        db,
        first,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    assert retry_attempt.execution_generation == old_attempt.execution_generation + 1
    late_attempt = state.TaskAttempt(
        identity=first,
        lease=lease,
        execution_generation=old_attempt.execution_generation,
        cancel_generation=old_attempt.cancel_generation,
    )
    with pytest.raises(state.StaleTaskError):
        state.settle_task(
            db,
            late_attempt,
            settlement_id="late-old-result",
            status="settled",
            result={"text": "too late"},
            clock=clock,
        )
    state.settle_task(
        db,
        retry_attempt,
        settlement_id="retry-result",
        status="settled",
        result={"text": "retry accepted"},
        clock=clock,
    )
    assert state.get_task(db, first)["result"]["text"] == "retry accepted"


def test_post_submit_observation_failure_preserves_recoverable_outcome(
    db: Path,
    monkeypatch,
):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    original_receipt = state.get_terminal_receipt
    failures = 1

    def fail_receipt_once(*args, **kwargs):
        nonlocal failures
        if failures:
            failures -= 1
            raise RuntimeError("transient durable receipt read failed")
        return original_receipt(*args, **kwargs)

    monkeypatch.setattr(state, "get_terminal_receipt", fail_receipt_once)
    runtime = _runtime(db, rpc)

    runtime.start()
    assert rpc.submitted.wait(1.0)
    _wait_for(
        lambda: (
            "observation failed after submit"
            in str(runtime.status()["last_error"] or "")
        )
    )
    assert state.get_task(db, identity)["status"] == "running"
    rpc.complete(identity.task_id, content="Recovered after a transient read.")
    runtime.wakeup()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=1.0)

    task = state.get_task(db, identity)
    assert task["result"]["text"] == "Recovered after a transient read."
    assert not [call for call in rpc.calls if call[0] == "submit"][1:]


def test_proven_pre_admission_submit_rejection_fails_without_ambiguity_delay(
    db: Path,
):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)

    def reject_before_admission(**_kwargs):
        exc = RuntimeError("hosted room session is busy")
        exc.not_admitted = True
        raise exc

    rpc.submit = reject_before_admission  # type: ignore[method-assign]
    runtime = _runtime(db, rpc)

    runtime._process_room(BINDING)

    task = state.get_task(db, identity)
    assert task["status"] == "failed"
    assert task["result"] == {"error": "hosted room session is busy"}
    assert ROOM_ID not in runtime.status()["blocked_rooms"]


def test_non_owner_runtime_leaves_stop_for_task_process_owner(db: Path):
    identity = _identity()
    _admit(db, identity)
    owner_rpc = FakeSessionRPC(auto_complete=False)
    owner_observed_active = threading.Event()
    owner_interrupted = threading.Event()
    owner_rpc.on_info = owner_observed_active.set
    owner_rpc.on_interrupt = owner_interrupted.set
    owner = _runtime(
        db,
        owner_rpc,
        process_generation="owner-process",
        active_poll_interval_seconds=10.0,
    )
    non_owner_rpc = FakeSessionRPC(auto_complete=False)
    non_owner_rpc.add_session(active=False, task_id=identity.task_id)
    non_owner = _runtime(
        db,
        non_owner_rpc,
        process_generation="non-owner-process",
    )

    owner.start()
    try:
        assert owner_rpc.submitted.wait(1.0)
        assert owner_observed_active.wait(1.0)
        running = state.get_task(db, identity)
        assert running["run_process_generation"] == owner.process_generation

        stopping = non_owner.cancel(identity, cancel_id="cancel-from-peer")

        assert stopping["status"] == "stopping"
        assert state.get_task(db, identity)["status"] == "stopping"
        assert non_owner_rpc.calls == []

        owner.wakeup()
        assert owner_interrupted.wait(5.0)
        _wait_for(
            lambda: state.get_task(db, identity)["status"] == "cancelled",
            timeout=5.0,
        )
        assert len([call for call in owner_rpc.calls if call[0] == "interrupt"]) == 1
    finally:
        assert owner.stop(timeout=1.0)


def test_cancellation_is_persisted_before_interrupt_and_fences_late_result(
    db: Path,
):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc)
    observed_status: list[str] = []
    interrupt_seen = threading.Event()

    def observe_interrupt() -> None:
        observed_status.append(state.get_task(db, identity)["status"])
        interrupt_seen.set()

    rpc.on_interrupt = observe_interrupt

    runtime.start()
    assert rpc.submitted.wait(1.0)
    cancelled = runtime.cancel(identity, cancel_id="cancel-user")
    if cancelled["status"] == "stopping":
        runtime.wakeup()
        assert interrupt_seen.wait(5.0)
        _wait_for(
            lambda: state.get_task(db, identity)["status"] == "cancelled",
            timeout=5.0,
        )
        cancelled = state.get_task(db, identity)
    rpc.complete(identity.task_id, content="Too late.")
    runtime.wakeup()
    time.sleep(0.05)
    assert runtime.stop(timeout=1.0)

    assert cancelled["status"] == "cancelled"
    assert observed_status == ["stopping"]


def test_stop_before_prompt_admission_never_starts_the_cancelled_task(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc)
    resolve_started = threading.Event()
    release_resolve = threading.Event()
    original_resolve = rpc.resolve_exact

    def blocked_resolve(**kwargs):
        resolve_started.set()
        assert release_resolve.wait(2.0)
        return original_resolve(**kwargs)

    rpc.resolve_exact = blocked_resolve
    runtime.start()
    try:
        assert resolve_started.wait(1.0)
        stopping = runtime.cancel(identity, cancel_id="cancel-before-admission")
        assert stopping["status"] == "stopping"

        release_resolve.set()
        _wait_for(lambda: state.get_task(db, identity)["status"] == "cancelled")
        assert not [call for call in rpc.calls if call[0] == "submit"]
    finally:
        release_resolve.set()
        assert runtime.stop(timeout=1.0)


def test_ambiguous_interrupt_result_keeps_stop_pending(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc, active_poll_interval_seconds=10.0)

    runtime.start()
    assert rpc.submitted.wait(1.0)
    original_interrupt = rpc.interrupt_admitted
    rpc.interrupt_admitted = lambda **_kwargs: None

    stopping = runtime.cancel(identity, cancel_id="cancel-ambiguous")
    assert stopping["status"] == "stopping"
    assert state.get_task(db, identity)["status"] == "stopping"

    rpc.interrupt_admitted = original_interrupt
    runtime.wakeup()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "cancelled")
    assert runtime.stop(timeout=1.0)


def test_stop_uses_admitted_proof_after_target_profile_is_deleted(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc)

    runtime.start()
    assert rpc.submitted.wait(1.0)
    rpc.resolve_exact = lambda **_kwargs: pytest.fail(
        "Stop must not query a deleted profile or the persisted session index"
    )
    runtime.profile_available = lambda _profile: False

    cancelled = runtime.cancel(identity, cancel_id="cancel-after-profile-delete")

    assert cancelled["status"] == "cancelled"
    assert len([call for call in rpc.calls if call[0] == "interrupt"]) == 1
    assert runtime.stop(timeout=1.0)


def test_transient_remote_stop_failure_stays_pending_and_retries(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc, active_poll_interval_seconds=10.0)
    original_interrupt = rpc.interrupt_admitted
    original_record_error = runtime._record_error
    attempts = 0
    worker_failure_recorded = threading.Event()
    release_worker = threading.Event()

    def flaky_interrupt(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise RuntimeError("temporary stop transport failure")
        return original_interrupt(**kwargs)

    def block_worker_after_retry_failure(message: str) -> None:
        original_record_error(message)
        if message.startswith("stop retry remains pending"):
            worker_failure_recorded.set()
            assert release_worker.wait(2.0)

    rpc.interrupt_admitted = flaky_interrupt
    runtime._record_error = block_worker_after_retry_failure
    runtime.start()
    assert rpc.submitted.wait(1.0)
    stopping = runtime.cancel(identity, cancel_id="cancel-retry")
    assert stopping["status"] == "stopping"
    assert state.get_task(db, identity)["status"] == "stopping"
    assert worker_failure_recorded.wait(1.0)
    with runtime._status_lock:
        room_worker = runtime._room_threads[ROOM_ID]
    release_worker.set()
    runtime.wakeup()
    room_worker.join(1.0)
    assert not room_worker.is_alive()
    cycles = runtime.status()["cycles"]
    runtime.wakeup()
    _wait_for(lambda: runtime.status()["cycles"] > cycles)
    _wait_for(lambda: state.get_task(db, identity)["status"] == "cancelled")
    assert attempts >= 3
    assert runtime.stop(timeout=1.0)
    assert state.get_task(db, identity)["status"] == "cancelled"


def test_completion_wins_a_race_with_unacknowledged_stop(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc)

    runtime.start()
    assert rpc.submitted.wait(1.0)
    original_interrupt = rpc.interrupt_admitted

    def finish_only_after_stop_intent(**kwargs):
        if state.get_task(db, identity)["status"] == "stopping":
            rpc.complete(identity.task_id, content="Already done.")
        return original_interrupt(**kwargs)

    rpc.interrupt_admitted = finish_only_after_stop_intent
    result = runtime.cancel(identity, cancel_id="cancel-raced")

    assert result["status"] == "settled"
    settled = state.get_task(db, identity)
    assert settled["result"]["text"] == "Already done."
    assert runtime.stop(timeout=1.0)


def test_durable_terminal_receipt_wins_before_immediate_stop_interrupt(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc)

    runtime.start()
    assert rpc.submitted.wait(1.0)
    running = state.get_task(db, identity)
    state.record_terminal_receipt(
        db,
        identity,
        execution_generation=running["execution_generation"],
        settlement_id="reply-before-stop",
        status="settled",
        result={"text": "Already complete."},
        clock=time.time,
    )

    result = runtime.cancel(identity, cancel_id="cancel-after-receipt")

    assert result["status"] == "settled"
    assert result["result"]["text"] == "Already complete."
    assert not [call for call in rpc.calls if call[0] == "interrupt"]
    assert runtime.stop(timeout=1.0)


def test_terminal_receipt_committed_during_interrupt_wins_cancel_ack(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc)

    runtime.start()
    assert rpc.submitted.wait(1.0)
    running = state.get_task(db, identity)
    rpc.on_interrupt = lambda: state.record_terminal_receipt(
        db,
        identity,
        execution_generation=running["execution_generation"],
        settlement_id="reply-during-stop",
        status="settled",
        result={"text": "Completed during Stop."},
        clock=time.time,
    )

    result = runtime.cancel(identity, cancel_id="cancel-racing-receipt")

    assert result["status"] == "settled"
    assert result["result"]["text"] == "Completed during Stop."
    assert len([call for call in rpc.calls if call[0] == "interrupt"]) == 1
    assert runtime.stop(timeout=1.0)


def test_completion_wins_stop_race_after_attempt_lease_expires(db: Path):
    identity = _identity()
    _admit(db, identity)
    now = [100.0]
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(
        db,
        rpc,
        clock=lambda: now[0],
        lease_ttl_seconds=0.4,
    )

    runtime.start()
    assert rpc.submitted.wait(1.0)
    original_interrupt = rpc.interrupt_admitted

    def finish_after_stop_intent_and_lease_expiry(**kwargs):
        if state.get_task(db, identity)["status"] == "stopping":
            now[0] = 101.0
            rpc.complete(identity.task_id, content="Already done after expiry.")
        return original_interrupt(**kwargs)

    rpc.interrupt_admitted = finish_after_stop_intent_and_lease_expiry
    result = runtime.cancel(identity, cancel_id="cancel-raced-expired-lease")

    assert result["status"] == "settled"
    assert result["result"]["text"] == "Already done after expiry."
    assert runtime.stop(timeout=1.0)


def test_restart_harvests_completion_before_retrying_durable_stop(db: Path):
    identity = _identity()
    _admit(db, identity)
    now = [100.0]
    old_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-process",
        ttl_seconds=1.0,
        clock=lambda: now[0],
    )
    attempt = state.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=lambda: now[0],
    )
    stopping = state.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-before-restart",
        expected_cancel_generation=attempt.cancel_generation,
        clock=lambda: now[0],
    )
    state.record_terminal_receipt(
        db,
        identity,
        execution_generation=attempt.execution_generation,
        settlement_id="reply-after-stop",
        status="settled",
        result={"text": "Finished before Stop reached the session."},
        clock=lambda: now[0],
    )
    rpc = FakeSessionRPC(auto_complete=False)
    rpc.add_session(
        active=False,
        task_id=identity.task_id,
        history=[],
    )
    rpc.history_failures = 1
    now[0] += 2.0
    runtime = _runtime(
        db,
        rpc,
        process_generation="new-process",
        clock=lambda: now[0],
    )

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=1.0)

    settled = state.get_task(db, identity)
    assert stopping["status"] == "stopping"
    assert settled["result"]["text"] == "Finished before Stop reached the session."
    assert not [call for call in rpc.calls if call[0] == "interrupt"]


def test_restart_leaves_stop_pending_while_previous_process_is_live(db: Path):
    identity = _identity()
    _admit(db, identity)
    now = [100.0]
    clock = lambda: now[0]
    old_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-process",
        process_pid=1111,
        process_start_time=7001,
        ttl_seconds=1.0,
        clock=clock,
    )
    attempt = state.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    state.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-before-restart",
        expected_cancel_generation=attempt.cancel_generation,
        clock=clock,
    )
    rpc = FakeSessionRPC(auto_complete=False)
    now[0] += 2.0
    runtime = _runtime(
        db,
        rpc,
        process_generation="new-process",
        process_pid=2222,
        process_start_time=8001,
        owner_liveness=lambda _pid, _started: "alive",
        clock=clock,
    )

    stopping = runtime.cancel(identity, cancel_id="cancel-before-restart")

    assert stopping["status"] == "stopping"
    assert state.get_task(db, identity)["status"] == "stopping"
    assert rpc.calls == []


def test_restart_reclaims_stop_only_after_previous_process_exit(db: Path):
    identity = _identity()
    _admit(db, identity)
    now = [100.0]
    clock = lambda: now[0]
    old_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-process",
        process_pid=1111,
        process_start_time=7001,
        ttl_seconds=1.0,
        clock=clock,
    )
    attempt = state.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    state.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-after-owner-exit",
        expected_cancel_generation=attempt.cancel_generation,
        clock=clock,
    )
    now[0] += 2.0
    runtime = _runtime(
        db,
        FakeSessionRPC(auto_complete=False),
        process_generation="new-process",
        process_pid=2222,
        process_start_time=8001,
        owner_liveness=lambda pid, started: (
            "dead" if (pid, started) == (1111, 7001) else "unknown"
        ),
        clock=clock,
    )

    cancelled = runtime.cancel(identity, cancel_id="cancel-after-owner-exit")

    assert cancelled["status"] == "cancelled"
    assert cancelled["run_process_generation"] == "new-process"


def test_same_process_runtime_replacement_can_finish_exact_stop(db: Path):
    identity = _identity()
    _admit(db, identity)
    now = [100.0]
    clock = lambda: now[0]
    old_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-runtime",
        process_pid=3333,
        process_start_time=9001,
        ttl_seconds=1.0,
        clock=clock,
    )
    attempt = state.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    state.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-same-process",
        expected_cancel_generation=attempt.cancel_generation,
        clock=clock,
    )
    now[0] += 2.0
    runtime = _runtime(
        db,
        FakeSessionRPC(auto_complete=False),
        process_generation="new-runtime",
        process_pid=3333,
        process_start_time=9001,
        owner_liveness=lambda _pid, _started: pytest.fail(
            "same-process replacement must not require liveness probing"
        ),
        clock=clock,
    )

    assert runtime.cancel(identity, cancel_id="cancel-same-process")["status"] == "cancelled"


def test_same_process_replacement_waits_for_submit_fence_before_cancelling(
    db: Path,
):
    identity = _identity()
    _admit(db, identity)
    now = [100.0]
    clock = lambda: now[0]
    owner_rpc = FakeSessionRPC(auto_complete=False)
    owner = _runtime(
        db,
        owner_rpc,
        process_generation="old-runtime",
        process_pid=3333,
        process_start_time=9001,
        owner_liveness=lambda _pid, _started: pytest.fail(
            "same-process replacement must not require liveness probing"
        ),
        clock=clock,
        lease_ttl_seconds=1.0,
    )
    resolve_started = threading.Event()
    release_resolve = threading.Event()
    original_resolve = owner_rpc.resolve_exact

    def blocked_resolve(**kwargs):
        resolve_started.set()
        assert release_resolve.wait(2.0)
        return original_resolve(**kwargs)

    owner_rpc.resolve_exact = blocked_resolve
    owner.start()
    try:
        assert resolve_started.wait(1.0)
        now[0] += 2.0
        replacement_rpc = FakeSessionRPC(auto_complete=False)
        replacement = _runtime(
            db,
            replacement_rpc,
            process_generation="new-runtime",
            process_pid=3333,
            process_start_time=9001,
            owner_liveness=lambda _pid, _started: pytest.fail(
                "same-process replacement must not require liveness probing"
            ),
            clock=clock,
            lease_ttl_seconds=1.0,
        )

        stopping = replacement.cancel(
            identity,
            cancel_id="cancel-during-submit-window",
        )
        assert stopping["status"] == "stopping"
        assert any(call[0] == "interrupt_absent" for call in replacement_rpc.calls)

        release_resolve.set()
        _wait_for(
            lambda: not driver_runtime._process_submission_is_registered(
                driver_runtime._submission_fence_key(
                    db,
                    identity,
                    int(stopping["execution_generation"]),
                )
            )
        )
        assert not [call for call in owner_rpc.calls if call[0] == "submit"]

        cancelled = replacement.cancel(
            identity,
            cancel_id="cancel-during-submit-window",
        )
        assert cancelled["status"] == "cancelled"
        assert not [call for call in replacement_rpc.calls if call[0] == "submit"]
    finally:
        release_resolve.set()
        assert owner.stop(timeout=1.0)


def test_pending_local_approval_is_reported_with_safe_choices(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    actions = []
    runtime = _runtime(
        db,
        rpc,
        pending_action=lambda room_id, member_id, action: actions.append((
            room_id,
            member_id,
            action,
        )),
    )

    runtime.start()
    assert rpc.submitted.wait(1.0)
    session_id = next(iter(rpc.states))
    with rpc._lock:
        rpc.states[session_id]["pending_approval"] = {
            "request_id": "approval-1",
            "command": "pytest -q tests/focused",
            "choices": ["once", "session", "always", "deny"],
        }
    runtime.wakeup()
    _wait_for(lambda: any(action for _room, _member, action in actions))

    _room, member, action = next(item for item in actions if item[2] is not None)
    assert member == PROFILE
    assert action["request_id"] == "approval-1"
    assert action["approval"]["choices"] == ["once", "deny"]
    assert runtime.stop(timeout=1.0)


def test_cancel_never_interrupts_a_newer_task_in_the_same_session(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc)

    runtime.start()
    assert rpc.submitted.wait(1.0)
    session_id = next(iter(rpc.states))
    original_interrupt = rpc.interrupt_admitted

    def switch_to_newer_task(**kwargs):
        with rpc._lock:
            rpc.states[session_id]["active"] = True
            rpc.states[session_id]["task_id"] = "task-2"
            rpc.states[session_id]["task"] = state.TaskIdentity(
                ROOM_ID, "task-2", "thread-2", "turn-2"
            )
        return original_interrupt(**kwargs)

    rpc.interrupt_admitted = switch_to_newer_task
    cancelled = runtime.cancel(identity, cancel_id="cancel-old-task")

    assert cancelled["status"] == "cancelled"
    assert not [call for call in rpc.calls if call[0] == "interrupt"]
    assert len([call for call in rpc.calls if call[0] == "interrupt_absent"]) == 1
    assert rpc.states[session_id]["active"] is True
    assert rpc.states[session_id]["task_id"] == "task-2"
    assert runtime.stop(timeout=1.0)


def test_status_reports_room_blocked_on_unresolved_indeterminate_task(db: Path):
    identity = _identity()
    now = [100.0]
    old_lease = state.acquire_lease(
        db,
        room_id=ROOM_ID,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-process",
        ttl_seconds=1.0,
        clock=lambda: now[0],
    )
    _admit(db, identity)
    state.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=lambda: now[0],
    )
    rpc = FakeSessionRPC(auto_complete=False)
    rpc.add_session(active=False, task_id=identity.task_id)
    now[0] += 2.0
    runtime = _runtime(db, rpc, clock=lambda: now[0])

    runtime.start()
    _wait_for(lambda: ROOM_ID in runtime.status()["blocked_rooms"])
    assert runtime.stop(timeout=1.0)

    assert state.get_task(db, identity)["status"] == "indeterminate"


def test_authority_loss_stops_terminal_commit(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc, lease_ttl_seconds=2.0)

    runtime.start()
    assert rpc.submitted.wait(1.0)
    hosted_rooms.claim_authority(
        db,
        room_id=ROOM_ID,
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        new_gateway_id="gateway-b",
        event_id="claim-gateway-b",
        now=time.time(),
    )
    rpc.complete(identity.task_id)
    runtime.wakeup()
    _wait_for(lambda: runtime.status()["last_error"] is not None)
    assert runtime.stop(timeout=1.0)

    assert state.get_task(db, identity)["status"] == "running"
    assert "authority changed" in runtime.status()["last_error"]


def test_profile_turn_lock_covers_resolve_submit_and_terminal_observation(db: Path):
    identity = _identity()
    _admit(db, identity)
    locks = RecordingTurnLocks()
    rpc = FakeSessionRPC(required_lock=locks)
    runtime = _runtime(db, rpc, locks)

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=1.0)

    assert locks.events == [("lock-enter", PROFILE), ("lock-exit", PROFILE)]
    methods = [method for method, _params in rpc.calls]
    assert methods.index("resolve_exact") < methods.index("submit")
    assert methods.index("submit") < methods.index("complete")
    assert "history" not in methods


def test_profile_lock_timeout_leaves_unsubmitted_task_queued(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC()

    @contextmanager
    def busy_lock(_profile: str):
        raise RuntimeError("target_busy")
        yield

    runtime = _runtime(db, rpc, locks=busy_lock)

    with pytest.raises(RuntimeError, match="target_busy"):
        runtime._process_room(BINDING)

    task = state.get_task(db, identity)
    assert task["status"] == "queued"
    assert task["execution_generation"] == 0
    assert not [call for call in rpc.calls if call[0] == "submit"]

    runtime.turn_lock = RecordingTurnLocks()
    runtime._process_room(BINDING)
    assert state.get_task(db, identity)["status"] == "settled"


def test_proven_not_admitted_submission_settles_without_ambiguity(db: Path):
    identity = _identity()
    _admit(db, identity)

    class ProvenNotAdmittedError(RuntimeError):
        not_admitted = True

    class RejectingSessionRPC(FakeSessionRPC):
        def submit(self, **kwargs):
            self.calls.append(("submit", dict(kwargs)))
            raise ProvenNotAdmittedError("hosted room member session is busy")

    runtime = _runtime(db, RejectingSessionRPC())

    runtime._process_room(BINDING)

    failed = state.get_task(db, identity)
    assert failed["status"] == "failed"
    assert failed["result"] == {"error": "hosted room member session is busy"}
    assert runtime._ambiguous_rooms == {}
    assert "observation failed after submit" not in str(runtime.status()["last_error"])


def test_stop_is_bounded_and_does_not_interrupt_active_turn(db: Path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc, poll_interval_seconds=0.01)

    runtime.start()
    assert rpc.submitted.wait(1.0)
    started = time.monotonic()
    stopped = runtime.stop(timeout=0.5)

    assert stopped is True
    assert time.monotonic() - started < 0.5
    assert state.get_task(db, identity)["status"] == "running"
    assert not [call for call in rpc.calls if call[0] == "interrupt"]
