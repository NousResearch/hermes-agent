"""Runtime adapter for gateway-owned hosted room turns.

The durable state machine lives in :mod:`gateway.hosted_room_driver`; this module owns the
process-local worker and an injected session adapter, never the gateway server or agents.
One bounded supervisor schedules independent room workers: profile turn locks serialize Bots
sharing a profile while a room waiting for approval cannot stall unrelated rooms. Member
sessions reuse ``Group: <room_id>`` so a local-to-hosted migration keeps one transcript.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ContextManager, Literal, Protocol, cast

import psutil

from gateway import hosted_room_driver as state
from gateway import status as gateway_status

_CANCEL_ROUTE_RETRIES = 8
_STOP_ACK_STATUSES = {"cancelled", "interrupted"}

ROOM_SESSION_SOURCE = "bot_room"
MAX_TERMINAL_TEXT_BYTES = 64 * 1024
_ROOM_SESSION_TITLE_PREFIX = "Group: "
_ROOM_SESSION_TITLE_MAX_CHARS = 100
_TERMINAL_TRUNCATION_NOTICE = (
    "\n\n[Reply truncated. Ask the Bot to share the full result as a file.]")
_STOP_PENDING = "stop retry remains pending: {exc}"

OwnerProcessState = Literal["alive", "dead", "unknown"]
_ProcessKey = tuple[str, state.TaskIdentity, int]
_ProcessAdmission = tuple[Any, str, str]
_PROCESS_SUBMISSION_LOCK = threading.Lock()
_PROCESS_SUBMISSIONS: set[_ProcessKey] = set()
_PROCESS_ADMISSIONS: dict[_ProcessKey, _ProcessAdmission] = {}


def _submission_fence_key(
    db_path: Path | str,
    identity: state.TaskIdentity,
    execution_generation: int,
) -> _ProcessKey:
    database = os.path.normcase(str(Path(db_path).resolve()))
    return database, identity, int(execution_generation)


def _register_process_submission(key: _ProcessKey) -> None:
    with _PROCESS_SUBMISSION_LOCK:
        _PROCESS_SUBMISSIONS.add(key)


def _unregister_process_submission(key: _ProcessKey) -> None:
    with _PROCESS_SUBMISSION_LOCK:
        _PROCESS_SUBMISSIONS.discard(key)


def _process_submission_is_registered(key: _ProcessKey) -> bool:
    with _PROCESS_SUBMISSION_LOCK:
        return key in _PROCESS_SUBMISSIONS


def _register_process_admission(
    key: _ProcessKey,
    *,
    transport: Any,
    profile: str,
    session_id: str,
) -> None:
    with _PROCESS_SUBMISSION_LOCK:
        _PROCESS_ADMISSIONS[key] = (transport, profile, session_id)


def _unregister_process_admission(key: _ProcessKey) -> None:
    with _PROCESS_SUBMISSION_LOCK:
        _PROCESS_ADMISSIONS.pop(key, None)


def _process_admission(key: _ProcessKey) -> _ProcessAdmission | None:
    with _PROCESS_SUBMISSION_LOCK:
        return _PROCESS_ADMISSIONS.get(key)


def _owner_process_state(pid: int, start_time: int) -> OwnerProcessState:
    """Read one process identity without sending a signal on Windows."""

    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return "dead"
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return "dead"
    except Exception:
        return "unknown"

    try:
        current_start = gateway_status.get_process_start_time(pid)
    except Exception:
        return "unknown"
    if (
        current_start is None
        or isinstance(current_start, bool)
        or not isinstance(current_start, int)
        or current_start < 1
    ):
        return "unknown"
    return "alive" if current_start == start_time else "dead"


class HostedRoomProfileUnavailableError(RuntimeError):
    """The frozen target profile disappeared before prompt admission."""


class InternalSessionRPC(Protocol):
    """Normalized in-process session operations required by the room driver.

    ``submit`` durably reports one fenced turn's terminal result via ``on_terminal``;
    ``interrupt`` acts only while the current turn still matches ``expected_task_id``.
    """

    def resolve_exact(
        self, *, profile: str, title: str, source: str) -> Mapping[str, Any] | None: ...
    def create(self, *, profile: str, title: str, source: str) -> Mapping[str, Any]: ...
    def resume(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]: ...
    def submit(
        self, *, profile: str, session_id: str, prompt: str, source: str, task: state.TaskIdentity,
        execution_generation: int, on_terminal: Callable[[Mapping[str, Any]], None], member_id: str,
    ) -> Mapping[str, Any]: ...
    def history(
        self, *, profile: str, session_id: str, source: str) -> Sequence[Mapping[str, Any]]: ...
    def info(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]: ...
    def interrupt(
        self, *, profile: str, session_id: str, source: str, expected_task_id: str,
        expected_task: state.TaskIdentity | None = None, expected_execution_generation: int | None = None,
        expected_member_id: str | None = None,
    ) -> Mapping[str, Any] | None: ...


MemberTransportResolver = Callable[["HostedRoomBinding", Mapping[str, Any]], InternalSessionRPC]


@dataclass(frozen=True)
class HostedRoomBinding:
    """Current server-issued authority coordinate for one hosted room."""

    room_id: str
    gateway_id: str
    authority_epoch: int


@dataclass(frozen=True)
class _TerminalReceipt:
    status: state.TerminalStatus
    settlement_id: str
    result: dict[str, Any]


@dataclass(frozen=True)
class _RecoveryInspection:
    terminal: _TerminalReceipt | None
    active: bool
    status: str | None


_NO_INSPECTION = _RecoveryInspection(terminal=None, active=False, status=None)


def _session_kw(profile: str, session_id: str) -> dict[str, str]:
    return {"profile": profile, "session_id": session_id, "source": ROOM_SESSION_SOURCE}


def _fences(task: Mapping[str, Any]) -> dict[str, int]:
    return {"expected_execution_generation": int(task["execution_generation"]),
            "expected_cancel_generation": int(task["cancel_generation"])}


class HostedRoomRuntime:
    """Run queued hosted-room tasks independently of Desktop connections."""

    def __init__(
        self, *, db_path: Path | str,
        rooms: Iterable[HostedRoomBinding] | Callable[[], Iterable[HostedRoomBinding]],
        turn_lock: Callable[[str], ContextManager[Any]], rpc: InternalSessionRPC | None = None,
        transport_resolver: MemberTransportResolver | None = None,
        profile_available: Callable[[str], bool] | None = None,
        prepare_room: Callable[[HostedRoomBinding], None] | None = None,
        publish_terminal: Callable[[HostedRoomBinding, Mapping[str, Any]], None] | None = None,
        pending_action: Callable[[str, str, Mapping[str, Any] | None], None] | None = None,
        clock: Callable[[], float] = time.time,
        lease_ttl_seconds: float = 30.0, poll_interval_seconds: float = 5.0,
        active_poll_interval_seconds: float = 0.25, turn_timeout_seconds: float = 1830.0,
        indeterminate_defer_seconds: float = 60.0, max_concurrent_rooms: int = 4,
        unavailable_retry_min_seconds: float = 1.0, unavailable_retry_max_seconds: float = 30.0,
        process_generation: str | None = None,
        process_pid: int | None = None, process_start_time: int | None = None,
        owner_liveness: Callable[[int, int], OwnerProcessState] | None = None) -> None:
        positive = dict(
            lease_ttl_seconds=lease_ttl_seconds, poll_interval_seconds=poll_interval_seconds,
            active_poll_interval_seconds=active_poll_interval_seconds,
            turn_timeout_seconds=turn_timeout_seconds,
            indeterminate_defer_seconds=indeterminate_defer_seconds)
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if (not isinstance(max_concurrent_rooms, int) or isinstance(max_concurrent_rooms, bool)
                or max_concurrent_rooms < 1):
            raise ValueError("max_concurrent_rooms must be a positive integer")
        if not 0 < unavailable_retry_min_seconds <= unavailable_retry_max_seconds:
            raise ValueError("unavailable retry bounds are invalid")
        if rpc is None and transport_resolver is None:
            raise ValueError("rpc or transport_resolver is required")
        self.db_path = Path(db_path)
        self.rpc, self.transport_resolver, self.turn_lock = rpc, transport_resolver, turn_lock
        self.prepare_room, self.publish_terminal = prepare_room, publish_terminal
        self.pending_action, self.clock = pending_action, clock
        for name, value in positive.items():
            setattr(self, name, float(value))
        self.profile_available = profile_available or (lambda _profile: True)
        self.max_concurrent_rooms = max_concurrent_rooms
        self.unavailable_retry_min_seconds, self.unavailable_retry_max_seconds = (
            float(unavailable_retry_min_seconds), float(unavailable_retry_max_seconds))
        self.process_generation = process_generation or uuid.uuid4().hex
        self.process_pid = os.getpid() if process_pid is None else process_pid
        if (
            isinstance(self.process_pid, bool)
            or not isinstance(self.process_pid, int)
            or self.process_pid < 1
        ):
            raise ValueError("process_pid must be a positive integer")
        if process_start_time is None:
            try:
                process_start_time = gateway_status.get_process_start_time(self.process_pid)
            except Exception:
                process_start_time = None
        if process_start_time is not None and (
            isinstance(process_start_time, bool)
            or not isinstance(process_start_time, int)
            or process_start_time < 1
        ):
            raise ValueError("process_start_time must be a positive integer or None")
        self.process_start_time = process_start_time
        self.owner_liveness = owner_liveness or _owner_process_state
        self._rooms_provider: Callable[[], Iterable[HostedRoomBinding]] = (
            cast(Callable[[], Iterable[HostedRoomBinding]], rooms) if callable(rooms)
            else (lambda bindings=tuple(rooms): bindings))
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = self._last_error = None
        self._room_threads: dict[str, threading.Thread] = {}
        self._rooms_needing_reschedule: set[str] = set()
        self._leases: dict[str, state.DriverLease] = {}
        self._recovered_leases: set[tuple[str, int]] = set()
        self._inspected_indeterminate_attempts: set[tuple[str, str, int]] = set()
        self._ambiguous_rooms: dict[str, float] = {}
        self._unavailable_route_retries: dict[tuple[str, str], dict[str, float]] = {}
        self._blocked_rooms: set[str] = set()
        self._status_lock = threading.Lock()
        self._activity_condition = threading.Condition(self._status_lock)
        self._activity_generation = 0
        self._current_tasks: dict[str, state.TaskIdentity] = {}
        self._submitting_tasks: dict[
            str, tuple[state.TaskIdentity, int]
        ] = {}
        self._room_schedule_cursor = 0
        self._last_error: str | None = None
        self._cycles = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start the bounded room-worker supervisor idempotently."""
        with self._status_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._wake.set()
            self._thread = threading.Thread(
                target=self._worker_loop, name="hosted-room-driver-supervisor", daemon=True)
            self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> bool:
        """Request a bounded clean stop without interrupting accepted turns."""
        self._stop.set()
        with self._activity_condition:
            self._activity_condition.notify_all()
        self._wake.set()
        with self._status_lock:
            thread = self._thread
        if thread is None:
            return True
        deadline = time.monotonic() + max(0.0, timeout)
        thread.join(max(0.0, deadline - time.monotonic()))
        with self._status_lock:
            room_threads = tuple(self._room_threads.values())
        for room_thread in room_threads:
            room_thread.join(max(0.0, deadline - time.monotonic()))
        return not any(t.is_alive() for t in (thread, *room_threads))

    def wakeup(self) -> None:
        """Wake the worker after task admission or a room-state change."""
        with self._activity_condition:
            # Rooms still owning a worker slot are revisited once that thread exits, closing
            # the terminal-publication/route-repair race without busy-looping idle rooms.
            self._rooms_needing_reschedule.update(self._room_threads)
            self._activity_generation += 1
            self._activity_condition.notify_all()
        self._wake.set()

    def _activity_snapshot(self) -> int:
        with self._activity_condition:
            return self._activity_generation

    def _wait_for_activity(self, generation: int, timeout: float) -> None:
        with self._activity_condition:
            self._activity_condition.wait_for(
                lambda: self._activity_generation != generation
                or self._stop.is_set(),
                timeout=max(0.0, timeout),
            )

    def status(self) -> dict[str, Any]:
        """Return a transport-neutral snapshot of runtime health."""
        with self._status_lock:
            thread = self._thread
            current_tasks = tuple(self._current_tasks.values())
            return {
                "running": bool(thread and thread.is_alive()), "stopping": self._stop.is_set(),
                "process_generation": self.process_generation,
                "current_task": current_tasks[0] if current_tasks else None,
                "current_tasks": current_tasks, "leased_rooms": tuple(sorted(self._leases)),
                "blocked_rooms": tuple(sorted(self._blocked_rooms)),
                "last_error": self._last_error, "cycles": self._cycles}

    def cancel(
        self,
        identity: state.TaskIdentity,
        *,
        cancel_id: str,
    ) -> dict[str, Any]:
        """Persist one exact Stop and commit it only after durable proof."""

        for _ in range(_CANCEL_ROUTE_RETRIES):
            before = state.get_task(self.db_path, identity)
            if before["status"] == "cancelled":
                return before
            if before["status"] in state.TERMINAL_STATUSES:
                return before
            if before["status"] in {"queued", "deferred"}:
                try:
                    cancelled = state.cancel_task(
                        self.db_path,
                        identity,
                        cancel_id=cancel_id,
                        expected_cancel_generation=before["cancel_generation"],
                        clock=self.clock,
                    )
                except (state.InvalidTaskTransitionError, state.StaleTaskError):
                    continue
                self.wakeup()
                return cancelled
            try:
                stopping = state.begin_task_cancel(
                    self.db_path, identity, cancel_id=cancel_id,
                    expected_cancel_generation=before["cancel_generation"], clock=self.clock)
            except (state.InvalidTaskTransitionError, state.StaleTaskError):
                continue
            binding = self._binding_for_room(identity.room_id)
            try:
                if binding is not None:
                    lease = self._ensure_lease(binding)
                    if self._peer_stop_acknowledged(binding, stopping):
                        if not self._settle_stopping_completion(binding, stopping, lease):
                            stopping = self._complete_acknowledged_stop(
                                binding, stopping, lease
                            )
                    elif self._settle_stopping_completion(binding, stopping, lease):
                        stopping = state.get_task(self.db_path, identity)
                    elif self._interrupt_stopping_task(binding, stopping):
                        lease = self._ensure_lease(binding)
                        current = state.get_task(self.db_path, identity)
                        if current["status"] in state.TERMINAL_STATUSES:
                            stopping = current
                        elif self._settle_stopping_completion(binding, current, lease):
                            stopping = state.get_task(self.db_path, identity)
                        else:
                            stopping = self._complete_acknowledged_stop(
                                binding, current, lease
                            )
            except Exception as exc:
                self._record_error(f"stop remains pending: {exc}")
            stopping = state.get_task(self.db_path, identity)
            self.wakeup()
            return stopping
        final = state.get_task(self.db_path, identity)
        if final["status"] == "cancelled":
            return final
        raise state.InvalidTaskTransitionError(
            "cancel kept losing races with task transitions "
            f"(last observed state '{final['status']}')")

    def retry_indeterminate(self, identity: state.TaskIdentity) -> dict[str, Any]:
        """Explicitly retry one uncertain attempt under the current room lease."""
        task = state.get_task(self.db_path, identity)
        if task["status"] not in {"indeterminate", "deferred"}:
            raise state.InvalidTaskTransitionError(f"cannot retry task in state '{task['status']}'")
        binding = self._binding_for_room(identity.room_id)
        if binding is None:
            raise state.RoomUnavailableError("hosted room is unavailable")
        lease = self._ensure_lease(binding)
        if task["status"] == "deferred":
            return self._requeue(state.requeue_deferred_task, task, lease, identity.room_id)
        # Explicit Retry may resume the exact stored session; the automatic abandoned-attempt
        # scan stays non-resuming for local sessions.
        inspection = self._inspect_recovery_session(binding, task)
        if inspection.terminal is not None:
            return self._resolve_indeterminate(binding, task, lease, inspection.terminal)
        if inspection.status == "cancelled":
            return self._fenced(
                state.resolve_indeterminate_cancellation, binding, task, lease,
                cancel_id=f"remote-cancel:{task['execution_generation']}")
        if inspection.active:
            self._set_blocked(identity.room_id, True)
            raise state.InvalidTaskTransitionError(
                "cannot retry while the original task attempt is still active")
        return self._requeue(state.requeue_indeterminate_task, task, lease, identity.room_id)

    def _publish(self, binding: HostedRoomBinding, task: dict[str, Any]) -> dict[str, Any]:
        if self.publish_terminal is not None:
            self.publish_terminal(binding, task)
        return task

    def _set_blocked(self, room_id: str, blocked: bool) -> None:
        with self._status_lock:
            (self._blocked_rooms.add if blocked else self._blocked_rooms.discard)(room_id)

    def _fenced(
        self, op: Callable[..., dict[str, Any]], binding: HostedRoomBinding | None,
        task: Mapping[str, Any], lease: state.DriverLease, *, publish: bool = True, **extra: Any,
    ) -> dict[str, Any]:
        """Run one lease-fenced state transition on ``task``; ``extra`` may override fences."""
        kwargs = {**_fences(task), "clock": self.clock, **extra}
        result = op(self.db_path, task["identity"], lease, **kwargs)
        return self._publish(binding, result) if publish and binding is not None else result

    def _requeue(
        self, requeue: Callable[..., dict[str, Any]], task: Mapping[str, Any],
        lease: state.DriverLease, room_id: str) -> dict[str, Any]:
        retried = self._fenced(requeue, None, task, lease)
        self._set_blocked(room_id, False)
        self.wakeup()
        return retried

    def _resolve_indeterminate(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease,
        terminal: _TerminalReceipt, *, publish: bool = True) -> dict[str, Any]:
        return self._fenced(
            state.resolve_indeterminate_task, binding, task, lease, publish=publish,
            **asdict(terminal))

    def _finish_stop(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease
    ) -> bool:
        """Settle an exact receipt first, then recheck the fence after Stop acknowledgement."""
        if self._settle_stopping_completion(binding, task, lease):
            return True
        if not self._interrupt_stopping_task(binding, task):
            return False
        lease = self._ensure_lease(binding)
        current = state.get_task(self.db_path, task["identity"])
        if current["status"] in state.TERMINAL_STATUSES:
            return True
        if not self._settle_stopping_completion(binding, current, lease):
            self._complete_acknowledged_stop(binding, current, lease)
        return True

    def _resume_exact(
        self, transport: InternalSessionRPC, room_id: str, profile: str) -> str | None:
        """Resume the canonical room session and return its runtime id (None when absent).

        Probes must use the returned id, not the stored one: resume may hand back another.
        """
        session = self._resolve_or_create(transport, profile, room_id, create=False)
        return None if session is None else _session_id(session)

    def _open_session(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], *, peer_only: bool = False
    ) -> tuple[InternalSessionRPC | None, str | None, str | None]:
        """Return ``(transport, profile, resumed session id)``; no id when the transport is
        missing (or local under ``peer_only``) or the session is absent."""
        transport = self._transport_for(binding, task)
        if transport is None or (peer_only and transport is self.rpc):
            return transport, None, None
        profile = task["payload"]["target_profile"]
        return transport, profile, self._resume_exact(transport, binding.room_id, profile)

    @staticmethod
    def _terminal_from_history(
        transport: InternalSessionRPC, profile: str, session_id: str, task: Mapping[str, Any]
    ) -> _TerminalReceipt | None:
        return _find_terminal_receipt(
            transport.history(**_session_kw(profile, session_id)),
            task["identity"], int(task["execution_generation"]))

    def _peer_stop_acknowledged(self, binding: HostedRoomBinding, task: Mapping[str, Any]) -> bool:
        """Probe a peer's exact durable terminal Stop receipt before reading history."""
        transport, profile, session_id = self._open_session(binding, task, peer_only=True)
        if session_id is None:
            return False
        info = transport.info(**_session_kw(profile, session_id))
        return (
            not _info_active(info)
            and str(info.get("status") or "") in _STOP_ACK_STATUSES
            and str(info.get("task_id") or "") == task["identity"].task_id
            and int(info.get("execution_generation") or 0) == int(task["execution_generation"]))

    @staticmethod
    def _info_acknowledges_peer_cancel(
        info: Mapping[str, Any], task: Mapping[str, Any]
    ) -> bool:
        """Accept only one exact peer task attempt's terminal Stop receipt."""

        return (
            not bool(info.get("active", info.get("running", False)))
            and str(info.get("status") or "") in {"cancelled", "interrupted"}
            and str(info.get("task_id") or "") == task["identity"].task_id
            and int(info.get("execution_generation") or 0)
            == int(task["execution_generation"])
        )

    def _require_profile_available(self, profile: str) -> None:
        if self.profile_available(profile) is not True:
            raise HostedRoomProfileUnavailableError(
                f"hosted room profile '{profile}' is unavailable"
            )


    def _stopping_owner_state(
        self,
        task: Mapping[str, Any],
    ) -> Literal["owned", "legacy", "alive", "dead", "unknown"]:
        if task.get("run_process_generation") == self.process_generation:
            return "owned"
        owner_pid = task.get("run_process_pid")
        owner_started = task.get("run_process_start_time")
        if (
            isinstance(owner_pid, bool)
            or not isinstance(owner_pid, int)
            or owner_pid < 1
            or isinstance(owner_started, bool)
            or not isinstance(owner_started, int)
            or owner_started < 1
        ):
            return "legacy"
        if owner_pid == self.process_pid and owner_started == self.process_start_time:
            return "owned"
        return self.owner_liveness(owner_pid, owner_started)

    def _interrupt_stopping_task(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
    ) -> bool:
        owner_state = self._stopping_owner_state(task)
        with self._status_lock:
            current_attempt_is_local = (
                self._current_tasks.get(binding.room_id) == task["identity"]
            )
        foreign_owner_may_be_live = (
            owner_state in {"alive", "unknown"} and not current_attempt_is_local
        )
        transport = self._transport_for(binding, task)
        if transport is None:
            return False
        profile = task["payload"]["target_profile"]

        if transport is not self.rpc:
            session = transport.resolve_exact(
                profile=profile,
                title=room_session_title(binding.room_id),
                source=ROOM_SESSION_SOURCE,
            )
            if session is None:
                return False
            resumed = transport.resume(
                profile=profile,
                session_id=_session_id(session),
                source=ROOM_SESSION_SOURCE,
            )
            session_id = _session_id(resumed)
            info = transport.info(
                profile=profile,
                session_id=session_id,
                source=ROOM_SESSION_SOURCE,
            )
            active = bool(info.get("active", info.get("running", False)))
            if not active:
                return self._info_acknowledges_peer_cancel(info, task)
            if not _info_is_active_for(info, task["identity"], require_exact=True):
                return False
            result = transport.interrupt(
                profile=profile,
                session_id=session_id,
                source=ROOM_SESSION_SOURCE,
                expected_task_id=task["identity"].task_id,
            )
            if result is None:
                return False
            return result.get("interrupted") is True or str(result.get("status") or "") in {
                "cancelled",
                "interrupted",
            }

        key = _submission_fence_key(
            self.db_path,
            task["identity"],
            int(task["execution_generation"]),
        )
        exact_submission = (
            task["identity"],
            int(task["execution_generation"]),
        )
        with self._status_lock:
            submission_in_progress = (
                self._submitting_tasks.get(binding.room_id) == exact_submission
            )
        submission_in_progress = (
            submission_in_progress or _process_submission_is_registered(key)
        )
        admission = _process_admission(key)
        if admission is not None:
            admitted_transport, admitted_profile, admitted_session_id = admission
            info = admitted_transport.info(
                profile=admitted_profile,
                session_id=admitted_session_id,
                source=ROOM_SESSION_SOURCE,
            )
            active = bool(info.get("active", info.get("running", False)))
            current = state.get_task(self.db_path, task["identity"])
            if current["status"] in state.TERMINAL_STATUSES:
                return True
            if not active:
                return True
            if not _info_is_active_for(
                info, task["identity"], require_exact=True,
                execution_generation=int(task["execution_generation"]),
                member_id=_member_id(task),
            ):
                return False
            result = admitted_transport.interrupt(
                profile=admitted_profile,
                session_id=admitted_session_id,
                source=ROOM_SESSION_SOURCE,
                expected_task_id=task["identity"].task_id,
                expected_task=task["identity"],
                expected_execution_generation=int(task["execution_generation"]),
                expected_member_id=_member_id(task),
            )
            if result is None:
                return False
            if result.get("interrupted") is True or str(result.get("status") or "") in {
                "cancelled",
                "interrupted",
                "stopping",
            }:
                return True
            info = admitted_transport.info(
                profile=admitted_profile,
                session_id=admitted_session_id,
                source=ROOM_SESSION_SOURCE,
            )
            return not bool(info.get("active", info.get("running", False)))
        if submission_in_progress:
            return False
        if owner_state == "dead":
            return True

        session = transport.resolve_exact(
            profile=profile,
            title=room_session_title(binding.room_id),
            source=ROOM_SESSION_SOURCE,
        )
        if session is None:
            return not foreign_owner_may_be_live
        resumed = transport.resume(
            profile=profile,
            session_id=_session_id(session),
            source=ROOM_SESSION_SOURCE,
        )
        session_id = _session_id(resumed)
        info = transport.info(
            profile=profile,
            session_id=session_id,
            source=ROOM_SESSION_SOURCE,
        )
        active = bool(info.get("active", info.get("running", False)))
        if not active:
            if self._info_acknowledges_peer_cancel(info, task):
                return True
            return not foreign_owner_may_be_live
        if not _info_is_active_for(
            info, task["identity"], require_exact=True,
            execution_generation=int(task["execution_generation"]),
            member_id=_member_id(task),
        ):
            return False
        result = transport.interrupt(
            **_session_kw(profile, session_id), expected_task_id=task["identity"].task_id,
            expected_task=task["identity"],
            expected_execution_generation=int(task["execution_generation"]),
            expected_member_id=_member_id(task))
        return result is not None and (
            result.get("interrupted") is True
            or str(result.get("status") or "") in _STOP_ACK_STATUSES)

    def _settle_stopping_completion(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease
    ) -> bool:
        """Publish a terminal receipt that arrived before Stop was acknowledged."""
        receipt = _durable_terminal_receipt(
            state.get_terminal_receipt(
                self.db_path,
                task["identity"],
                execution_generation=int(task["execution_generation"]),
            )
        )
        if receipt is None:
            transport = self._transport_for(binding, task)
            if transport is None:
                return False
            key = _submission_fence_key(
                self.db_path,
                task["identity"],
                int(task["execution_generation"]),
            )
            if transport is self.rpc and _process_admission(key) is not None:
                return False
            profile = task["payload"]["target_profile"]
            session = transport.resolve_exact(
                profile=profile,
                title=room_session_title(binding.room_id),
                source=ROOM_SESSION_SOURCE,
            )
            if session is None:
                return False
            resumed = transport.resume(
                profile=profile,
                session_id=_session_id(session),
                source=ROOM_SESSION_SOURCE,
            )
            receipt = _find_terminal_receipt(
                transport.history(
                    profile=profile,
                    session_id=_session_id(resumed),
                    source=ROOM_SESSION_SOURCE,
                ),
                task["identity"],
                int(task["execution_generation"]),
            )
        if receipt is None:
            return False
        self._fenced(state.settle_stopping_task, binding, task, lease, **asdict(receipt))
        return True

    def _report_pending_action(
        self, task: Mapping[str, Any], *, session_id: str, info: Mapping[str, Any]) -> None:
        if self.pending_action is None:
            return
        approval, action = info.get("pending_approval") or info.get("approval"), None
        if isinstance(approval, Mapping):
            choices = [c for c in approval.get("choices") or () if c in {"once", "deny"}]
            safe_approval = {**approval, "choices": choices or ["once", "deny"]}
            action = {
                "kind": "approval",
                "task_id": task["identity"].task_id,
                "thread_id": task["identity"].thread_id,
                "turn_id": task["identity"].turn_id,
                "execution_generation": int(task["execution_generation"]),
                "run_id": info.get("run_id"), "session_id": session_id,
                "request_id": safe_approval.get("request_id"), "approval": safe_approval}
        self.pending_action(task["identity"].room_id, _member_id(task), action)

    def _retry_stopping_tasks(self, binding: HostedRoomBinding, lease: state.DriverLease) -> bool:
        for task in self._tasks(binding, "stopping"):
            try:
                lease = self._renew_lease_if_needed(lease)
                if self._peer_stop_acknowledged(binding, task):
                    # Exact peer terminal status is already the durable Stop
                    # acknowledgement. Do not read peer history afterward: an
                    # interrupted historical reply can look like a failed turn
                    # and overwrite the cancellation contract.
                    self._complete_acknowledged_stop(binding, task, lease)
                    continue
                if not self._finish_stop(binding, task, lease):
                    return True
            except Exception as exc:
                self._record_error(_STOP_PENDING.format(exc=exc))
                return True
        return False

    # ------------------------------------------------------------------ scheduling
    def _worker_loop(self) -> None:
        try:
            while not self._stop.is_set():
                # Clear before work so a write racing the cycle forces a follow-up pass.
                self._wake.clear()
                try:
                    self._run_cycle()
                except Exception as exc:  # keep independent rooms serviceable
                    self._record_error(f"worker cycle failed: {exc}")
                with self._status_lock:
                    self._cycles += 1
                self._wake.wait(self.poll_interval_seconds)
        finally:
            while True:
                with self._status_lock:
                    room_threads = tuple(t for t in self._room_threads.values() if t.is_alive())
                if not room_threads:
                    break
                for room_thread in room_threads:
                    room_thread.join(self.active_poll_interval_seconds)
            self._release_idle_leases()

    def _run_cycle(self) -> None:
        with self._status_lock:
            supervisor = self._thread
        if threading.current_thread() is not supervisor:
            for binding in tuple(self._rooms_provider()):
                if self._stop.is_set():
                    return
                self._run_room_once(binding)
            return
        with self._status_lock:
            self._room_threads = {
                room_id: t for room_id, t in self._room_threads.items() if t.is_alive()}
            available = self.max_concurrent_rooms - len(self._room_threads)
            active_rooms = set(self._room_threads)
        if available <= 0:
            return
        bindings = tuple(self._rooms_provider())
        if not bindings:
            return
        start = self._room_schedule_cursor % len(bindings)
        self._room_schedule_cursor = (start + 1) % len(bindings)
        for binding in bindings[start:] + bindings[:start]:
            if self._stop.is_set() or available <= 0:
                return
            if binding.room_id in active_rooms:
                continue
            room_thread = threading.Thread(
                target=self._run_room_once, args=(binding,),
                name=f"hosted-room-{binding.room_id[:24]}", daemon=True)
            with self._status_lock:
                self._room_threads[binding.room_id] = room_thread
            active_rooms.add(binding.room_id)
            available -= 1
            room_thread.start()

    def _run_room_once(self, binding: HostedRoomBinding) -> None:
        try:
            self._process_room(binding)
        except state.LeaseHeldError:
            pass
        except Exception as exc:
            if isinstance(exc, (state.RoomUnavailableError, state.StaleLeaseError)):
                self._drop_lease(binding.room_id)
                self._set_blocked(binding.room_id, False)
            self._record_error(f"room {binding.room_id}: {exc}")
        finally:
            with self._status_lock:
                if self._room_threads.get(binding.room_id) is threading.current_thread():
                    self._room_threads.pop(binding.room_id, None)
                should_wake = binding.room_id in self._rooms_needing_reschedule
                self._rooms_needing_reschedule.discard(binding.room_id)
            if should_wake:
                self.wakeup()

    def _process_room(self, binding: HostedRoomBinding) -> None:
        if self.prepare_room is not None:
            self.prepare_room(binding)
        self._inspect_abandoned_attempts(binding)
        deferred_until = self._ambiguous_rooms.get(binding.room_id)
        if deferred_until is not None:
            if self._tasks(binding, "running") and self.clock() < deferred_until:
                return
            self._ambiguous_rooms.pop(binding.room_id, None)
        lease = self._ensure_lease(binding)
        if (lease.room_id, lease.lease_generation) not in self._recovered_leases:
            state.recover_room(self.db_path, lease, clock=self.clock)
            self._recovered_leases.add((lease.room_id, lease.lease_generation))
        if self._retry_stopping_tasks(binding, lease):
            self._set_blocked(binding.room_id, True)
            return
        if self._reconcile_indeterminate(binding, lease):
            return
        for task in self._tasks(binding, "queued"):
            retry = self._unavailable_route_retries.get(
                (task["identity"].room_id, _member_id(task)))
            if self._stop.is_set() or (
                    retry is not None and self.clock() < retry["next_attempt_at"]):
                return
            profile = task["payload"]["target_profile"]
            # Keep the durable task queued until both its remote route and
            # local profile lock are available. Do not mark it running before
            # the model's serialization lock is owned.
            with self.turn_lock(profile):
                if self._transport_for(binding, task) is self.rpc:
                    try:
                        self._require_profile_available(profile)
                    except HostedRoomProfileUnavailableError as exc:
                        self._defer_unavailable_route(task)
                        self._record_error(str(exc))
                        return
                lease = self._renew_lease_if_needed(lease)
                attempt = state.start_task(
                    self.db_path,
                    task["identity"],
                    lease,
                    expected_cancel_generation=task["cancel_generation"],
                    clock=self.clock,
                )
                self._execute_attempt(
                    binding,
                    task,
                    attempt,
                    turn_lock_held=True,
                )
            current = state.get_task(self.db_path, task["identity"])
            if current["status"] not in state.TERMINAL_STATUSES:
                return

    def _defer_unavailable_route(self, task: Mapping[str, Any]) -> float:
        key = (task["identity"].room_id, _member_id(task))
        previous = self._unavailable_route_retries.get(key)
        lo, hi = self.unavailable_retry_min_seconds, self.unavailable_retry_max_seconds
        delay = lo if previous is None else min(hi, max(lo, previous["delay"] * 2))
        self._unavailable_route_retries[key] = {
            "delay": delay, "next_attempt_at": self.clock() + delay}
        return delay

    # ------------------------------------------------------------------ leases
    def _ensure_lease(self, binding: HostedRoomBinding) -> state.DriverLease:
        with self._status_lock:
            current = self._leases.get(binding.room_id)
        if current is not None:
            try:
                return self._renew_lease_if_needed(current)
            except state.StaleLeaseError:
                self._drop_lease(binding.room_id)
        lease = state.acquire_lease(
            self.db_path,
            room_id=binding.room_id,
            gateway_id=binding.gateway_id,
            authority_epoch=binding.authority_epoch,
            process_generation=self.process_generation,
            process_pid=self.process_pid,
            process_start_time=self.process_start_time,
            ttl_seconds=self.lease_ttl_seconds,
            clock=self.clock,
        )
        with self._status_lock:
            self._leases[binding.room_id] = lease
            self._recovered_leases = {k for k in self._recovered_leases if k[0] != binding.room_id}
        return lease

    def _renew_lease_if_needed(
        self, lease: state.DriverLease, *, force: bool = False) -> state.DriverLease:
        if not force and self.clock() < lease.expires_at - (self.lease_ttl_seconds / 2):
            return lease
        renewed = state.renew_lease(
            self.db_path, lease, ttl_seconds=self.lease_ttl_seconds, clock=self.clock)
        with self._status_lock:
            self._leases[lease.room_id] = renewed
        return renewed

    def _drop_lease(self, room_id: str) -> None:
        with self._status_lock:
            self._leases.pop(room_id, None)

    def _release_idle_leases(self) -> None:
        for room_id, lease in tuple(self._leases.items()):
            with suppress(state.DriverStateError):
                state.release_lease(self.db_path, lease, clock=self.clock)
                self._drop_lease(room_id)

    # ------------------------------------------------------------------ attempt execution
    def _execute_attempt(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
        attempt: state.TaskAttempt,
        *,
        turn_lock_held: bool = False,
    ) -> None:
        profile, submit_attempted = task["payload"]["target_profile"], False
        transport = self._transport_for(binding, task)
        submission_key = _submission_fence_key(
            self.db_path,
            attempt.identity,
            attempt.execution_generation,
        )
        _register_process_submission(submission_key)
        with self._status_lock:
            self._current_tasks[binding.room_id] = attempt.identity
            self._submitting_tasks[binding.room_id] = (
                attempt.identity,
                attempt.execution_generation,
            )
        try:
            lock_context = (
                nullcontext()
                if turn_lock_held
                else self.turn_lock(profile)
            )
            with lock_context:
                if transport is self.rpc:
                    self._require_profile_available(profile)
                session = self._resolve_or_create(transport, profile, binding.room_id)
                session_id = _session_id(session)
                if transport is self.rpc:
                    _register_process_admission(
                        submission_key,
                        transport=transport,
                        profile=profile,
                        session_id=session_id,
                    )
                    self._require_profile_available(profile)
                deadline_monotonic = time.monotonic() + self.turn_timeout_seconds
                current = state.get_task(self.db_path, attempt.identity)
                if (
                    current["status"] != "running"
                    or int(current["execution_generation"])
                    != attempt.execution_generation
                    or int(current["cancel_generation"])
                    != attempt.cancel_generation
                ):
                    return
                submit_attempted = True
                try:
                    transport.submit(
                        profile=profile,
                        session_id=session_id,
                        prompt=task["payload"]["prompt"],
                        source=ROOM_SESSION_SOURCE,
                        task=attempt.identity,
                        execution_generation=attempt.execution_generation,
                        on_terminal=lambda receipt: self._on_terminal(binding, attempt, receipt),
                        member_id=_member_id(task),
                    )
                finally:
                    _unregister_process_submission(submission_key)
                    with self._status_lock:
                        if (
                            self._submitting_tasks.get(binding.room_id)
                            == (attempt.identity, attempt.execution_generation)
                        ):
                            self._submitting_tasks.pop(binding.room_id, None)
                self._unavailable_route_retries.pop(
                    (task["identity"].room_id, _member_id(task)), None)
                receipt = self._wait_for_terminal(
                    binding,
                    profile=profile,
                    session_id=session_id,
                    attempt=attempt,
                    transport=transport,
                    deadline_monotonic=deadline_monotonic,
                )
                if receipt is None:
                    return
                self._settle_attempt_terminal(binding, attempt, receipt)
        except HostedRoomProfileUnavailableError as exc:
            try:
                state.requeue_not_admitted_task(
                    self.db_path,
                    attempt,
                    clock=self.clock,
                )
            except (state.StaleLeaseError, state.StaleTaskError) as fence_exc:
                self._drop_lease(binding.room_id)
                self._record_error(
                    f"task {attempt.identity.task_id} unavailable profile lost "
                    f"its fence: {fence_exc}"
                )
            else:
                delay = self._defer_unavailable_route(task)
                self._record_error(
                    f"task {attempt.identity.task_id} profile unavailable; "
                    f"queued for retry in {delay:g}s: {exc}"
                )
        except (state.StaleLeaseError, state.StaleTaskError) as exc:
            self._drop_lease(binding.room_id)
            self._record_task_error(attempt, f"fenced: {exc}")
        except Exception as exc:
            if submit_attempted and bool(getattr(exc, "not_admitted", False)):
                if not bool(getattr(exc, "retryable", False)):
                    self._settle_failure_if_current(attempt, exc)
                else:
                    try:
                        state.requeue_not_admitted_task(
                            self.db_path,
                            attempt,
                            clock=self.clock,
                        )
                    except (state.StaleLeaseError, state.StaleTaskError) as fence_exc:
                        self._drop_lease(binding.room_id)
                        self._ambiguous_rooms[binding.room_id] = attempt.lease.expires_at
                        self._record_error(
                            f"task {attempt.identity.task_id} not-admitted proof lost "
                            f"its fence: {fence_exc}"
                        )
                    else:
                        delay = self._defer_unavailable_route(task)
                        self._record_error(
                            f"task {attempt.identity.task_id} was not admitted; "
                            f"queued for retry in {delay:g}s"
                        )
            elif submit_attempted:
                self._mark_ambiguous(binding, attempt)
                self._record_task_error(attempt, f"observation failed after submit: {exc}")
            else:
                self._settle_failure_if_current(attempt, exc)
        finally:
            _unregister_process_submission(submission_key)
            _unregister_process_admission(submission_key)
            with self._status_lock:
                self._current_tasks.pop(binding.room_id, None)
                if (
                    self._submitting_tasks.get(binding.room_id)
                    == (attempt.identity, attempt.execution_generation)
                ):
                    self._submitting_tasks.pop(binding.room_id, None)
                # The task may have published a reply or exposed the next turn while this
                # thread held its slot: schedule exactly one follow-up after it leaves
                # (idle room scans never set this marker).
                self._rooms_needing_reschedule.add(binding.room_id)

    def _mark_ambiguous(self, binding: HostedRoomBinding, attempt: state.TaskAttempt) -> None:
        self._drop_lease(binding.room_id)
        self._ambiguous_rooms[binding.room_id] = attempt.lease.expires_at

    def _settle_attempt_terminal(
        self, binding: HostedRoomBinding, attempt: state.TaskAttempt, terminal: _TerminalReceipt,
    ) -> None:
        task = state.settle_task(self.db_path, attempt, **asdict(terminal), clock=self.clock)
        # Callback and durable-receipt observer race to the same SQLite transition.
        # Only its winner publishes; the service's durable publication scan retries
        # a failed publication even when this process dies after settlement.
        if not task.get("idempotent"):
            self._publish(binding, task)

    def _on_terminal(
        self, binding: HostedRoomBinding, attempt: state.TaskAttempt, receipt: Mapping[str, Any]
    ) -> None:
        """Durably commit one in-process terminal receipt for ``attempt``."""
        status = receipt.get("status")
        if status == "cancelled":
            self.wakeup()
            return
        terminal = _durable_terminal_receipt(receipt) or _TerminalReceipt(
            status="settled" if status == "settled" else "failed",
            settlement_id=receipt.get("settlement_id")
            or f"reply:{attempt.identity.task_id}:{attempt.execution_generation}",
            result=_bounded_terminal_result(receipt))
        try:
            self._settle_attempt_terminal(binding, attempt, terminal)
        except (state.StaleLeaseError, state.StaleTaskError) as fence_exc:
            recovered = False
            try:
                current = state.get_task(self.db_path, attempt.identity)
                if int(current["execution_generation"]) == attempt.execution_generation:
                    if current["status"] == "stopping":
                        lease = self._ensure_lease(binding)
                        self._fenced(
                            state.settle_stopping_task, binding, current, lease,
                            **asdict(terminal),
                            expected_execution_generation=attempt.execution_generation)
                        recovered = True
                    elif current["status"] in state.TERMINAL_STATUSES:
                        recovered = True
            except (state.LeaseHeldError, state.StaleLeaseError, state.StaleTaskError):
                recovered = False
            if not recovered:
                self._drop_lease(binding.room_id)
                self._record_task_error(attempt, f"terminal commit fenced: {fence_exc}")
        except state.DriverStateError as exc:
            # A malformed receipt must not escape the callback and hold the profile lock.
            self._settle_failure_if_current(
                attempt, RuntimeError(f"terminal result could not be committed: {exc}"))
        self.wakeup()

    def _wait_for_terminal(
        self, binding: HostedRoomBinding, *, profile: str, session_id: str,
        attempt: state.TaskAttempt, transport: InternalSessionRPC, deadline_monotonic: float,
    ) -> _TerminalReceipt | None:
        lease = attempt.lease
        while not self._stop.is_set():
            activity_generation = self._activity_snapshot()
            task = state.get_task(self.db_path, attempt.identity)
            if task["status"] in state.TERMINAL_STATUSES:
                return None
            if task["status"] == "stopping":
                try:
                    lease = self._renew_lease_if_needed(lease)
                    if self._finish_stop(binding, task, lease):
                        return None
                except Exception as exc:
                    self._record_error(_STOP_PENDING.format(exc=exc))
                self._wait_for_activity(
                    activity_generation, self.active_poll_interval_seconds
                )
                continue
            if time.monotonic() >= deadline_monotonic:
                self._expire_attempt_deadline(binding, task, lease)
                return None

            lease = self._renew_lease_if_needed(lease)
            receipt = _durable_terminal_receipt(
                state.get_terminal_receipt(
                    self.db_path,
                    attempt.identity,
                    execution_generation=attempt.execution_generation,
                )
            )
            if receipt is not None:
                return receipt
            info = transport.info(**_session_kw(profile, session_id))
            self._report_pending_action(task, session_id=session_id, info=info)
            if (
                transport is not self.rpc
                and not bool(info.get("active", info.get("running", False)))
            ):
                receipt = _find_terminal_receipt(
                    transport.history(
                        profile=profile,
                        session_id=session_id,
                        source=ROOM_SESSION_SOURCE,
                    ),
                    attempt.identity,
                    attempt.execution_generation,
                )
                if receipt is not None:
                    return receipt
            remaining = max(0.0, deadline_monotonic - time.monotonic())
            self._wait_for_activity(
                activity_generation,
                min(self.active_poll_interval_seconds, remaining),
            )
        return None

    def _complete_acknowledged_stop(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease
    ) -> dict[str, Any]:
        """Terminalize an acknowledged Stop: deadline stops publish an explicit failure."""
        if not str(task.get("cancel_id") or "").startswith("deadline:"):
            return state.complete_task_cancel(
                self.db_path, task["identity"], cancel_id=task["cancel_id"],
                expected_cancel_generation=task["cancel_generation"], clock=self.clock)
        return self._fenced(
            state.settle_stopping_task, binding, task, lease,
            settlement_id=f"deadline:{int(task['execution_generation'])}", status="failed",
            result={
                "error": "This Group Chat turn exceeded its configured time limit and was stopped.",
                "reason_code": "turn_deadline_exceeded",
                "timeout_seconds": self.turn_timeout_seconds})

    def _expire_attempt_deadline(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease
    ) -> None:
        """Fence, stop, and terminalize one exact attempt at its deadline."""
        if task["status"] == "running":
            task = state.begin_task_cancel(
                self.db_path, task["identity"], clock=self.clock,
                cancel_id=f"deadline:{int(task['execution_generation'])}",
                expected_cancel_generation=int(task["cancel_generation"]))
        elif task["status"] != "stopping":
            return

        if not str(task.get("cancel_id") or "").startswith("deadline:"):
            return
        lease = self._renew_lease_if_needed(lease, force=True)
        if not self._finish_stop(binding, task, lease):
            self._record_error(
                f"task {task['identity'].task_id} exceeded its deadline; stop remains pending")

    # ------------------------------------------------------------------ recovery
    def _inspect_abandoned_attempts(self, binding: HostedRoomBinding) -> None:
        for task in self._tasks(binding, "running"):
            if task["run_process_generation"] == self.process_generation:
                continue
            inspection = (
                self._inspect_local_recovery_session(task)
                if self._transport_for(binding, task) is self.rpc
                else self._inspect_recovery_session(binding, task))
            if inspection.terminal is not None:
                self._harvest_previous_attempt(binding, task, inspection.terminal)
            elif inspection.active:
                # The prior session still owns the turn: no lease contention, no duplicate prompt.
                raise state.LeaseHeldError("recovered session turn is still active")

    def _inspect_session(
        self, transport: InternalSessionRPC, task: Mapping[str, Any], session_id: str,
        *, read_history: bool) -> _RecoveryInspection:
        """Probe one resolved session: optional terminal receipt from history, then live info."""
        profile = task["payload"]["target_profile"]
        receipt = (
            self._terminal_from_history(transport, profile, session_id, task)
            if read_history else None)
        info = transport.info(**_session_kw(profile, session_id))
        self._report_pending_action(task, session_id=session_id, info=info)
        return _RecoveryInspection(
            terminal=receipt, active=_info_is_active_for(info, task["identity"]),
            status=str(info.get("status") or "") or None)

    def _inspect_recovery_session(
        self, binding: HostedRoomBinding, task: Mapping[str, Any]) -> _RecoveryInspection:
        profile, transport = task["payload"]["target_profile"], self._transport_for(binding, task)
        with self.turn_lock(profile):
            session_id = self._resume_exact(transport, task["identity"].room_id, profile)
            if session_id is None:
                return _NO_INSPECTION
            return self._inspect_session(transport, task, session_id, read_history=True)

    def _inspect_local_recovery_session(
        self,
        task: Mapping[str, Any],
    ) -> _RecoveryInspection:
        """Recover exact durable proof, then inspect live process state.

        Ordinary session history is a display projection and cannot prove which
        task generation authored a row. Terminal recovery therefore uses the
        private receipt written before callback publication; without one, an
        inactive abandoned attempt remains indeterminate for explicit retry.
        """

        receipt = _durable_terminal_receipt(
            state.get_terminal_receipt(
                self.db_path,
                task["identity"],
                execution_generation=int(task["execution_generation"]),
            )
        )
        if receipt is not None:
            return _RecoveryInspection(terminal=receipt, active=False, status=None)
        profile = task["payload"]["target_profile"]
        with self.turn_lock(profile):
            session = self.rpc.resolve_exact(
                profile=profile, title=room_session_title(task["identity"].room_id),
                source=ROOM_SESSION_SOURCE)
            if session is None:
                return _NO_INSPECTION
            return self._inspect_session(self.rpc, task, _session_id(session), read_history=False)

    def _reconcile_indeterminate(
        self, binding: HostedRoomBinding, lease: state.DriverLease) -> bool:
        unresolved = self._tasks(binding, "indeterminate")
        if not unresolved:
            self._set_blocked(binding.room_id, False)
            return False
        inspected = self._inspected_indeterminate_attempts
        for task in unresolved:
            attempt_key = (
                binding.room_id, task["identity"].task_id, int(task["execution_generation"]))
            is_local = self._transport_for(binding, task) is self.rpc
            if is_local and attempt_key not in inspected:
                inspection = self._inspect_local_recovery_session(task)
                inspected.add(attempt_key)
                if inspection.terminal is not None:
                    self._resolve_indeterminate(binding, task, lease, inspection.terminal)
                    inspected.discard(attempt_key)
                    continue
                if inspection.active:
                    self._set_blocked(binding.room_id, True)
                    return True
            deadline = self.indeterminate_defer_seconds + float(
                task.get("indeterminate_at") or task.get("updated_at") or task.get("created_at")
                or self.clock())
            inspection = _NO_INSPECTION
            if attempt_key not in inspected or self.clock() >= deadline:
                try:
                    if self._transport_for(binding, task) is not self.rpc:
                        inspection = self._inspect_recovery_session(binding, task)
                except Exception as exc:
                    self._record_error(
                        f"task {task['identity'].task_id} recovery probe failed: {exc}")
                inspected.add(attempt_key)
            if inspection.status == "cancelled":
                # Remote-probe resolutions are not republished here.
                self._fenced(
                    state.resolve_indeterminate_cancellation, binding, task, lease, publish=False,
                    cancel_id=f"remote-cancel:{task['execution_generation']}")
                inspected.discard(attempt_key)
                continue
            if inspection.terminal is not None:
                self._resolve_indeterminate(
                    binding, task, lease, inspection.terminal, publish=False)
                inspected.discard(attempt_key)
                continue
            if self.clock() < deadline:
                self._set_blocked(binding.room_id, True)
                return True
            deferred = self._fenced(
                state.defer_indeterminate_task, None, task, lease, reason="member_unavailable")
            inspected.discard(attempt_key)
            self._publish(binding, deferred)
        self._set_blocked(binding.room_id, False)
        return False

    def _harvest_previous_attempt(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], receipt: _TerminalReceipt
    ) -> None:
        previous_attempt = state.TaskAttempt(
            identity=task["identity"], execution_generation=task["execution_generation"],
            cancel_generation=task["cancel_generation"],
            lease=state.DriverLease(
                room_id=binding.room_id, gateway_id=task["run_gateway_id"],
                authority_epoch=binding.authority_epoch,
                process_generation=task["run_process_generation"],
                lease_generation=task["run_lease_generation"], expires_at=0.0))
        # Once the previous proof has expired there is deliberately no "trust this historical
        # output" escape hatch; fenced recovery leaves the task indeterminate for the user.
        with suppress(state.StaleLeaseError, state.StaleTaskError):
            state.settle_task(self.db_path, previous_attempt, **asdict(receipt), clock=self.clock)

    def _tasks(self, binding: HostedRoomBinding, status: str) -> list[dict[str, Any]]:
        return state.list_tasks(self.db_path, room_id=binding.room_id, status=status)

    def _binding_for_room(self, room_id: str) -> HostedRoomBinding | None:
        return next((b for b in self._rooms_provider() if b.room_id == room_id), None)

    def _transport_for(
        self, binding: HostedRoomBinding, task: Mapping[str, Any]) -> InternalSessionRPC:
        if self.transport_resolver is not None:
            return self.transport_resolver(binding, task)
        if self.rpc is None:
            raise RuntimeError("hosted room transport is unavailable")
        return self.rpc

    def _resolve_or_create(
        self, transport: InternalSessionRPC, profile: str, room_id: str, *, create: bool = True
    ) -> Mapping[str, Any] | None:
        """Resolve + resume the canonical room session; create it (or return None) when absent."""
        coords = {
            "profile": profile, "title": room_session_title(room_id), "source": ROOM_SESSION_SOURCE}
        session = transport.resolve_exact(**coords)
        if session is None:
            return transport.create(**coords) if create else None
        return transport.resume(**_session_kw(profile, _session_id(session)))

    def _settle_failure_if_current(self, attempt: state.TaskAttempt, exc: Exception) -> None:
        with suppress(state.DriverStateError, state.RoomUnavailableError):
            state.settle_task(
                self.db_path, attempt,
                settlement_id=f"failure:{attempt.identity.task_id}:{attempt.execution_generation}",
                status="failed", result={"error": str(exc)}, clock=self.clock)
        self._record_task_error(attempt, f"failed: {exc}")

    def _record_task_error(self, attempt: state.TaskAttempt, message: str) -> None:
        self._record_error(f"task {attempt.identity.task_id} {message}")

    def _record_error(self, message: str) -> None:
        with self._status_lock:
            self._last_error = message


def room_session_title(room_id: str) -> str:
    """Return the bounded canonical hidden-session title for one room."""
    title = f"{_ROOM_SESSION_TITLE_PREFIX}{room_id}"
    if len(title) <= _ROOM_SESSION_TITLE_MAX_CHARS:
        return title

    digest = hashlib.sha256(room_id.encode("utf-8")).hexdigest()
    room_prefix_chars = (
        _ROOM_SESSION_TITLE_MAX_CHARS
        - len(_ROOM_SESSION_TITLE_PREFIX)
        - 1
        - len(digest)
    )
    return f"{_ROOM_SESSION_TITLE_PREFIX}{room_id[:room_prefix_chars]}~{digest}"


def _member_id(task: Mapping[str, Any]) -> str:
    p = task.get("payload") or {}
    return str(p.get("target_member_id") or p.get("target_profile") or "")


def _session_id(session: Mapping[str, Any]) -> str:
    value = session.get("session_id", session.get("id"))
    if not isinstance(value, str) or not value:
        raise ValueError("session adapter returned no session_id")
    return value


def _truncate_utf8(value: Any, *, max_bytes: int) -> tuple[str, bool]:
    text, encoded = str(value or ""), str(value or "").encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    prefix = encoded[: max(0, max_bytes - len(_TERMINAL_TRUNCATION_NOTICE.encode("utf-8")))]
    while prefix:
        try:
            return prefix.decode("utf-8") + _TERMINAL_TRUNCATION_NOTICE, True
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return _TERMINAL_TRUNCATION_NOTICE.strip(), True


def _bounded_terminal_result(receipt: Mapping[str, Any]) -> dict[str, Any]:
    text, truncated = _truncate_utf8(receipt.get("text", ""), max_bytes=MAX_TERMINAL_TEXT_BYTES)
    error, error_truncated = _truncate_utf8(receipt.get("error", ""), max_bytes=4096)
    return {
        "message_id": receipt.get("message_id"), "text": text,
        **({"error": error} if error else {}),
        **({"truncated": True} if truncated or error_truncated else {})}


def _find_terminal_receipt(
    messages: Sequence[Mapping[str, Any]],
    identity: state.TaskIdentity,
    execution_generation: int,
) -> _TerminalReceipt | None:
    for message in reversed(tuple(messages)):
        if not isinstance(message, Mapping):
            continue
        if message.get("task_id") != identity.task_id:
            continue
        generation = message.get("execution_generation")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation != execution_generation
        ):
            continue
        status = message.get("status")
        if message.get("role") != "assistant" or status not in {"settled", "failed"}:
            continue
        settlement_id = (
            message.get("settlement_id")
            or message.get("message_id")
            or message.get("id")
        )
        if not isinstance(settlement_id, str) or not settlement_id:
            continue
        normalized = dict(message)
        if "text" not in normalized and "content" in normalized:
            normalized["text"] = normalized.get("content")
        return _TerminalReceipt(
            status=cast(state.TerminalStatus, status),
            settlement_id=settlement_id,
            result=_bounded_terminal_result(normalized),
        )
    return None


def _durable_terminal_receipt(
    receipt: Mapping[str, Any] | None,
) -> _TerminalReceipt | None:
    if receipt is None:
        return None
    status = receipt.get("status")
    if status not in {"settled", "failed"}:
        return None
    result = receipt.get("result")
    if not isinstance(result, Mapping):
        return None
    settlement_id = receipt.get("settlement_id")
    if not isinstance(settlement_id, str) or not settlement_id:
        return None
    return _TerminalReceipt(
        status=cast(state.TerminalStatus, status),
        settlement_id=settlement_id,
        result=dict(result),
    )


def _info_active(info: Mapping[str, Any]) -> bool:
    return bool(info.get("active", info.get("running", False)))


def _info_is_active_for(
    info: Mapping[str, Any], identity: state.TaskIdentity, *, require_exact: bool = False,
    execution_generation: int | None = None, member_id: str | None = None,
) -> bool:
    if execution_generation is not None:
        return _info_active(info) and info.get("hosted_task") == {
            **asdict(identity), "execution_generation": execution_generation, "member_id": member_id}
    accepted = (identity.task_id,) if require_exact else (None, identity.task_id)
    return _info_active(info) and info.get("task_id") in accepted


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import contextlib  # noqa: F401,E402

@contextlib.contextmanager
def null_turn_lock(_profile: str) -> Any:
    """Provide an explicit no-op lock for narrow embedding tests."""
    yield
# ---- END PLUGIN-COMPAT ----
