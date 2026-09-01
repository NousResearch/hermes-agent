"""Runtime adapter for gateway-owned hosted room turns.

The durable state machine lives in :mod:`gateway.hosted_room_driver`.  This
module owns the process-local worker and a deliberately small, injected session
adapter.  It does not import the gateway server, construct agents, or depend on
any client transport.

The adapter normalizes existing internal session RPCs into seven methods. A
future server integration can implement those methods with the in-process
handlers while tests use deterministic fakes and no models or network.

One bounded supervisor schedules independent room workers. Profile turn locks
still serialize Bots that share one profile, while a room waiting for approval
cannot stop unrelated rooms from progressing. Hosted member sessions reuse a
stable ``Group: <room_id>`` title, with a bounded digest form for long ids, so
migration preserves the canonical transcript instead of forking a conversation.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager, Protocol, cast

import psutil

from gateway import hosted_room_driver as state
from gateway import hosted_rooms

_CANCEL_ROUTE_RETRIES = 8
ROOM_SESSION_SOURCE = "bot_room"
MAX_TERMINAL_TEXT_BYTES = 64 * 1024
_ROOM_SESSION_TITLE_PREFIX = "Group: "
_ROOM_SESSION_TITLE_MAX_CHARS = 100
_TERMINAL_TRUNCATION_NOTICE = (
    "\n\n[Reply truncated. Ask the Bot to share the full result as a file.]"
)
_PROCESS_SUBMISSION_LOCK = threading.Lock()
_PROCESS_SUBMISSIONS: set[tuple[str, state.TaskIdentity, int]] = set()


def _submission_fence_key(
    db_path: Path | str,
    identity: state.TaskIdentity,
    execution_generation: int,
) -> tuple[str, state.TaskIdentity, int]:
    database = os.path.normcase(str(Path(db_path).resolve()))
    return database, identity, int(execution_generation)


def _register_process_submission(
    key: tuple[str, state.TaskIdentity, int],
) -> None:
    with _PROCESS_SUBMISSION_LOCK:
        _PROCESS_SUBMISSIONS.add(key)


def _unregister_process_submission(
    key: tuple[str, state.TaskIdentity, int],
) -> None:
    with _PROCESS_SUBMISSION_LOCK:
        _PROCESS_SUBMISSIONS.discard(key)


def _process_submission_is_registered(
    key: tuple[str, state.TaskIdentity, int],
) -> bool:
    with _PROCESS_SUBMISSION_LOCK:
        return key in _PROCESS_SUBMISSIONS


class RoomSessionIdentityUnavailableError(RuntimeError):
    """The durable room registry could not safely resolve a session title."""


class HostedRoomProfileUnavailableError(RuntimeError):
    """The target profile disappeared before a hosted turn could be admitted."""


class InternalSessionRPC(Protocol):
    """Normalized in-process session operations required by the room driver."""

    def resolve_exact(
        self, *, profile: str, title: str, source: str
    ) -> Mapping[str, Any] | None:
        """Return the exact titled session under ``profile``, if it exists."""

    def create(self, *, profile: str, title: str, source: str) -> Mapping[str, Any]:
        """Create a session without model or provider overrides."""

    def resume(
        self, *, profile: str, session_id: str, source: str
    ) -> Mapping[str, Any]:
        """Resume the canonical room session."""

    def submit(
        self,
        *,
        profile: str,
        session_id: str,
        prompt: str,
        source: str,
        task: state.TaskIdentity,
        execution_generation: int,
        on_terminal: Callable[[Mapping[str, Any]], None],
    ) -> Mapping[str, Any]:
        """Submit one fenced room turn and durably report its terminal result."""

    def history(
        self, *, profile: str, session_id: str, source: str
    ) -> Sequence[Mapping[str, Any]]:
        """Return normalized session messages in durable order."""

    def info(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]:
        """Return normalized live status for a session."""

    def interrupt_admitted(
        self,
        *,
        task: state.TaskIdentity,
        execution_generation: int,
        source: str,
    ) -> Mapping[str, Any] | None:
        """Interrupt the exact process-local admitted task proof, if present."""


def _owner_process_state(pid: int, start_time: int) -> state.OwnerProcessState:
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
        from gateway.status import get_process_start_time

        current_start = get_process_start_time(pid)
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


class HostedRoomRuntime:
    """Run queued hosted-room tasks independently of Desktop connections."""

    def __init__(
        self,
        *,
        db_path: Path | str,
        rooms: Iterable[HostedRoomBinding] | Callable[[], Iterable[HostedRoomBinding]],
        turn_lock: Callable[[str], ContextManager[Any]],
        rpc: InternalSessionRPC,
        profile_available: Callable[[str], bool] | None = None,
        prepare_room: Callable[[HostedRoomBinding], None] | None = None,
        publish_terminal: Callable[[HostedRoomBinding, Mapping[str, Any]], None]
        | None = None,
        pending_action: Callable[[str, str, Mapping[str, Any] | None], None]
        | None = None,
        clock: Callable[[], float] = time.time,
        lease_ttl_seconds: float = 30.0,
        poll_interval_seconds: float = 5.0,
        active_poll_interval_seconds: float = 0.25,
        turn_timeout_seconds: float = 1830.0,
        indeterminate_defer_seconds: float = 60.0,
        max_concurrent_rooms: int = 4,
        process_generation: str | None = None,
        process_pid: int | None = None,
        process_start_time: int | None = None,
        owner_liveness: state.OwnerLiveness | None = None,
    ) -> None:
        if lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if active_poll_interval_seconds <= 0:
            raise ValueError("active_poll_interval_seconds must be positive")
        if turn_timeout_seconds <= 0:
            raise ValueError("turn_timeout_seconds must be positive")
        if indeterminate_defer_seconds <= 0:
            raise ValueError("indeterminate_defer_seconds must be positive")
        if (
            isinstance(max_concurrent_rooms, bool)
            or not isinstance(max_concurrent_rooms, int)
            or max_concurrent_rooms < 1
        ):
            raise ValueError("max_concurrent_rooms must be a positive integer")
        self.db_path = Path(db_path)
        self.rpc = rpc
        self.turn_lock = turn_lock
        self.profile_available = profile_available or (lambda _profile: True)
        self.prepare_room = prepare_room
        self.publish_terminal = publish_terminal
        self.pending_action = pending_action
        self.clock = clock
        self.lease_ttl_seconds = float(lease_ttl_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.active_poll_interval_seconds = float(active_poll_interval_seconds)
        self.turn_timeout_seconds = float(turn_timeout_seconds)
        self.indeterminate_defer_seconds = float(indeterminate_defer_seconds)
        self.max_concurrent_rooms = max_concurrent_rooms
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
                from gateway.status import get_process_start_time

                process_start_time = get_process_start_time(self.process_pid)
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
        self._rooms_provider: Callable[[], Iterable[HostedRoomBinding]]
        if callable(rooms):
            self._rooms_provider = cast(
                Callable[[], Iterable[HostedRoomBinding]], rooms
            )
        else:
            room_bindings = tuple(rooms)
            self._rooms_provider = lambda: room_bindings

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._room_threads: dict[str, threading.Thread] = {}
        self._rooms_needing_reschedule: set[str] = set()
        self._leases: dict[str, state.DriverLease] = {}
        self._recovered_leases: set[tuple[str, int]] = set()
        self._inspected_indeterminate_attempts: set[tuple[str, str, int]] = set()
        self._ambiguous_rooms: dict[str, float] = {}
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

    def start(self) -> None:
        """Start the bounded room-worker supervisor idempotently."""
        with self._status_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._wake.set()
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="hosted-room-driver-supervisor",
                daemon=True,
            )
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
        return not thread.is_alive() and all(
            not room_thread.is_alive() for room_thread in room_threads
        )

    def wakeup(self) -> None:
        """Wake the worker after task admission or a room-state change."""
        with self._activity_condition:
            # If the supervisor observes this signal while a room still owns a
            # worker slot, remember to revisit it once that thread exits. This
            # closes the race between terminal publication/route repair and the
            # longer idle fallback without turning idle rooms into a busy loop.
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
                "running": bool(thread and thread.is_alive()),
                "stopping": self._stop.is_set(),
                "process_generation": self.process_generation,
                "current_task": current_tasks[0] if current_tasks else None,
                "current_tasks": current_tasks,
                "leased_rooms": tuple(sorted(self._leases)),
                "blocked_rooms": tuple(sorted(self._blocked_rooms)),
                "last_error": self._last_error,
                "cycles": self._cycles,
            }

    def cancel(
        self,
        identity: state.TaskIdentity,
        *,
        cancel_id: str,
    ) -> dict[str, Any]:
        """Persist a stop intent, then commit cancellation after acknowledgement.

        The worker thread transitions tasks concurrently with cancellation
        (queued -> running -> terminal), so the status read below is only a
        routing hint. Every fast-path failure caused by a concurrent
        transition re-reads and re-routes instead of surfacing a transient
        `InvalidTaskTransitionError`/`StaleTaskError` to the caller.
        """
        for _ in range(_CANCEL_ROUTE_RETRIES):
            before = state.get_task(self.db_path, identity)
            if before["status"] == "cancelled":
                return before
            if before["status"] in state.TERMINAL_STATUSES:
                # Completion already won before this Stop could establish a
                # fence. Return the durable outcome idempotently rather than
                # turning a harmless late Stop into a routing error.
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
                    # Lost the race with the worker; re-read and re-route.
                    continue
                self.wakeup()
                return cancelled
            try:
                stopping = state.begin_task_cancel(
                    self.db_path,
                    identity,
                    cancel_id=cancel_id,
                    expected_cancel_generation=before["cancel_generation"],
                    clock=self.clock,
                )
            except (state.InvalidTaskTransitionError, state.StaleTaskError):
                # Task settled or re-queued mid-flight; re-read and re-route.
                continue
            binding = self._binding_for_room(identity.room_id)
            try:
                if binding is not None:
                    lease = self._ensure_lease(binding)
                    receipt_exists = state.get_terminal_receipt(
                        self.db_path,
                        identity,
                        execution_generation=int(stopping["execution_generation"]),
                    ) is not None
                    if receipt_exists and self._settle_stopping_completion(
                        binding, stopping, lease
                    ):
                        stopping = state.get_task(self.db_path, identity)
                    else:
                        stopping = self._claim_stopping_owner(stopping, lease)
                    if (
                        stopping["status"] == "stopping"
                        and self._interrupt_stopping_task(binding, stopping)
                    ):
                        # A terminal callback can commit while Stop is in
                        # flight. Harvest again after acknowledgement; the
                        # state-layer cancellation commit also refuses to
                        # overwrite a matching durable receipt transactionally.
                        receipt_exists = state.get_terminal_receipt(
                            self.db_path,
                            identity,
                            execution_generation=int(
                                stopping["execution_generation"]
                            ),
                        ) is not None
                        if receipt_exists:
                            lease = self._ensure_lease(binding)
                        if receipt_exists and self._settle_stopping_completion(
                            binding, stopping, lease
                        ):
                            stopping = state.get_task(self.db_path, identity)
                        else:
                            try:
                                stopping = state.complete_task_cancel(
                                    self.db_path,
                                    identity,
                                    lease,
                                    cancel_id=cancel_id,
                                    expected_execution_generation=int(
                                        stopping["execution_generation"]
                                    ),
                                    expected_cancel_generation=stopping[
                                        "cancel_generation"
                                    ],
                                    clock=self.clock,
                                )
                            except state.TaskConflictError:
                                lease = self._ensure_lease(binding)
                                if not self._settle_stopping_completion(
                                    binding, stopping, lease
                                ):
                                    raise
                                stopping = state.get_task(self.db_path, identity)
            except Exception as exc:
                self._record_error(f"stop remains pending: {exc}")
            stopping = state.get_task(self.db_path, identity)
            self.wakeup()
            return stopping
        # Exhausted routing retries under sustained contention: surface the
        # live status honestly rather than a transient transition error.
        final = state.get_task(self.db_path, identity)
        if final["status"] == "cancelled":
            return final
        raise state.InvalidTaskTransitionError(
            f"cancel kept losing races with task transitions "
            f"(last observed state '{final['status']}')"
        )

    def retry_indeterminate(
        self,
        identity: state.TaskIdentity,
        *,
        require_active_source_event_seq: int | None = None,
    ) -> dict[str, Any]:
        """Explicitly retry one uncertain attempt under the current room lease."""
        task = state.get_task(self.db_path, identity)
        if task["status"] not in {"indeterminate", "deferred"}:
            raise state.InvalidTaskTransitionError(
                f"cannot retry task in state '{task['status']}'"
            )
        binding = self._binding_for_room(identity.room_id)
        if binding is None:
            raise state.RoomUnavailableError("hosted room is unavailable")
        lease = self._ensure_lease(binding)
        if task["status"] == "deferred":
            retried = state.requeue_deferred_task(
                self.db_path,
                identity,
                lease,
                expected_execution_generation=task["execution_generation"],
                expected_cancel_generation=task["cancel_generation"],
                require_active_source_event_seq=require_active_source_event_seq,
                clock=self.clock,
            )
            with self._status_lock:
                self._blocked_rooms.discard(identity.room_id)
            self.wakeup()
            return retried
        inspection = self._inspect_local_recovery_session(task)
        if inspection.active:
            with self._status_lock:
                self._blocked_rooms.add(identity.room_id)
            raise state.InvalidTaskTransitionError(
                "cannot retry while the original task attempt is still active"
            )
        retried = state.requeue_indeterminate_task(
            self.db_path,
            identity,
            lease,
            expected_execution_generation=task["execution_generation"],
            expected_cancel_generation=task["cancel_generation"],
            clock=self.clock,
        )
        with self._status_lock:
            self._blocked_rooms.discard(identity.room_id)
        self.wakeup()
        return retried

    def _claim_stopping_owner(
        self,
        task: Mapping[str, Any],
        lease: state.DriverLease,
    ) -> dict[str, Any]:
        return state.claim_stopping_task(
            self.db_path,
            task["identity"],
            lease,
            expected_execution_generation=int(task["execution_generation"]),
            expected_cancel_generation=int(task["cancel_generation"]),
            owner_liveness=self.owner_liveness,
            clock=self.clock,
        )

    def _interrupt_stopping_task(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
    ) -> bool:
        del binding
        if task["run_process_generation"] != self.process_generation:
            return False
        transport = self.rpc
        if transport is None:
            return False
        exact_submission = (
            task["identity"],
            int(task["execution_generation"]),
        )
        submission_key = _submission_fence_key(
            self.db_path,
            task["identity"],
            int(task["execution_generation"]),
        )
        with self._status_lock:
            submission_in_progress = (
                self._submitting_tasks.get(task["identity"].room_id)
                == exact_submission
            )
        submission_in_progress = (
            submission_in_progress
            or _process_submission_is_registered(submission_key)
        )
        result = transport.interrupt_admitted(
            task=task["identity"],
            execution_generation=int(task["execution_generation"]),
            source=ROOM_SESSION_SOURCE,
        )
        if result is None:
            return False
        if result.get("found") is False or result.get("status") == "absent":
            return not submission_in_progress
        if submission_in_progress and result.get("interrupted") is not True:
            return False
        if result.get("active") is False:
            return True
        return result.get("interrupted") is True or str(result.get("status") or "") in {
            "cancelled",
            "interrupted",
            "stopping",
        }

    def _settle_stopping_completion(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
        lease: state.DriverLease,
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
            return False
        settled = state.settle_stopping_task(
            self.db_path,
            task["identity"],
            lease,
            expected_execution_generation=int(task["execution_generation"]),
            expected_cancel_generation=int(task["cancel_generation"]),
            settlement_id=receipt.settlement_id,
            status=receipt.status,
            result=receipt.result,
            clock=self.clock,
        )
        if self.publish_terminal is not None:
            self.publish_terminal(binding, settled)
        return True

    def _report_pending_action(
        self,
        task: Mapping[str, Any],
        *,
        session_id: str,
        info: Mapping[str, Any],
    ) -> None:
        if self.pending_action is None:
            return
        payload = task.get("payload") or {}
        target_member_id = payload.get("target_member_id")
        legacy_member_id: str | None = None
        if target_member_id is None:
            target_profile = str(payload.get("target_profile") or "")
            room = hosted_rooms.room_state(
                self.db_path,
                room_id=task["identity"].room_id,
            )
            matching_members = [
                member
                for member in room.get("members") or ()
                if isinstance(member, Mapping)
                and str(member.get("profile") or "") == target_profile
            ]
            if len(matching_members) != 1:
                raise state.DriverStateError(
                    "legacy task target profile does not match the frozen room roster"
                )
            target_member_id = matching_members[0].get("member_id") or target_profile
            if target_member_id != target_profile:
                legacy_member_id = target_profile
        member_id = str(target_member_id or "")
        if not member_id:
            raise state.DriverStateError("task target member id is missing")
        approval = info.get("pending_approval") or info.get("approval")
        action = None
        if isinstance(approval, Mapping):
            safe_approval = dict(approval)
            choices = [
                choice
                for choice in safe_approval.get("choices") or ()
                if choice in {"once", "deny"}
            ]
            safe_approval["choices"] = choices or ["once", "deny"]
            action = {
                "kind": "approval",
                "task_id": task["identity"].task_id,
                "thread_id": task["identity"].thread_id,
                "turn_id": task["identity"].turn_id,
                "execution_generation": int(task["execution_generation"]),
                "run_id": info.get("run_id"),
                "session_id": session_id,
                "request_id": safe_approval.get("request_id"),
                "approval": safe_approval,
            }
        if action is not None and legacy_member_id is not None:
            state.migrate_legacy_approval_request_member(
                self.db_path,
                task["identity"],
                execution_generation=int(task["execution_generation"]),
                legacy_member_id=legacy_member_id,
                member_id=member_id,
                request_id=str(action.get("request_id") or ""),
                session_id=session_id,
                action={**action, "member_id": member_id},
                clock=self.clock,
            )
        self.pending_action(task["identity"].room_id, member_id, action)

    def _retry_stopping_tasks(
        self, binding: HostedRoomBinding, lease: state.DriverLease
    ) -> bool:
        pending = state.list_tasks(
            self.db_path,
            room_id=binding.room_id,
            status="stopping",
        )
        for task in pending:
            try:
                lease = self._renew_lease_if_needed(binding, lease)
                if self._settle_stopping_completion(binding, task, lease):
                    continue
                task = self._claim_stopping_owner(task, lease)
                if not self._interrupt_stopping_task(binding, task):
                    return True
                self._complete_acknowledged_stop(binding, task, lease)
            except Exception as exc:
                self._record_error(f"stop retry remains pending: {exc}")
                return True
        return False

    def _worker_loop(self) -> None:
        try:
            while not self._stop.is_set():
                # Clear before work so a write racing the cycle remains set and
                # causes an immediate follow-up pass rather than being lost.
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
                    room_threads = tuple(
                        thread
                        for thread in self._room_threads.values()
                        if thread.is_alive()
                    )
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
                room_id: thread
                for room_id, thread in self._room_threads.items()
                if thread.is_alive()
            }
            available = self.max_concurrent_rooms - len(self._room_threads)
            active_rooms = set(self._room_threads)
        if available <= 0:
            return

        bindings = tuple(self._rooms_provider())
        if not bindings:
            return
        start = self._room_schedule_cursor % len(bindings)
        ordered_bindings = bindings[start:] + bindings[:start]
        self._room_schedule_cursor = (start + 1) % len(bindings)

        for binding in ordered_bindings:
            if self._stop.is_set() or available <= 0:
                return
            if binding.room_id in active_rooms:
                continue
            room_thread = threading.Thread(
                target=self._run_room_once,
                args=(binding,),
                name=f"hosted-room-{binding.room_id[:24]}",
                daemon=True,
            )
            with self._status_lock:
                self._room_threads[binding.room_id] = room_thread
            active_rooms.add(binding.room_id)
            available -= 1
            room_thread.start()

    def _run_room_once(self, binding: HostedRoomBinding) -> None:
        try:
            self._process_room(binding)
        except state.LeaseHeldError:
            return
        except (state.RoomUnavailableError, state.StaleLeaseError) as exc:
            self._drop_lease(binding.room_id)
            with self._status_lock:
                self._blocked_rooms.discard(binding.room_id)
            self._record_error(f"room {binding.room_id}: {exc}")
        except Exception as exc:
            self._record_error(f"room {binding.room_id}: {exc}")
        finally:
            current = threading.current_thread()
            with self._status_lock:
                if self._room_threads.get(binding.room_id) is current:
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
            running = state.list_tasks(
                self.db_path,
                room_id=binding.room_id,
                status="running",
            )
            if not running:
                self._ambiguous_rooms.pop(binding.room_id, None)
            elif self.clock() < deferred_until:
                return
            else:
                self._ambiguous_rooms.pop(binding.room_id, None)
        lease = self._ensure_lease(binding)
        recovery_key = (lease.room_id, lease.lease_generation)
        if recovery_key not in self._recovered_leases:
            state.recover_room(self.db_path, lease, clock=self.clock)
            self._recovered_leases.add(recovery_key)
        if self._retry_stopping_tasks(binding, lease):
            with self._status_lock:
                self._blocked_rooms.add(binding.room_id)
            return
        if self._reconcile_indeterminate(binding, lease):
            return

        queued = state.list_tasks(
            self.db_path,
            room_id=binding.room_id,
            status="queued",
        )
        for task in queued:
            if self._stop.is_set():
                return
            profile = task["payload"]["target_profile"]
            # Keep the durable task queued while another room owns this
            # profile. A bounded lock timeout must not terminalize work that
            # never reached the model.
            with self.turn_lock(profile):
                lease = self._renew_lease_if_needed(binding, lease)
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

    def _ensure_lease(self, binding: HostedRoomBinding) -> state.DriverLease:
        with self._status_lock:
            current = self._leases.get(binding.room_id)
        if current is not None:
            try:
                renewed = self._renew_lease_if_needed(binding, current)
            except state.StaleLeaseError:
                self._drop_lease(binding.room_id)
            else:
                return renewed

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
            self._recovered_leases = {
                key for key in self._recovered_leases if key[0] != binding.room_id
            }
        return lease

    def _renew_lease_if_needed(
        self,
        binding: HostedRoomBinding,
        lease: state.DriverLease,
        *,
        force: bool = False,
    ) -> state.DriverLease:
        del binding
        renew_at = lease.expires_at - (self.lease_ttl_seconds / 2)
        if not force and self.clock() < renew_at:
            return lease
        renewed = state.renew_lease(
            self.db_path,
            lease,
            ttl_seconds=self.lease_ttl_seconds,
            clock=self.clock,
        )
        with self._status_lock:
            self._leases[lease.room_id] = renewed
        return renewed

    def _require_profile_available(self, profile: str) -> None:
        try:
            available = self.profile_available(profile)
        except Exception as exc:
            raise HostedRoomProfileUnavailableError(
                "hosted room target profile availability could not be verified"
            ) from exc
        if not available:
            raise HostedRoomProfileUnavailableError(
                "hosted room target profile is unavailable"
            )

    def _execute_attempt(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
        attempt: state.TaskAttempt,
        *,
        turn_lock_held: bool = False,
    ) -> None:
        profile = task["payload"]["target_profile"]
        transport = self.rpc
        submit_attempted = False
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
                contextlib.nullcontext()
                if turn_lock_held
                else self.turn_lock(profile)
            )
            with lock_context:
                self._require_profile_available(profile)
                session = self._resolve_or_create(transport, profile, binding.room_id)
                # An in-process submit should fail before admission or return
                # after it, but an unexpected exception at that boundary is
                # still ambiguous. Never terminalize it as a proven failure.
                self._require_profile_available(profile)
                submit_attempted = True

                def on_terminal(receipt: Mapping[str, Any]) -> None:
                    status = receipt.get("status")
                    if status == "cancelled":
                        self.wakeup()
                        return
                    terminal_status: state.TerminalStatus = (
                        "settled" if status == "settled" else "failed"
                    )
                    try:
                        settled = state.settle_task(
                            self.db_path,
                            attempt,
                            settlement_id=(
                                receipt.get("settlement_id")
                                or f"reply:{attempt.identity.task_id}:{attempt.execution_generation}"
                            ),
                            status=terminal_status,
                            result=_bounded_terminal_result(receipt),
                            clock=self.clock,
                        )
                        if self.publish_terminal is not None:
                            self.publish_terminal(binding, settled)
                    except (state.StaleLeaseError, state.StaleTaskError):
                        try:
                            current = state.get_task(self.db_path, attempt.identity)
                            if current["status"] == "stopping":
                                # The Stop path may run after this attempt's
                                # original lease expired. Re-acquire the
                                # current room lease before committing a
                                # completion that beat the unacknowledged Stop.
                                lease = self._ensure_lease(binding)
                                settled = state.settle_stopping_task(
                                    self.db_path,
                                    attempt.identity,
                                    lease,
                                    expected_execution_generation=attempt.execution_generation,
                                    expected_cancel_generation=int(
                                        current["cancel_generation"]
                                    ),
                                    settlement_id=(
                                        receipt.get("settlement_id")
                                        or f"reply:{attempt.identity.task_id}:{attempt.execution_generation}"
                                    ),
                                    status=terminal_status,
                                    result=_bounded_terminal_result(receipt),
                                    clock=self.clock,
                                )
                                if self.publish_terminal is not None:
                                    self.publish_terminal(binding, settled)
                        except (
                            state.LeaseHeldError,
                            state.StaleLeaseError,
                            state.StaleTaskError,
                        ):
                            pass
                    except state.DriverStateError as exc:
                        # A malformed terminal receipt must not escape the
                        # callback and hold the profile lock until the deadline.
                        self._settle_failure_if_current(
                            attempt,
                            RuntimeError(
                                f"terminal result could not be committed: {exc}"
                            ),
                        )
                    self.wakeup()

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
                try:
                    transport.submit(
                        profile=profile,
                        session_id=_session_id(session),
                        prompt=task["payload"]["prompt"],
                        source=ROOM_SESSION_SOURCE,
                        task=attempt.identity,
                        execution_generation=attempt.execution_generation,
                        on_terminal=on_terminal,
                    )
                finally:
                    _unregister_process_submission(submission_key)
                    with self._status_lock:
                        if (
                            self._submitting_tasks.get(binding.room_id)
                            == (attempt.identity, attempt.execution_generation)
                        ):
                            self._submitting_tasks.pop(binding.room_id, None)
                receipt = self._wait_for_terminal(
                    binding,
                    task=task,
                    profile=profile,
                    session_id=_session_id(session),
                    attempt=attempt,
                    transport=transport,
                    deadline_monotonic=deadline_monotonic,
                )
                if receipt is None:
                    return
                state.settle_task(
                    self.db_path,
                    attempt,
                    settlement_id=receipt.settlement_id,
                    status=receipt.status,
                    result=receipt.result,
                    clock=self.clock,
                )
        except HostedRoomProfileUnavailableError:
            try:
                deferred = state.defer_running_task(
                    self.db_path,
                    attempt,
                    reason="member_unavailable",
                    clock=self.clock,
                )
                if self.publish_terminal is not None:
                    self.publish_terminal(binding, deferred)
            except (state.StaleLeaseError, state.StaleTaskError) as exc:
                self._record_error(
                    f"task {attempt.identity.task_id} profile deferral fenced: {exc}"
                )
        except (state.StaleLeaseError, state.StaleTaskError) as exc:
            self._drop_lease(binding.room_id)
            self._record_error(f"task {attempt.identity.task_id} fenced: {exc}")
        except Exception as exc:
            if submit_attempted and getattr(exc, "not_admitted", False) is True:
                # The in-process RPC marks error envelopes returned before
                # prompt admission.  That proof is stronger than the generic
                # submit-boundary ambiguity rule, so settle it immediately.
                self._settle_failure_if_current(attempt, exc)
            elif submit_attempted:
                self._drop_lease(binding.room_id)
                self._ambiguous_rooms[binding.room_id] = attempt.lease.expires_at
                self._record_error(
                    f"task {attempt.identity.task_id} observation failed after submit: {exc}"
                )
            else:
                self._settle_failure_if_current(attempt, exc)
        finally:
            _unregister_process_submission(submission_key)
            with self._status_lock:
                self._current_tasks.pop(binding.room_id, None)
                if (
                    self._submitting_tasks.get(binding.room_id)
                    == (attempt.identity, attempt.execution_generation)
                ):
                    self._submitting_tasks.pop(binding.room_id, None)
                # The task may have published a reply, deferred a member, or
                # exposed the next turn while this room thread still occupied
                # its slot. Schedule exactly one immediate follow-up after the
                # thread leaves; idle room scans never set this marker.
                self._rooms_needing_reschedule.add(binding.room_id)

    def _wait_for_terminal(
        self,
        binding: HostedRoomBinding,
        *,
        task: Mapping[str, Any],
        profile: str,
        session_id: str,
        attempt: state.TaskAttempt,
        transport: InternalSessionRPC,
        deadline_monotonic: float,
    ) -> _TerminalReceipt | None:
        lease = attempt.lease
        while not self._stop.is_set():
            activity_generation = self._activity_snapshot()
            task = state.get_task(self.db_path, attempt.identity)
            if task["status"] in state.TERMINAL_STATUSES:
                return None
            if task["status"] == "stopping":
                try:
                    lease = self._renew_lease_if_needed(binding, lease)
                    if self._settle_stopping_completion(binding, task, lease):
                        return None
                    task = self._claim_stopping_owner(task, lease)
                    if self._interrupt_stopping_task(binding, task):
                        self._complete_acknowledged_stop(binding, task, lease)
                        return None
                except Exception as exc:
                    self._record_error(f"stop retry remains pending: {exc}")
                self._wait_for_activity(
                    activity_generation, self.active_poll_interval_seconds
                )
                continue

            if time.monotonic() >= deadline_monotonic:
                self._expire_attempt_deadline(binding, task, lease)
                return None

            lease = self._renew_lease_if_needed(binding, lease)
            receipt = _durable_terminal_receipt(
                state.get_terminal_receipt(
                    self.db_path,
                    attempt.identity,
                    execution_generation=attempt.execution_generation,
                )
            )
            if receipt is not None:
                return receipt

            info = transport.info(
                profile=profile,
                session_id=session_id,
                source=ROOM_SESSION_SOURCE,
            )
            self._report_pending_action(task, session_id=session_id, info=info)
            remaining = max(0.0, deadline_monotonic - time.monotonic())
            self._wait_for_activity(
                activity_generation,
                min(self.active_poll_interval_seconds, remaining),
            )
        return None

    @staticmethod
    def _deadline_cancel_id(task: Mapping[str, Any]) -> str:
        return f"deadline:{int(task['execution_generation'])}"

    @staticmethod
    def _is_deadline_stop(task: Mapping[str, Any]) -> bool:
        return str(task.get("cancel_id") or "").startswith("deadline:")

    def _settle_deadline_failure(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
        lease: state.DriverLease,
    ) -> dict[str, Any]:
        """Publish the explicit terminal outcome after exact Stop acknowledgement."""

        execution_generation = int(task["execution_generation"])
        settled = state.settle_stopping_task(
            self.db_path,
            task["identity"],
            lease,
            expected_execution_generation=execution_generation,
            expected_cancel_generation=int(task["cancel_generation"]),
            settlement_id=f"deadline:{execution_generation}",
            status="failed",
            result={
                "error": (
                    "This Group Chat turn exceeded its configured time limit and "
                    "was stopped."
                ),
                "reason_code": "turn_deadline_exceeded",
                "timeout_seconds": self.turn_timeout_seconds,
            },
            clock=self.clock,
        )
        if self.publish_terminal is not None:
            self.publish_terminal(binding, settled)
        return settled

    def _complete_acknowledged_stop(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
        lease: state.DriverLease,
    ) -> dict[str, Any]:
        if self._is_deadline_stop(task):
            return self._settle_deadline_failure(binding, task, lease)
        return state.complete_task_cancel(
            self.db_path,
            task["identity"],
            lease,
            cancel_id=task["cancel_id"],
            expected_execution_generation=int(task["execution_generation"]),
            expected_cancel_generation=task["cancel_generation"],
            clock=self.clock,
        )

    def _expire_attempt_deadline(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
        lease: state.DriverLease,
    ) -> None:
        """Fence, stop, and terminalize one exact attempt at its deadline."""

        if task["status"] == "running":
            task = state.begin_task_cancel(
                self.db_path,
                task["identity"],
                cancel_id=self._deadline_cancel_id(task),
                expected_cancel_generation=int(task["cancel_generation"]),
                clock=self.clock,
            )
        elif task["status"] != "stopping":
            return

        # A user Stop that won the race keeps its own cancellation semantics.
        if not self._is_deadline_stop(task):
            return
        lease = self._renew_lease_if_needed(binding, lease, force=True)
        if self._settle_stopping_completion(binding, task, lease):
            return
        task = self._claim_stopping_owner(task, lease)
        if self._interrupt_stopping_task(binding, task):
            self._complete_acknowledged_stop(binding, task, lease)
            return
        self._record_error(
            f"task {task['identity'].task_id} exceeded its deadline; stop remains pending"
        )

    def _inspect_abandoned_attempts(self, binding: HostedRoomBinding) -> None:
        running = state.list_tasks(
            self.db_path,
            room_id=binding.room_id,
            status="running",
        )
        for task in running:
            if task["run_process_generation"] == self.process_generation:
                continue
            inspection = self._inspect_local_recovery_session(task)
            if inspection.terminal is not None:
                self._harvest_previous_attempt(binding, task, inspection.terminal)
            elif inspection.active:
                # The prior session still owns the turn. Do not contend for its
                # lease or submit a duplicate prompt.
                raise state.LeaseHeldError("recovered session turn is still active")

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
                profile=profile,
                title=room_session_title(task["identity"].room_id),
                source=ROOM_SESSION_SOURCE,
            )
            if session is None:
                return _RecoveryInspection(terminal=None, active=False, status=None)
            session_id = _session_id(session)
            info = self.rpc.info(
                profile=profile,
                session_id=session_id,
                source=ROOM_SESSION_SOURCE,
            )
            self._report_pending_action(task, session_id=session_id, info=info)
            return _RecoveryInspection(
                terminal=None,
                active=_info_is_active_for(info, task["identity"]),
                status=str(info.get("status") or "") or None,
            )

    def _reconcile_indeterminate(
        self,
        binding: HostedRoomBinding,
        lease: state.DriverLease,
    ) -> bool:
        unresolved = state.list_tasks(
            self.db_path,
            room_id=binding.room_id,
            status="indeterminate",
        )
        if not unresolved:
            with self._status_lock:
                self._blocked_rooms.discard(binding.room_id)
            return False
        for task in unresolved:
            generation = int(task["execution_generation"])
            attempt_key = (
                binding.room_id,
                task["identity"].task_id,
                generation,
            )
            if attempt_key not in self._inspected_indeterminate_attempts:
                inspection = self._inspect_local_recovery_session(task)
                self._inspected_indeterminate_attempts.add(attempt_key)
                if inspection.terminal is not None:
                    resolved = state.resolve_indeterminate_task(
                        self.db_path,
                        task["identity"],
                        lease,
                        expected_execution_generation=generation,
                        expected_cancel_generation=task["cancel_generation"],
                        settlement_id=inspection.terminal.settlement_id,
                        status=inspection.terminal.status,
                        result=inspection.terminal.result,
                        clock=self.clock,
                    )
                    self._inspected_indeterminate_attempts.discard(attempt_key)
                    if self.publish_terminal is not None:
                        self.publish_terminal(binding, resolved)
                    continue
                if inspection.active:
                    with self._status_lock:
                        self._blocked_rooms.add(binding.room_id)
                    return True
            deferred_at = float(
                task.get("indeterminate_at")
                or task.get("updated_at")
                or task.get("created_at")
                or self.clock()
            )
            if self.clock() < deferred_at + self.indeterminate_defer_seconds:
                with self._status_lock:
                    self._blocked_rooms.add(binding.room_id)
                return True
            deferred = state.defer_indeterminate_task(
                self.db_path,
                task["identity"],
                lease,
                expected_execution_generation=task["execution_generation"],
                expected_cancel_generation=task["cancel_generation"],
                reason="member_unavailable",
                clock=self.clock,
            )
            self._report_pending_action(task, session_id="", info={})
            if self.publish_terminal is not None:
                self.publish_terminal(binding, deferred)
        with self._status_lock:
            self._blocked_rooms.discard(binding.room_id)
        return False

    def _harvest_previous_attempt(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
        receipt: _TerminalReceipt,
    ) -> None:
        previous_lease = state.DriverLease(
            room_id=binding.room_id,
            gateway_id=task["run_gateway_id"],
            authority_epoch=binding.authority_epoch,
            process_generation=task["run_process_generation"],
            lease_generation=task["run_lease_generation"],
            expires_at=0.0,
        )
        previous_attempt = state.TaskAttempt(
            identity=task["identity"],
            lease=previous_lease,
            execution_generation=task["execution_generation"],
            cancel_generation=task["cancel_generation"],
        )
        try:
            state.settle_task(
                self.db_path,
                previous_attempt,
                settlement_id=receipt.settlement_id,
                status=receipt.status,
                result=receipt.result,
                clock=self.clock,
            )
        except (state.StaleLeaseError, state.StaleTaskError):
            # Once the previous proof has expired, the current state contract
            # deliberately offers no unsafe "trust this historical output"
            # escape hatch. The subsequent fenced recovery leaves it
            # indeterminate for explicit user action.
            return

    def _binding_for_room(self, room_id: str) -> HostedRoomBinding | None:
        return next(
            (
                binding
                for binding in self._rooms_provider()
                if binding.room_id == room_id
            ),
            None,
        )

    def _resolve_or_create(
        self,
        transport: InternalSessionRPC,
        profile: str,
        room_id: str,
    ) -> Mapping[str, Any]:
        title = room_session_title(room_id)
        session = transport.resolve_exact(
            profile=profile,
            title=title,
            source=ROOM_SESSION_SOURCE,
        )
        if session is None:
            return transport.create(
                profile=profile,
                title=title,
                source=ROOM_SESSION_SOURCE,
            )
        return transport.resume(
            profile=profile,
            session_id=_session_id(session),
            source=ROOM_SESSION_SOURCE,
        )

    def _settle_failure_if_current(
        self, attempt: state.TaskAttempt, exc: Exception
    ) -> None:
        try:
            state.settle_task(
                self.db_path,
                attempt,
                settlement_id=f"failure:{attempt.identity.task_id}:{attempt.execution_generation}",
                status="failed",
                result={"error": str(exc)},
                clock=self.clock,
            )
        except (state.DriverStateError, state.RoomUnavailableError):
            pass
        self._record_error(f"task {attempt.identity.task_id} failed: {exc}")

    def _record_error(self, message: str) -> None:
        with self._status_lock:
            self._last_error = message

    def _drop_lease(self, room_id: str) -> None:
        with self._status_lock:
            self._leases.pop(room_id, None)

    def _release_idle_leases(self) -> None:
        for room_id, lease in tuple(self._leases.items()):
            try:
                state.release_lease(self.db_path, lease, clock=self.clock)
            except state.DriverStateError:
                continue
            self._drop_lease(room_id)


def room_session_title(room_id: str) -> str:
    """Return the canonical hidden session title for one hosted room."""
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


def recover_room_id_from_session_title(
    db_path: Path | str,
    title: str,
) -> str | None:
    """Resolve a canonical title against the durable active-room registry.

    Long room ids cannot be recovered by reversing their bounded digest title.
    The title is therefore only a lookup key: identity comes from an exact
    canonical-title match against room ids stored in the shared room database.
    Missing registries and unmatched legacy display titles return ``None``;
    unavailable or ambiguous registries fail closed.
    """
    if not isinstance(title, str) or not title.startswith(
        _ROOM_SESSION_TITLE_PREFIX
    ):
        return None
    path = Path(db_path)
    if not path.is_file():
        return None

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=0.05,
        )
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='hosted_rooms' LIMIT 1"
        ).fetchone()
        if table is None:
            return None
        room_ids = (
            str(row[0])
            for row in conn.execute(
                "SELECT room_id FROM hosted_rooms WHERE disbanded_at IS NULL"
            ).fetchall()
        )
        matches = [
            room_id
            for room_id in room_ids
            if room_session_title(room_id) == title
        ]
    except sqlite3.Error as exc:
        raise RoomSessionIdentityUnavailableError(
            "hosted room session identity is temporarily unavailable"
        ) from exc
    finally:
        if conn is not None:
            conn.close()

    if len(matches) > 1:
        raise RoomSessionIdentityUnavailableError(
            "hosted room session title is ambiguous"
        )
    return matches[0] if matches else None


def _session_id(session: Mapping[str, Any]) -> str:
    value = session.get("session_id", session.get("id"))
    if not isinstance(value, str) or not value:
        raise ValueError("session adapter returned no session_id")
    return value


def _truncate_utf8(value: Any, *, max_bytes: int) -> tuple[str, bool]:
    text = str(value or "")
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    suffix = _TERMINAL_TRUNCATION_NOTICE.encode("utf-8")
    prefix = encoded[: max(0, max_bytes - len(suffix))]
    while prefix:
        try:
            return prefix.decode("utf-8") + _TERMINAL_TRUNCATION_NOTICE, True
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return _TERMINAL_TRUNCATION_NOTICE.strip(), True


def _bounded_terminal_result(receipt: Mapping[str, Any]) -> dict[str, Any]:
    text, truncated = _truncate_utf8(
        receipt.get("text", ""),
        max_bytes=MAX_TERMINAL_TEXT_BYTES,
    )
    error, error_truncated = _truncate_utf8(
        receipt.get("error", ""),
        max_bytes=4096,
    )
    return {
        "message_id": receipt.get("message_id"),
        "text": text,
        **({"error": error} if error else {}),
        **({"truncated": True} if truncated or error_truncated else {}),
    }


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


def _info_is_active_for(
    info: Mapping[str, Any],
    identity: state.TaskIdentity,
    *,
    require_exact: bool = False,
) -> bool:
    if not bool(info.get("active", info.get("running", False))):
        return False
    active_task_id = info.get("task_id")
    if require_exact:
        return active_task_id == identity.task_id
    return active_task_id in {None, identity.task_id}


@contextlib.contextmanager
def null_turn_lock(_profile: str) -> Any:
    """Provide an explicit no-op lock for narrow embedding tests."""
    yield
