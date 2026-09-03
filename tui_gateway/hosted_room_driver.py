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
cannot stop unrelated rooms from progressing. Hosted member sessions
intentionally reuse ``Group: <room_id>`` so a local-to-hosted migration
preserves the same canonical transcript instead of forking a second conversation.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager, Protocol, cast

from gateway import hosted_room_driver as state

_CANCEL_ROUTE_RETRIES = 8


ROOM_SESSION_SOURCE = "bot_room"
MAX_TERMINAL_TEXT_BYTES = 64 * 1024
_TERMINAL_TRUNCATION_NOTICE = (
    "\n\n[Reply truncated. Ask the Bot to share the full result as a file.]"
)


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

    def interrupt(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        expected_task_id: str,
    ) -> Mapping[str, Any] | None:
        """Interrupt only when the current turn still matches the expected task."""


class MemberTransportResolver(Protocol):
    """Resolve the session transport for one durable room task."""

    def __call__(
        self,
        binding: "HostedRoomBinding",
        task: Mapping[str, Any],
    ) -> InternalSessionRPC:
        """Return a local or peer transport without changing task identity."""


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
        rpc: InternalSessionRPC | None = None,
        transport_resolver: MemberTransportResolver | None = None,
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
        unavailable_retry_min_seconds: float = 1.0,
        unavailable_retry_max_seconds: float = 30.0,
        process_generation: str | None = None,
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
        if not (
            unavailable_retry_min_seconds > 0
            and unavailable_retry_max_seconds >= unavailable_retry_min_seconds
        ):
            raise ValueError("unavailable retry bounds are invalid")
        if rpc is None and transport_resolver is None:
            raise ValueError("rpc or transport_resolver is required")
        self.db_path = Path(db_path)
        self.rpc = rpc
        self.transport_resolver = transport_resolver
        self.turn_lock = turn_lock
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
        self.unavailable_retry_min_seconds = float(unavailable_retry_min_seconds)
        self.unavailable_retry_max_seconds = float(unavailable_retry_max_seconds)
        self.process_generation = process_generation or uuid.uuid4().hex
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
        self._unavailable_route_retries: dict[
            tuple[str, str], dict[str, float]
        ] = {}
        self._blocked_rooms: set[str] = set()
        self._inspected_indeterminate_attempts: set[tuple[str, str, int]] = set()
        self._status_lock = threading.Lock()
        self._current_tasks: dict[str, state.TaskIdentity] = {}
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
        with self._status_lock:
            # If the supervisor observes this signal while a room still owns a
            # worker slot, remember to revisit it once that thread exits. This
            # closes the race between terminal publication/route repair and the
            # longer idle fallback without turning idle rooms into a busy loop.
            self._rooms_needing_reschedule.update(self._room_threads)
        self._wake.set()

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
                raise state.InvalidTaskTransitionError(
                    f"cannot cancel task in state '{before['status']}'"
                )
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
                    if self._peer_stop_acknowledged(binding, stopping):
                        stopping = state.complete_task_cancel(
                            self.db_path,
                            identity,
                            cancel_id=cancel_id,
                            expected_cancel_generation=stopping["cancel_generation"],
                            clock=self.clock,
                        )
                    elif not self._settle_stopping_completion(
                        binding, stopping, lease
                    ):
                        if self._interrupt_stopping_task(binding, stopping):
                            stopping = state.complete_task_cancel(
                                self.db_path,
                                identity,
                                cancel_id=cancel_id,
                                expected_cancel_generation=stopping[
                                    "cancel_generation"
                                ],
                                clock=self.clock,
                            )
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

    def _peer_stop_acknowledged(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
    ) -> bool:
        """Probe a peer's exact durable terminal status before reading history."""

        transport = self._transport_for(binding, task)
        if transport is None or transport is self.rpc:
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
        info = transport.info(
            profile=profile,
            session_id=_session_id(resumed),
            source=ROOM_SESSION_SOURCE,
        )
        return self._info_acknowledges_peer_cancel(info, task)

    def retry_indeterminate(self, identity: state.TaskIdentity) -> dict[str, Any]:
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
                clock=self.clock,
            )
            with self._status_lock:
                self._blocked_rooms.discard(identity.room_id)
            self.wakeup()
            return retried
        # Explicit Retry may resume the exact stored session. Use the returned
        # runtime id for every subsequent history/info probe; an automatic
        # abandoned-attempt scan remains non-resuming for local sessions.
        inspection = self._inspect_recovery_session(binding, task)
        if inspection.terminal is not None:
            resolved = state.resolve_indeterminate_task(
                self.db_path,
                identity,
                lease,
                expected_execution_generation=task["execution_generation"],
                expected_cancel_generation=task["cancel_generation"],
                settlement_id=inspection.terminal.settlement_id,
                status=inspection.terminal.status,
                result=inspection.terminal.result,
                clock=self.clock,
            )
            if self.publish_terminal is not None:
                self.publish_terminal(binding, resolved)
            return resolved
        if inspection.status == "cancelled":
            resolved = state.resolve_indeterminate_cancellation(
                self.db_path,
                identity,
                lease,
                expected_execution_generation=task["execution_generation"],
                expected_cancel_generation=task["cancel_generation"],
                cancel_id=f"remote-cancel:{task['execution_generation']}",
                clock=self.clock,
            )
            if self.publish_terminal is not None:
                self.publish_terminal(binding, resolved)
            return resolved
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

    def _interrupt_stopping_task(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
    ) -> bool:
        transport = self._transport_for(binding, task)
        if transport is None:
            return False
        profile = task["payload"]["target_profile"]
        session = transport.resolve_exact(
            profile=profile,
            title=room_session_title(binding.room_id),
            source=ROOM_SESSION_SOURCE,
        )
        if session is None:
            # A local accepted turn cannot survive without its canonical
            # session. Resolution errors raise; an authoritative absence is a
            # safe Stop acknowledgement. A peer remains uncertain instead.
            return transport is self.rpc
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
            # History was checked immediately before this probe. An exact
            # local session that is no longer active cannot keep executing, and
            # after a restart its process-local task marker is expected to be
            # absent.
            return True
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

    def _settle_stopping_completion(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
        lease: state.DriverLease,
    ) -> bool:
        """Publish a terminal receipt that arrived before Stop was acknowledged."""
        transport = self._transport_for(binding, task)
        if transport is None:
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
        member_id = str(
            payload.get("target_member_id") or payload.get("target_profile") or ""
        )
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
                "execution_generation": int(task["execution_generation"]),
                "run_id": info.get("run_id"),
                "session_id": session_id,
                "request_id": safe_approval.get("request_id"),
                "approval": safe_approval,
            }
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
                if self._peer_stop_acknowledged(binding, task):
                    state.complete_task_cancel(
                        self.db_path,
                        task["identity"],
                        cancel_id=task["cancel_id"],
                        expected_cancel_generation=task["cancel_generation"],
                        clock=self.clock,
                    )
                    continue
                if self._settle_stopping_completion(binding, task, lease):
                    continue
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
            if self._route_retry_is_deferred(task):
                return
            lease = self._renew_lease_if_needed(binding, lease)
            attempt = state.start_task(
                self.db_path,
                task["identity"],
                lease,
                expected_cancel_generation=task["cancel_generation"],
                clock=self.clock,
            )
            self._execute_attempt(binding, task, attempt)
            current = state.get_task(self.db_path, task["identity"])
            if current["status"] not in state.TERMINAL_STATUSES:
                return

    @staticmethod
    def _route_retry_key(task: Mapping[str, Any]) -> tuple[str, str]:
        payload = task.get("payload") or {}
        member_id = str(
            payload.get("target_member_id") or payload.get("target_profile") or ""
        )
        return task["identity"].room_id, member_id

    def _route_retry_is_deferred(self, task: Mapping[str, Any]) -> bool:
        retry = self._unavailable_route_retries.get(self._route_retry_key(task))
        return retry is not None and self.clock() < retry["next_attempt_at"]

    def _defer_unavailable_route(self, task: Mapping[str, Any]) -> float:
        key = self._route_retry_key(task)
        previous = self._unavailable_route_retries.get(key)
        delay = (
            self.unavailable_retry_min_seconds
            if previous is None
            else min(
                self.unavailable_retry_max_seconds,
                max(
                    self.unavailable_retry_min_seconds,
                    previous["delay"] * 2,
                ),
            )
        )
        self._unavailable_route_retries[key] = {
            "delay": delay,
            "next_attempt_at": self.clock() + delay,
        }
        return delay

    def _clear_unavailable_route_retry(self, task: Mapping[str, Any]) -> None:
        self._unavailable_route_retries.pop(self._route_retry_key(task), None)

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

    def _execute_attempt(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
        attempt: state.TaskAttempt,
    ) -> None:
        profile = task["payload"]["target_profile"]
        transport = self._transport_for(binding, task)
        submit_attempted = False
        with self._status_lock:
            self._current_tasks[binding.room_id] = attempt.identity
        try:
            with self.turn_lock(profile):
                session = self._resolve_or_create(transport, profile, binding.room_id)
                # An in-process submit should fail before admission or return
                # after it, but an unexpected exception at that boundary is
                # still ambiguous. Never terminalize it as a proven failure.
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
                    except state.StaleTaskError:
                        try:
                            current = state.get_task(self.db_path, attempt.identity)
                            if current["status"] == "stopping":
                                settled = state.settle_stopping_task(
                                    self.db_path,
                                    attempt.identity,
                                    attempt.lease,
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
                        except (state.StaleLeaseError, state.StaleTaskError):
                            pass
                    except state.StaleLeaseError:
                        # Cancellation, disband, or authority transfer won the
                        # durable race. The model result is intentionally
                        # discarded; never turn a correct fence into a worker
                        # thread exception.
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
                transport.submit(
                    profile=profile,
                    session_id=_session_id(session),
                    prompt=task["payload"]["prompt"],
                    source=ROOM_SESSION_SOURCE,
                    task=attempt.identity,
                    execution_generation=attempt.execution_generation,
                    on_terminal=on_terminal,
                )
                self._clear_unavailable_route_retry(task)
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
        except (state.StaleLeaseError, state.StaleTaskError) as exc:
            self._drop_lease(binding.room_id)
            self._record_error(f"task {attempt.identity.task_id} fenced: {exc}")
        except Exception as exc:
            if submit_attempted and bool(getattr(exc, "not_admitted", False)):
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
                self._drop_lease(binding.room_id)
                self._ambiguous_rooms[binding.room_id] = attempt.lease.expires_at
                self._record_error(
                    f"task {attempt.identity.task_id} observation failed after submit: {exc}"
                )
            else:
                self._settle_failure_if_current(attempt, exc)
        finally:
            with self._status_lock:
                self._current_tasks.pop(binding.room_id, None)
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
            task = state.get_task(self.db_path, attempt.identity)
            if task["status"] in state.TERMINAL_STATUSES:
                return None
            if task["status"] == "stopping":
                try:
                    lease = self._renew_lease_if_needed(binding, lease)
                    if self._settle_stopping_completion(binding, task, lease):
                        return None
                    if self._interrupt_stopping_task(binding, task):
                        self._complete_acknowledged_stop(binding, task, lease)
                        return None
                except Exception as exc:
                    self._record_error(f"stop retry remains pending: {exc}")
                self._wake.wait(self.active_poll_interval_seconds)
                self._wake.clear()
                continue

            if time.monotonic() >= deadline_monotonic:
                self._expire_attempt_deadline(binding, task, lease)
                return None

            lease = self._renew_lease_if_needed(binding, lease)
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

            info = transport.info(
                profile=profile,
                session_id=session_id,
                source=ROOM_SESSION_SOURCE,
            )
            self._report_pending_action(task, session_id=session_id, info=info)
            remaining = max(0.0, deadline_monotonic - time.monotonic())
            self._wake.wait(min(self.active_poll_interval_seconds, remaining))
            self._wake.clear()
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
            cancel_id=task["cancel_id"],
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
            transport = self._transport_for(binding, task)
            inspection = (
                self._inspect_local_recovery_session(task)
                if transport is self.rpc
                else self._inspect_recovery_session(binding, task)
            )
            if inspection.terminal is not None:
                self._harvest_previous_attempt(binding, task, inspection.terminal)
            elif inspection.active:
                # The prior session still owns the turn. Do not contend for its
                # lease or submit a duplicate prompt.
                raise state.LeaseHeldError("recovered session turn is still active")

    def _inspect_recovery_session(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
    ) -> _RecoveryInspection:
        profile = task["payload"]["target_profile"]
        transport = self._transport_for(binding, task)
        with self.turn_lock(profile):
            session = transport.resolve_exact(
                profile=profile,
                title=room_session_title(task["identity"].room_id),
                source=ROOM_SESSION_SOURCE,
            )
            if session is None:
                return _RecoveryInspection(terminal=None, active=False, status=None)
            resumed = transport.resume(
                profile=profile,
                session_id=_session_id(session),
                source=ROOM_SESSION_SOURCE,
            )
            session_id = _session_id(resumed)
            receipt = _find_terminal_receipt(
                transport.history(
                    profile=profile,
                    session_id=session_id,
                    source=ROOM_SESSION_SOURCE,
                ),
                task["identity"],
                task["execution_generation"],
            )
            info = transport.info(
                profile=profile,
                session_id=session_id,
                source=ROOM_SESSION_SOURCE,
            )
            self._report_pending_action(task, session_id=session_id, info=info)
            return _RecoveryInspection(
                terminal=receipt,
                active=_info_is_active_for(info, task["identity"]),
                status=str(info.get("status") or "") or None,
            )

    def _inspect_local_recovery_session(
        self,
        task: Mapping[str, Any],
    ) -> _RecoveryInspection:
        """Check only live process state before explicit local recovery.

        A restart loses the in-process terminal callback identity. The ordinary
        session history is a display projection and cannot prove which durable
        task attempt authored a row, so never hydrate or infer completion from
        it. An inactive abandoned attempt remains indeterminate until the user
        explicitly retries it under a new fenced generation.
        """

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
        with self._status_lock:
            if not unresolved:
                self._blocked_rooms.discard(binding.room_id)
                return False
        for task in unresolved:
            generation = int(task["execution_generation"])
            attempt_key = (
                binding.room_id,
                task["identity"].task_id,
                generation,
            )
            if (
                self._transport_for(binding, task) is self.rpc
                and attempt_key not in self._inspected_indeterminate_attempts
            ):
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
            deadline = deferred_at + self.indeterminate_defer_seconds
            should_inspect = (
                attempt_key not in self._inspected_indeterminate_attempts
                or self.clock() >= deadline
            )
            inspection = _RecoveryInspection(terminal=None, active=False, status=None)
            if should_inspect:
                try:
                    if self._transport_for(binding, task) is not self.rpc:
                        inspection = self._inspect_recovery_session(binding, task)
                except Exception as exc:
                    self._record_error(
                        f"task {task['identity'].task_id} recovery probe failed: {exc}"
                    )
                self._inspected_indeterminate_attempts.add(attempt_key)
            if inspection.status == "cancelled":
                state.resolve_indeterminate_cancellation(
                    self.db_path,
                    task["identity"],
                    lease,
                    expected_execution_generation=task["execution_generation"],
                    expected_cancel_generation=task["cancel_generation"],
                    cancel_id=f"remote-cancel:{task['execution_generation']}",
                    clock=self.clock,
                )
                self._inspected_indeterminate_attempts.discard(attempt_key)
                continue
            if inspection.terminal is not None:
                state.resolve_indeterminate_task(
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
                continue
            if self.clock() < deadline:
                with self._status_lock:
                    self._blocked_rooms.add(binding.room_id)
                return True
            deferred = state.defer_indeterminate_task(
                self.db_path,
                task["identity"],
                lease,
                expected_execution_generation=generation,
                expected_cancel_generation=task["cancel_generation"],
                reason="member_unavailable",
                clock=self.clock,
            )
            self._inspected_indeterminate_attempts.discard(attempt_key)
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

    def _transport_for(
        self,
        binding: HostedRoomBinding,
        task: Mapping[str, Any],
    ) -> InternalSessionRPC:
        if self.transport_resolver is not None:
            return self.transport_resolver(binding, task)
        if self.rpc is None:
            raise RuntimeError("hosted room transport is unavailable")
        return self.rpc

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
    return f"Group: {room_id}"


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


def _find_terminal_receipt(
    history: Sequence[Mapping[str, Any]],
    identity: state.TaskIdentity,
    execution_generation: int,
) -> _TerminalReceipt | None:
    for message in reversed(history):
        if message.get("task_id") != identity.task_id:
            continue
        if message.get("execution_generation") != execution_generation:
            continue
        if message.get("role") != "assistant":
            continue
        status = message.get("status")
        if status not in {"settled", "failed"}:
            continue
        terminal_status = cast(state.TerminalStatus, status)
        receipt_id = message.get("message_id")
        if not isinstance(receipt_id, str) or not receipt_id:
            receipt_id = f"reply:{identity.task_id}:{execution_generation}"
        return _TerminalReceipt(
            status=terminal_status,
            settlement_id=receipt_id,
            result=_bounded_terminal_result(
                {
                    "message_id": receipt_id,
                    "text": message.get("content", ""),
                }
            ),
        )
    return None


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
