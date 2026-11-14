"""Runtime adapter for gateway-owned hosted room turns.

The durable state machine lives in :mod:`gateway.hosted_room_driver`; this module owns the
process-local worker and an injected session adapter, never the gateway server or agents.
One bounded supervisor schedules independent room workers: profile turn locks serialize Bots
sharing a profile while a room waiting for approval cannot stall unrelated rooms. Member
sessions reuse ``Group: <room_id>`` so a local-to-hosted migration keeps one transcript.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ContextManager, Protocol, cast

from gateway import hosted_room_driver as state
from tui_gateway.hosted_room_lease_keeper import LeaseKeeper
from gateway.hosted_room_lead import LOST_ROUTING_CANCEL_PREFIX

logger = logging.getLogger(__name__)

_CANCEL_ROUTE_RETRIES = 8
_STOP_ACK_STATUSES = {"cancelled", "interrupted"}

ROOM_SESSION_SOURCE = "bot_room"
MAX_TERMINAL_TEXT_BYTES = 64 * 1024
_TERMINAL_TRUNCATION_NOTICE = (
    "\n\n[Reply truncated. Ask the Bot to share the full result as a file.]")
_STOP_PENDING = "stop retry remains pending: {exc}"


class AttachmentsUnsupportedError(RuntimeError):
    """The member's route cannot carry the turn's files; refused before submit."""


class InternalSessionRPC(Protocol):
    """Normalized in-process session operations required by the room driver.

    ``submit`` durably reports one fenced turn's terminal result via ``on_terminal``;
    ``interrupt`` acts only while the current turn still matches ``expected_task_id``.
    Only a transport with ``accepts_attachments`` is handed a turn's bound files.
    """

    accepts_attachments: bool = False

    def resolve_exact(
        self, *, profile: str, title: str, source: str) -> Mapping[str, Any] | None: ...
    def create(self, *, profile: str, title: str, source: str) -> Mapping[str, Any]: ...
    def resume(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]: ...
    def submit(
        self, *, profile: str, session_id: str, prompt: str, source: str, task: state.TaskIdentity,
        execution_generation: int, on_terminal: Callable[[Mapping[str, Any]], None], member_id: str,
        attachments: Sequence[Mapping[str, Any]] | None = None,
    ) -> Mapping[str, Any]: ...
    def history(
        self, *, profile: str, session_id: str, source: str) -> Sequence[Mapping[str, Any]]: ...
    def info(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]: ...
    def interrupt(
        self, *, profile: str, session_id: str, source: str, expected_task_id: str,
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
    progress_at: float | None = None
    receipts_error: Exception | None = None  # the store may hold a reply this probe could not read


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
        prepare_room: Callable[[HostedRoomBinding], None] | None = None,
        publish_terminal: Callable[[HostedRoomBinding, Mapping[str, Any]], None] | None = None,
        pending_action: Callable[[str, str, Mapping[str, Any] | None], None] | None = None,
        task_progress: Callable[[str, str, int, Mapping[str, Any]], None] | None = None,
        retire_task_progress: Callable[[str, str, int], None] | None = None,
        remote_member: Callable[[str, str], bool] | None = None,
        confirm_deferred_stop: Callable[[str, Mapping[str, Any]], bool] | None = None,
        maintain: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.time,
        lease_ttl_seconds: float = 30.0, poll_interval_seconds: float = 5.0,
        active_poll_interval_seconds: float = 0.25, turn_timeout_seconds: float = 1830.0,
        indeterminate_defer_seconds: float = 60.0, max_concurrent_rooms: int = 4,
        unavailable_retry_min_seconds: float = 1.0, unavailable_retry_max_seconds: float = 30.0,
        process_generation: str | None = None) -> None:
        positive = dict(
            lease_ttl_seconds=lease_ttl_seconds, poll_interval_seconds=poll_interval_seconds,
            active_poll_interval_seconds=active_poll_interval_seconds,
            turn_timeout_seconds=turn_timeout_seconds if turn_timeout_seconds > 0 else math.inf,
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
        self.pending_action, self.task_progress, self.clock = pending_action, task_progress, clock
        self.retire_task_progress = retire_task_progress
        self.remote_member, self.confirm_deferred_stop = remote_member, confirm_deferred_stop
        self.maintain = maintain  # once per worker cycle, outside any room (owed grant revokes)
        for name, value in positive.items():
            setattr(self, name, float(value))
        self.max_concurrent_rooms = max_concurrent_rooms
        self.unavailable_retry_min_seconds, self.unavailable_retry_max_seconds = (
            float(unavailable_retry_min_seconds), float(unavailable_retry_max_seconds))
        self.process_generation = process_generation or uuid.uuid4().hex
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
        self._deferred_cursor: dict[str, int] = {}
        self._unavailable_route_retries: dict[tuple[str, str], dict[str, float]] = {}
        self._blocked_rooms: set[str] = set()
        self._status_lock, self._current_tasks = threading.Lock(), {}
        self._terminal_receipts: dict[state.TaskAttempt, _TerminalReceipt] = {}
        # (room, task, generation) -> (progress time the bot last reported, home time it last changed)
        self._attempt_progress: dict[tuple[str, str, int], tuple[float | None, float]] = {}
        self._room_schedule_cursor, self._cycles = 0, 0

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
        with self._status_lock:
            # Rooms still owning a worker slot are revisited once that thread exits, closing
            # the terminal-publication/route-repair race without busy-looping idle rooms.
            self._rooms_needing_reschedule.update(self._room_threads)
        self._wake.set()

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

    # ------------------------------------------------------------------ public ops
    def cancel(
        self, identity: state.TaskIdentity, *, cancel_id: str, acknowledge: bool = True
    ) -> dict[str, Any]:
        """Persist a stop intent, then commit cancellation after acknowledgement.

        The worker transitions tasks concurrently, so the status read is only a routing
        hint: a fast-path fence failure re-reads and re-routes instead of surfacing it.
        Preparation defers acknowledgement to the worker: only durable state and wakeup
        are safe under its policy lock, which Discard may also hold recursively.
        A turn this read finds deferred (the worker may defer it after the caller's snapshot)
        is cancelled only once ``confirm_deferred_stop`` confirms its member's Stop; unconfirmed,
        or without acknowledgement (no peer call under the policy lock), it is returned as is. The cancel is fenced on the cancel generation this read saw, so
        a turn that moved on meanwhile is re-routed.
        """
        for _ in range(_CANCEL_ROUTE_RETRIES):
            before = state.get_task(self.db_path, identity)
            if before["status"] == "cancelled":
                return before
            if before["status"] in state.TERMINAL_STATUSES:
                raise state.InvalidTaskTransitionError(
                    f"cannot cancel task in state '{before['status']}'")
            if before["status"] == "stopping":
                cancel_id = before["cancel_id"]  # join the already persisted stop generation
            direct = before["status"] in {"queued", "deferred"}
            if before["status"] == "deferred" and self.confirm_deferred_stop is not None and (
                    not acknowledge or not self.confirm_deferred_stop(identity.room_id, before)):
                return state.get_task(self.db_path, identity)  # the check may have found its answer
            try:
                cancel = (state.cancel_deferred_task if before["status"] == "deferred"
                          else state.cancel_task if direct else state.begin_task_cancel)
                result = cancel(
                    self.db_path, identity, cancel_id=cancel_id,
                    expected_cancel_generation=before["cancel_generation"], clock=self.clock,
                    **({"expected_execution_generation": before["execution_generation"]}
                       if before["status"] == "deferred" else {}))
            except (state.InvalidTaskTransitionError, state.StaleTaskError):
                continue  # lost the race with the worker (settled or re-queued); re-route
            if not direct and acknowledge:
                binding = self._binding_for_room(identity.room_id)
                try:
                    if binding is not None:
                        lease = self._ensure_lease(binding)
                        if self._peer_stop_acknowledged(binding, result) or (
                            not self._settle_stopping_completion(binding, result, lease)
                            and self._stop_or_settle(binding, result, lease) is True):
                            self._complete_cancel(result, cancel_id=cancel_id)
                except Exception as exc:
                    self._record_error(f"stop remains pending: {exc}")
            self.wakeup()
            return result if direct else state.get_task(self.db_path, identity)
        # Routing retries exhausted under contention: surface the live status honestly.
        final = state.get_task(self.db_path, identity)
        if final["status"] == "cancelled":
            return final
        raise state.InvalidTaskTransitionError(
            "cancel kept losing races with task transitions "
            f"(last observed state '{final['status']}')")

    def retry_indeterminate(
        self, identity: state.TaskIdentity, *, execution_generation: int | None = None) -> dict[str, Any]:
        """Explicitly retry one uncertain attempt under the current room lease."""
        binding = self._binding_for_room(identity.room_id)
        published = self._publish_retained_receipts(binding) if binding is not None else set()
        task = state.get_task(self.db_path, identity)
        # The write below fences on this re-read generation, so pinning it to the
        # caller's selected attempt keeps a stale Retry off a newer attempt.
        if execution_generation not in (None, task["execution_generation"]):
            raise state.StaleTaskError("task attempt changed before retry")
        if identity in published and task["status"] in state.TERMINAL_STATUSES:
            return task  # this Retry published the attempt's kept reply, as _requeue does
        if task["status"] not in {"indeterminate", "deferred"}:
            raise state.InvalidTaskTransitionError(f"cannot retry task in state '{task['status']}'")
        if binding is None:
            raise state.RoomUnavailableError("hosted room is unavailable")
        lease = self._ensure_lease(binding)
        if task["status"] == "deferred":
            return self._retry_deferred(binding, task, lease)
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
        if inspection.receipts_error is not None:
            raise inspection.receipts_error  # running the turn again could answer it twice
        return self._requeue(state.requeue_indeterminate_task, task, lease, identity.room_id)

    def _retry_deferred(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease
    ) -> dict[str, Any]:
        """Retry a deferred turn without minting a new generation a peer may already have run."""
        room_id = task["identity"].room_id
        if not self._remote(task):
            return self._requeue(state.requeue_deferred_task, task, lease, room_id)
        # Resolving the peer transport replays this generation under its own idempotency key and
        # raises, leaving the turn deferred, while the peer is unreachable. Once it succeeds the
        # peer owns this generation (the replay may have just started it), so the turn is reopened
        # before anything else can fail: the worker then watches that run to its end, even when the
        # status read below fails.
        self._transport_for(binding, task)
        # Admission replay can outlast the lease; recheck ownership before reopening this attempt.
        lease = self._ensure_lease(binding)
        reopened = self._fenced(state.reopen_deferred_task, None, task, lease)
        try:
            inspection = self._inspect_recovery_session(binding, reopened)
        except (RuntimeError, ValueError, OSError) as exc:
            self._record_error(f"task {reopened['identity'].task_id} retry is being watched: {exc}")
            self.wakeup()
            return reopened
        if inspection.terminal is not None:
            return self._resolve_indeterminate(binding, reopened, lease, inspection.terminal)
        if inspection.status == "cancelled":
            return self._fenced(
                state.resolve_indeterminate_cancellation, binding, reopened, lease,
                cancel_id=f"remote-cancel:{reopened['execution_generation']}")
        self.wakeup()
        return reopened

    def _remote(self, task: Mapping[str, Any]) -> bool:
        """Whether *task*'s member admits on another gateway, which may hold an attempt this one
        lost track of: its recovery replays that exact generation, never a new one."""
        return self.remote_member is not None and bool(
            self.remote_member(task["identity"].room_id, _member_id(task)))

    def _defer_unconfirmed_stop(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease) -> bool:
        """A member on another gateway that has not confirmed a Stop within the deferral window may
        stay unreachable: release the room's queue. The turn is deferred, never reported cancelled,
        since its run may still be going there."""
        if (not self._served_elsewhere(binding, task)
                or self.clock() < float(task.get("updated_at") or 0) + self.indeterminate_defer_seconds):
            return False
        # A Stop probe that timed out can outlast the lease this commits under.
        lease = self._ensure_lease(binding)
        self._publish(binding, self._fenced(
            state.defer_unconfirmed_stop, None, task, lease, reason="member_unavailable"))
        return True

    def deferred_stop_confirmed(self, room_id: str, task: Mapping[str, Any]) -> bool:
        """Confirm Stop on the serving process, or publish this attempt's completed reply."""
        binding = self._binding_for_room(room_id)
        if binding is None:
            return False
        try:
            transport, profile, session_id = self._open_session(binding, task)
            if session_id is None:
                return transport is not None  # that daemon answered: no room session runs anything
            outcome = self._terminal_receipt(transport, profile, session_id, task)
            if outcome is None:
                outcome = self._stop_outcome(binding, task)
            if isinstance(outcome, _TerminalReceipt):
                self._commit_lost_attempt_receipt(binding, task, outcome)
                return False
            return outcome
        except Exception as exc:  # health: allow BLE001 -- an unreachable daemon keeps the turn deferred
            self._record_error(f"stop of deferred task {task['identity'].task_id} is unconfirmed: {exc}")
            return False

    def _served_elsewhere(self, binding: HostedRoomBinding, task: Mapping[str, Any]) -> bool:
        """Whether another process serves this member, which can go away without taking this one."""
        if self._remote(task):
            return True
        try:
            return bool(getattr(self._transport_for(binding, task), "served_elsewhere", False))
        except Exception:  # health: allow BLE001 -- a member whose route is gone is not served here
            return True

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
        # Canonical local Retry calls this shared path directly, after its live-session probe.
        self._publish_retained_receipts(HostedRoomBinding(room_id, lease.gateway_id, lease.authority_epoch))
        current = state.get_task(self.db_path, task["identity"])
        if current["status"] in state.TERMINAL_STATUSES:
            return current
        retried = self._fenced(requeue, None, task, lease)
        # Only a successful requeue replaces the attempt. A refused Retry or peer recovery
        # still owns its original generation and must keep accepting that run's progress.
        if self.retire_task_progress is not None:
            self.retire_task_progress(room_id, task["identity"].task_id, int(task["execution_generation"]))
        self._set_blocked(room_id, False)
        self.wakeup()
        return retried

    def _complete_cancel(
        self, task: Mapping[str, Any], *, cancel_id: str | None = None) -> dict[str, Any]:
        return state.complete_task_cancel(
            self.db_path, task["identity"], clock=self.clock,
            cancel_id=task["cancel_id"] if cancel_id is None else cancel_id,
            expected_cancel_generation=task["cancel_generation"])

    def _resolve_indeterminate(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease,
        terminal: _TerminalReceipt, *, publish: bool = True) -> dict[str, Any]:
        return self._fenced(
            state.resolve_indeterminate_task, binding, task, lease, publish=publish,
            **asdict(terminal))

    def _finish_stop(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease,
        *, admission_id: str | None = None,
    ) -> bool:
        """Terminalize a stopping task from its receipt or an acknowledged interrupt."""
        try:
            if self._settle_stopping_completion(binding, task, lease, admission_id=admission_id):
                return True
        except state.DriverStateError:
            raise
        except Exception as exc:  # health: allow BLE001 -- a member whose history fails can still take the interrupt
            self._record_error(_STOP_PENDING.format(exc=exc))
        acknowledged = self._stop_or_settle(binding, task, lease)
        if acknowledged:
            self._complete_acknowledged_stop(binding, task, lease)
        return acknowledged is not False

    def _stop_or_settle(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease
    ) -> bool | None:
        """True when the member acknowledged the Stop. None when resolving the lost attempt
        committed its own retained answer instead: that receipt is settled (and published)
        here, never reported cancelled. False while the Stop is unacknowledged."""
        outcome = self._stop_outcome(binding, task)
        if isinstance(outcome, _TerminalReceipt):
            self._fenced(state.settle_stopping_task, binding, task, lease, **asdict(outcome))
            return None
        return outcome

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
    def _terminal_receipt(
        transport: InternalSessionRPC, profile: str, session_id: str, task: Mapping[str, Any],
        *, admission_id: str | None = None,
    ) -> _TerminalReceipt | None:
        results = getattr(transport, "results", None)
        if callable(results):
            receipts = results(**_session_kw(profile, session_id),
                **({"admission_ids": [admission_id]} if admission_id is not None else {
                    "task_id": task["identity"].task_id, "execution_generation": task["execution_generation"]}))
            history = [{**receipt, "role": "assistant", "content": receipt.get("text", "")}
                       for receipt in receipts]
        else:
            history = transport.history(**_session_kw(profile, session_id))
        return _find_terminal_receipt(
            history, task["identity"], int(task["execution_generation"]))

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

    def _interrupt_stopping_task(self, binding: HostedRoomBinding, task: Mapping[str, Any]) -> bool:
        return self._stop_outcome(binding, task) is True

    def _stop_outcome(
        self, binding: HostedRoomBinding, task: Mapping[str, Any]) -> bool | _TerminalReceipt:
        transport, profile, session_id = self._open_session(binding, task)
        if session_id is None:
            # A local turn cannot survive without its canonical session, so an authoritative
            # absence is a safe Stop acknowledgement (errors raise); a peer stays uncertain.
            return transport is not None and transport is self.rpc
        info = transport.info(**_session_kw(profile, session_id))
        if info.get("status") == "unknown" and callable(getattr(transport, "discard", None)):
            return self._discard_unknown_stop(transport, profile, session_id, task, info)
        if not _info_active(info):
            # Completion can commit between the earlier receipt read and this idle report.
            return self._terminal_receipt(transport, profile, session_id, task) or True
        if not _info_is_active_for(info, task["identity"], require_exact=True):
            return self._interrupt_stale_admission(transport, binding, profile, session_id, info)
        result = transport.interrupt(
            **_session_kw(profile, session_id), expected_task_id=task["identity"].task_id)
        return result is not None and (
            result.get("interrupted") is True
            or str(result.get("status") or "") in _STOP_ACK_STATUSES)

    @staticmethod
    def _discard_unknown_stop(transport, profile, session_id, task, info) -> bool | _TerminalReceipt:
        """Resolve only this lost attempt, preserving its captured answer."""
        generation = int(task["execution_generation"])
        discard = transport.discard
        if (_info_active(info) or info.get("task_id") != task["identity"].task_id
                or type(info.get("execution_generation")) is not int
                or info["execution_generation"] != generation):
            return False
        result = discard(**_session_kw(profile, session_id),
                         expected_task_id=task["identity"].task_id, execution_generation=generation)
        if not (result and result.get("discarded") is True
                and result.get("task_id") == task["identity"].task_id
                and type(result.get("execution_generation")) is int
                and result["execution_generation"] == generation):
            return False
        return discarded_receipt(result, task) or result.get("status") == "cancelled"

    @classmethod
    def _settle_lost_attempt(
        cls, transport: InternalSessionRPC, profile: str, session_id: str, task: Mapping[str, Any],
        info: Mapping[str, Any]) -> bool | _TerminalReceipt | None:
        """Resolve this unknown head: its answer, a stop confirmation, or None for another head."""
        if (info.get("status") != "unknown" or info.get("task_id") not in (None, task["identity"].task_id)
                or info.get("execution_generation") not in (None, task["execution_generation"])):
            return None
        return cls._discard_unknown_stop(transport, profile, session_id, task, info)

    def _commit_lost_attempt_receipt(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], receipt: _TerminalReceipt) -> None:
        self._retain_terminal_receipt(binding, state.TaskAttempt(
            identity=task["identity"], lease=self._ensure_lease(binding),
            execution_generation=int(task["execution_generation"]),
            cancel_generation=int(task["cancel_generation"])), receipt)

    def _interrupt_stale_admission(
        self, transport: InternalSessionRPC, binding: HostedRoomBinding, profile: str,
        session_id: str, info: Mapping[str, Any]) -> bool:
        """Clear a session turn still held by a driver task with no live attempt.

        A member turn can outlive its driver task (a deferred task whose approval never
        arrived, for example), so ``info`` stays active under that stale identity and
        every exact-stop retry would refuse to act on it forever. Interrupt that stale
        turn by its own id — both sides match on the exact task id — and keep this stop
        pending; the next retry sees the exact identity (or an inactive session) and
        finishes. Turns of live attempts and unknown identities stay untouched.
        """
        stale_task_id = str(info.get("task_id") or "")
        if not stale_task_id:
            return False
        live_ids = {
            t["identity"].task_id
            for status in ("running", "indeterminate") for t in self._tasks(binding, status)}
        if stale_task_id in live_ids:
            return False
        stale = next((t for t in self._tasks(binding, None) if t["identity"].task_id == stale_task_id), None)
        # A peer member's deferred attempt is kept on purpose: its Retry recovers that same run.
        if stale is None or self._remote(stale):
            return False
        with suppress(RuntimeError, OSError, ValueError):
            transport.interrupt(
                **_session_kw(profile, session_id), expected_task_id=stale_task_id)
        return False

    def _settle_stopping_completion(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease,
        *, admission_id: str | None = None,
    ) -> bool:
        """Publish a terminal receipt that arrived before Stop was acknowledged."""
        transport, profile, session_id = self._open_session(binding, task)
        if session_id is None:
            return False
        receipt = self._terminal_receipt(transport, profile, session_id, task, admission_id=admission_id)
        if receipt is None:
            return False
        self._fenced(state.settle_stopping_task, binding, task, lease, **asdict(receipt))
        return True

    def _report_pending_action(
        self, task: Mapping[str, Any], *, session_id: str, info: Mapping[str, Any]) -> None:
        if self.task_progress is not None:
            # The member's current tool rides the same info poll as its pending approval.
            self.task_progress(
                task["identity"].room_id, task["identity"].task_id, int(task["execution_generation"]), info)
        if self.pending_action is None:
            return
        approval, action = info.get("pending_approval") or info.get("approval"), None
        if isinstance(approval, Mapping):
            choices = [c for c in approval.get("choices") or () if c in {"once", "deny"}]
            safe_approval = {**approval, "choices": choices or ["once", "deny"]}
            action = {
                "kind": "approval", "task_id": task["identity"].task_id,
                "execution_generation": int(task["execution_generation"]),
                "run_id": info.get("run_id"), "session_id": session_id,
                "request_id": safe_approval.get("request_id"), "approval": safe_approval}
        self.pending_action(task["identity"].room_id, _member_id(task), action)

    def _retry_stopping_tasks(self, binding: HostedRoomBinding, lease: state.DriverLease) -> bool:
        for task in self._tasks(binding, "stopping"):
            try:
                lease = self._renew_lease_if_needed(lease)
                if self._peer_stop_acknowledged(binding, task):
                    self._complete_cancel(task)
                    continue
                if self._finish_stop(binding, task, lease):
                    continue
            except Exception as exc:
                self._record_error(_STOP_PENDING.format(exc=exc))
            if not self._defer_unconfirmed_stop(binding, task, lease):
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
                if self.maintain is not None:
                    try:
                        self.maintain()
                    except Exception as exc:  # health: allow BLE001 -- recorded; the next cycle retries
                        self._record_error(f"worker maintenance failed: {exc}")
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
            self._forget_unlisted_rooms()
            for binding in tuple(self._rooms_provider()):
                if self._stop.is_set():
                    return
                self._run_room_once(binding)
            return
        self._forget_unlisted_rooms()
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
        self._publish_retained_receipts(binding)
        # A persisted Stop must recover before historical publication/checkpoint
        # work. Otherwise a slow preparation can indefinitely delay cancellation.
        if self._tasks(binding, "stopping"):
            lease = self._ensure_lease(binding)
            if self._retry_stopping_tasks(binding, lease):
                self._set_blocked(binding.room_id, True)
                return
            self._set_blocked(binding.room_id, False)
        self._forget_ended_attempts(binding)
        self._observe_deferred_receipt(binding)
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
            lease = self._renew_lease_if_needed(lease)
            attempt = state.start_task(
                self.db_path, task["identity"], lease,
                expected_cancel_generation=task["cancel_generation"], clock=self.clock)
            self._execute_attempt(binding, task, attempt)
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
            self.db_path, room_id=binding.room_id, gateway_id=binding.gateway_id,
            authority_epoch=binding.authority_epoch, process_generation=self.process_generation,
            ttl_seconds=self.lease_ttl_seconds, clock=self.clock)
        lease = self._cache_lease(lease)
        with self._status_lock:
            self._recovered_leases = {k for k in self._recovered_leases if k[0] != binding.room_id}
        return lease

    def _cache_lease(self, lease: state.DriverLease) -> state.DriverLease:
        """Cache ``lease`` unless the cache already holds a later expiry of the same generation.

        A LeaseKeeper thread and the turn's thread renew concurrently and commit in either order;
        a stale write would let ``_mark_ambiguous`` defer to an expiry the row has outlived, and
        the next scan would then renew that generation in place instead of recovering the room.
        """
        with self._status_lock:
            known = self._leases.get(lease.room_id)
            if (known is not None and (known.process_generation, known.lease_generation)
                    == (lease.process_generation, lease.lease_generation)
                    and known.expires_at >= lease.expires_at):
                return known
            self._leases[lease.room_id] = lease
            return lease

    def _renew_lease_if_needed(
        self, lease: state.DriverLease, *, force: bool = False) -> state.DriverLease:
        with self._status_lock:
            known = self._leases.get(lease.room_id)
        if (known is not None and known.lease_generation == lease.lease_generation
                and known.expires_at > lease.expires_at):
            lease = known  # a LeaseKeeper renewed this generation since the caller's copy
        if not force and self.clock() < lease.expires_at - (self.lease_ttl_seconds / 2):
            return lease
        return self._cache_lease(state.renew_lease(
            self.db_path, lease, ttl_seconds=self.lease_ttl_seconds, clock=self.clock))

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
        self, binding: HostedRoomBinding, task: Mapping[str, Any], attempt: state.TaskAttempt
    ) -> None:
        profile, submit_attempted, submitted = task["payload"]["target_profile"], False, False
        attachments = task["payload"].get("attachments")
        with self._status_lock:
            self._current_tasks[binding.room_id] = attempt.identity
        try:
            # Inside the try: a member whose destination is gone (an unserved bot) fails its own
            # turn before submit instead of leaving the task running and the room wedged.
            transport = self._transport_for(binding, task)
            if attachments and not getattr(transport, "accepts_attachments", False):
                # A route that cannot carry the files would answer without them, and a
                # retry cannot change the route: settle a proven failure before submit.
                raise AttachmentsUnsupportedError(
                    "attachments_unsupported_on_route: this member's route cannot carry "
                    "attached files")
            # The profile lock can be held by another turn for up to `turn_wait_seconds`, past the
            # lease ttl, and an expired lease cannot be renewed: keep it alive through the wait
            # and the whole turn, and fence before submit if a renewal proved it lost.
            with LeaseKeeper(
                    attempt.lease, lambda lease: self._renew_lease_if_needed(lease, force=True),
                    interval=self.lease_ttl_seconds / 3) as keeper, self.turn_lock(profile):
                keeper.renew_now()
                session = self._resolve_or_create(transport, profile, binding.room_id)
                # A submit should fail before admission or return after it; an unexpected
                # exception at that boundary is ambiguous, never a proven failure.
                submit_attempted, session_id = True, _session_id(session)
                submission = transport.submit(
                    **_session_kw(profile, session_id), prompt=task["payload"]["prompt"],
                    task=attempt.identity, execution_generation=attempt.execution_generation,
                    attachments=attachments,
                    member_id=_member_id(task),
                    on_terminal=lambda receipt: self._on_terminal(binding, attempt, receipt))
                submitted = True
                self._unavailable_route_retries.pop(
                    (task["identity"].room_id, _member_id(task)), None)
                receipt = self._wait_for_terminal(
                    binding, profile=profile, session_id=session_id, attempt=attempt,
                    transport=transport, admission_id=submission.get("admission_id"))
                if receipt is None:
                    return
                self._retain_terminal_receipt(binding, attempt, receipt)
        except (state.StaleLeaseError, state.StaleTaskError) as exc:
            self._drop_lease(binding.room_id)
            self._record_task_error(attempt, f"fenced: {exc}")
        except Exception as exc:
            # Exceptions may contain prompt text or credentials. Persist only a fixed
            # reason, not the free-form messages retained in the in-memory status.
            reason_code = getattr(exc, 'reason', None)
            reason = ('request_too_large' if reason_code == 'request_too_large'
                      else f'attempt_failed ({type(exc).__name__})')
            logger.warning('hosted room task %s: %s', attempt.identity.task_id, reason)
            refusal = reason_code if submit_attempted and not submitted else None
            if getattr(exc, "refused_before_submit", False):
                # File preparation refused work that never started; no retry can change it.
                self._settle_failure_if_current(attempt, exc)
            elif refusal == "unknown_execution" or (
                    submit_attempted and not bool(getattr(exc, "ambiguous", False))
                    and bool(getattr(exc, "not_admitted", False))):
                if refusal == "unknown_execution":
                    self._clear_refusing_head(binding, task, attempt)
                try:
                    state.requeue_not_admitted_task(self.db_path, attempt, clock=self.clock)
                except (state.StaleLeaseError, state.StaleTaskError) as fence_exc:
                    self._mark_ambiguous(binding, attempt)
                    self._record_task_error(
                        attempt, f"not-admitted proof lost its fence: {fence_exc}")
                else:
                    delay = self._defer_unavailable_route(task)
                    self._record_task_error(
                        attempt, f"was not admitted; queued for retry in {delay:g}s")
            elif submit_attempted:
                self._mark_ambiguous(binding, attempt)
                self._record_task_error(attempt, f"observation failed after submit: {exc}")
            else:
                self._settle_failure_if_current(attempt, exc)
        finally:
            with self._status_lock:
                self._current_tasks.pop(binding.room_id, None)
                # The task may have published a reply or exposed the next turn while this
                # thread held its slot: schedule exactly one follow-up after it leaves
                # (idle room scans never set this marker).
                self._rooms_needing_reschedule.add(binding.room_id)

    def _clear_refusing_head(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], attempt: state.TaskAttempt) -> None:
        """Clear an older unknown head blocking admission, preserving its answer."""
        try:
            transport, profile, session_id = self._open_session(binding, task)
            info = transport.info(**_session_kw(profile, session_id))
            if info.get("status") != "unknown":
                return
            head_id, generation = info["task_id"], info["execution_generation"]
            result = transport.discard(**_session_kw(profile, session_id), expected_task_id=head_id,
                                       execution_generation=generation)
            head = next((t for t in self._tasks(binding, None) if t["identity"].task_id == head_id
                         and t["execution_generation"] == generation), None)
            receipt = None if head is None else discarded_receipt(result or {}, head)
            if receipt is not None:
                self._commit_lost_attempt_receipt(binding, head, receipt)
        except Exception as exc:  # health: allow BLE001 -- retried with the submit
            self._record_task_error(attempt, f"could not clear what holds its bot's queue: {exc}")

    def _mark_ambiguous(self, binding: HostedRoomBinding, attempt: state.TaskAttempt) -> None:
        # Wait out the lease as last renewed (a LeaseKeeper may have extended it), so the room is
        # reclaimed under a new generation and recovery inspects the ambiguous attempt.
        with self._status_lock:
            known = self._leases.pop(binding.room_id, None)
        expires_at = attempt.lease.expires_at
        if known is not None and known.lease_generation == attempt.lease.lease_generation:
            expires_at = max(expires_at, known.expires_at)
        self._ambiguous_rooms[binding.room_id] = expires_at

    def _on_terminal(
        self, binding: HostedRoomBinding, attempt: state.TaskAttempt, receipt: Mapping[str, Any]
    ) -> None:
        """Durably commit one in-process terminal receipt for ``attempt``."""
        status = receipt.get("status")
        if status == "cancelled":
            self.wakeup()
            return
        terminal = _receipt_from(receipt, attempt.identity, attempt.execution_generation)
        self._retain_terminal_receipt(binding, attempt, terminal)

    def _retain_terminal_receipt(
        self, binding: HostedRoomBinding, attempt: state.TaskAttempt, terminal: _TerminalReceipt
    ) -> None:
        # Dashboard history cannot identify an attempt. Keep its callback receipt until a
        # current lease can commit it, or a newer attempt / completed control makes it obsolete.
        with self._status_lock:
            self._terminal_receipts[attempt] = terminal
        self._commit_retained_receipt(binding, attempt, terminal)
        self.wakeup()

    def _publish_retained_receipts(self, binding: HostedRoomBinding) -> set[state.TaskIdentity]:
        """Commit the room's kept receipts; return the tasks whose receipt is no longer kept."""
        with self._status_lock:
            receipts = [(attempt, receipt) for attempt, receipt in self._terminal_receipts.items()
                        if attempt.identity.room_id == binding.room_id]
        return {attempt.identity for attempt, receipt in receipts
                if self._commit_retained_receipt(binding, attempt, receipt)}

    def _forget_unlisted_rooms(self) -> None:

        listed = {binding.room_id for binding in self._rooms_provider()}
        with self._status_lock:  # this gateway no longer runs those group chats
            for attempt in [a for a in self._terminal_receipts if a.identity.room_id not in listed]:
                self._terminal_receipts.pop(attempt)
            for key in [k for k in self._attempt_progress if k[0] not in listed]:
                self._attempt_progress.pop(key)
            for room_id in [r for r in self._deferred_cursor if r not in listed]:
                self._deferred_cursor.pop(room_id)

    def _commit_retained_receipt(
        self, binding: HostedRoomBinding, attempt: state.TaskAttempt, terminal: _TerminalReceipt
    ) -> bool:
        """Whether the receipt is done with: committed, obsolete, or no longer this gateway's."""
        try:
            self._commit_terminal_receipt(binding, attempt, terminal)
        except state.AuthorityChangedError as exc:
            self._record_task_error(attempt, f"terminal receipt dropped: {exc}")
        except (state.StaleLeaseError, state.StaleTaskError, state.LeaseHeldError) as exc:
            self._record_task_error(attempt, f"terminal receipt is waiting for reconciliation: {exc}")
            return False  # Retry with the receipt intact after the current lease/control race.
        except state.RoomUnavailableError:
            pass  # A disbanded group can no longer publish.
        except state.DriverStateError as exc:
            self._settle_failure_if_current(
                attempt, RuntimeError(f"terminal result could not be committed: {exc}"))
        except Exception:
            logger.exception("task %s terminal reconciliation failed", attempt.identity.task_id)
            return False
        with self._status_lock:
            self._terminal_receipts.pop(attempt, None)
        return True

    def _commit_terminal_receipt(
        self, binding: HostedRoomBinding, attempt: state.TaskAttempt, terminal: _TerminalReceipt
    ) -> None:
        try:
            self._publish(binding, state.settle_task(
                self.db_path, attempt, **asdict(terminal), clock=self.clock, allow_expired_lease=True))
            return
        except (state.StaleLeaseError, state.StaleTaskError):
            pass  # Recovery can still accept this exact attempt under the current lease.
        current = state.get_task(self.db_path, attempt.identity)
        # Only a Stop raises the cancel generation, and this row is not cancelled: that Stop was
        # never confirmed (stopping, or deferred after it), so this run's answer still wins it.
        if (current["execution_generation"] != attempt.execution_generation
                or current["cancel_generation"] < attempt.cancel_generation
                or current["status"] not in {"running", "indeterminate", "deferred", "stopping"}):
            return
        lease = self._ensure_lease(binding)
        if current["status"] == "indeterminate":
            self._resolve_indeterminate(binding, current, lease, terminal)
        else:
            self._fenced(
                state.settle_stopping_task if current["status"] == "stopping" else state.settle_recovered_task,
                binding, current, lease, **asdict(terminal))
        self._inspected_indeterminate_attempts.discard((
            binding.room_id, attempt.identity.task_id, attempt.execution_generation))

    def _wait_for_terminal(
        self, binding: HostedRoomBinding, *, profile: str, session_id: str,
        attempt: state.TaskAttempt, transport: InternalSessionRPC, admission_id: str | None = None,
    ) -> _TerminalReceipt | None:
        lease = attempt.lease
        unobserved_since = None
        wait_cap = min(self.active_poll_interval_seconds, self.lease_ttl_seconds / 4)
        with self._status_lock:  # the wait for the profile lock and the session spends none of the limit
            self._attempt_progress[
                attempt.identity.room_id, attempt.identity.task_id, attempt.execution_generation
            ] = (None, self.clock())
        while not self._stop.is_set():
            task = state.get_task(self.db_path, attempt.identity)
            if task["status"] in state.TERMINAL_STATUSES:
                return None
            if task["status"] == "stopping":
                try:
                    lease = self._renew_lease_if_needed(lease)
                    if self._finish_stop(binding, task, lease, admission_id=admission_id):
                        return None
                except state.StaleLeaseError:
                    raise  # the room's scan keeps this Stop going under a new lease
                except Exception as exc:
                    self._record_error(_STOP_PENDING.format(exc=exc))
                    if unobserved_since is None:
                        unobserved_since = self.clock()
                    if self.clock() - unobserved_since >= self.lease_ttl_seconds:
                        return None  # the room's scan keeps this Stop going, without the profile lock
                if self._defer_unconfirmed_stop(binding, task, lease):
                    return None
                self._wake.wait(wait_cap)
                self._wake.clear()
                continue
            lease = self._renew_lease_if_needed(lease)
            progress_at, alive = None, False
            try:
                receipt = self._terminal_receipt(
                    transport, profile, session_id, task, admission_id=admission_id)
                if receipt is not None:
                    return receipt
                info = transport.info(**_session_kw(profile, session_id))
                progress_at = _attempt_progress_at(info, task)
                self._report_pending_action(task, session_id=session_id, info=info)
                alive = _info_active(info)
            except state.DriverStateError:
                raise
            except Exception as exc:
                if getattr(exc, "not_admitted", False) or getattr(exc, "refused_before_submit", False):
                    raise
                self._record_task_error(attempt, f"observation failed after submit: {exc}")
            # A turn queued behind another admission or waiting for an approval reports active. For
            # one its bot lost, as for reads that keep failing, no limit may be set to end this watch.
            if alive:
                unobserved_since = None
            elif unobserved_since is None:
                unobserved_since = self.clock()
            elif self.clock() - unobserved_since >= self.lease_ttl_seconds:
                state.mark_attempt_indeterminate(self.db_path, attempt, lease, clock=self.clock)
                return None
            if self._past_execution_deadline(task, progress_at):
                self._expire_attempt_deadline(binding, task, lease, admission_id=admission_id)
                return None
            self._wake.wait(wait_cap)
            self._wake.clear()
        return None

    def _complete_acknowledged_stop(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease
    ) -> dict[str, Any]:
        """Terminalize an acknowledged Stop: deadline stops publish an explicit failure."""
        if not str(task.get("cancel_id") or "").startswith("deadline:"):
            return self._complete_cancel(task)
        return self._fenced(
            state.settle_stopping_task, binding, task, lease,
            settlement_id=f"deadline:{int(task['execution_generation'])}", status="failed",
            result={
                "error": "This group chat turn exceeded its time limit and was stopped.",
                "reason_code": "turn_deadline_exceeded",
                # A Stop begun under a finite limit may finish after a restart turned it off.
                **({"timeout_seconds": self.turn_timeout_seconds}
                   if math.isfinite(self.turn_timeout_seconds) else {})})

    def _expire_attempt_deadline(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease,
        *, admission_id: str | None = None,
    ) -> None:
        """Fence, stop, and terminalize one exact attempt at its deadline."""
        if task["status"] in {"running", "indeterminate"}:
            task = state.begin_task_cancel(
                self.db_path, task["identity"], clock=self.clock,
                cancel_id=f"deadline:{int(task['execution_generation'])}",
                expected_cancel_generation=int(task["cancel_generation"]))
        elif task["status"] != "stopping":
            return
        # A user Stop that won the race keeps its own cancellation semantics.
        if not str(task.get("cancel_id") or "").startswith("deadline:"):
            return
        lease = self._renew_lease_if_needed(lease, force=True)
        if not self._finish_stop(binding, task, lease, admission_id=admission_id):
            self._record_error(
                f"task {task['identity'].task_id} exceeded its deadline; stop remains pending")

    # ------------------------------------------------------------------ recovery
    def _observe_deferred_receipt(self, binding: HostedRoomBinding) -> None:
        # One old local attempt per pass: receipts may become readable long after deferral.
        # Resolving a peer transport replays its admission, so only explicit Retry may do that.
        deferred = [t for t in self._tasks(binding, "deferred") if not self._remote(t)]
        with self._status_lock:
            if not deferred:
                self._deferred_cursor.pop(binding.room_id, None)
                return
            index = self._deferred_cursor.get(binding.room_id, 0) % len(deferred)
            self._deferred_cursor[binding.room_id] = index + 1
        task = deferred[index]
        try:
            transport, profile, session_id = self._open_session(binding, task)
            receipt = (self._terminal_receipt(transport, profile, session_id, task)
                       if session_id and callable(getattr(transport, "results", None)) else None)
            if receipt is not None:
                self._fenced(state.settle_recovered_task, binding, task,
                             self._ensure_lease(binding), **asdict(receipt))
            elif session_id and callable(getattr(transport, "discard", None)):
                inspection = self._inspect_session(transport, task, session_id, read_history=False)
                if inspection.status == "unknown" and not inspection.active:
                    task = self._fenced(state.reopen_deferred_task, None, task, self._ensure_lease(binding))
                    self._give_up_on_attempt(binding, task, inspection, probe_failed=False)
        except Exception as exc:  # health: allow BLE001 -- a failed old receipt must not block later work
            logger.debug("deferred receipt read failed: %s", type(exc).__name__)

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
                # Its limit still applies. While the old process holds the lease this raises instead.
                if not self._past_execution_deadline(task, inspection.progress_at):
                    raise state.LeaseHeldError("recovered session turn is still active")
                self._expire_attempt_deadline(binding, task, self._ensure_lease(binding))

    def _inspect_session(
        self, transport: InternalSessionRPC, task: Mapping[str, Any], session_id: str,
        *, read_history: bool) -> _RecoveryInspection:
        """Probe one resolved session: optional terminal receipt from history, then live info."""
        profile = task["payload"]["target_profile"]
        receipt = receipts_error = None
        if read_history:
            try:
                receipt = self._terminal_receipt(transport, profile, session_id, task)
            except Exception as exc:  # health: allow BLE001 -- receipt availability says nothing about the live turn
                # By type only: exceptions may contain prompt text. A peer repeats this every cycle.
                logger.debug("task %s recovery receipt read failed: %s",
                             task["identity"].task_id, type(exc).__name__)
                receipts_error = exc
        info = transport.info(**_session_kw(profile, session_id))
        self._report_pending_action(task, session_id=session_id, info=info)
        return _RecoveryInspection(
            terminal=receipt, active=_info_is_active_for(info, task["identity"]),
            status=str(info.get("status") or "") or None, progress_at=_attempt_progress_at(info, task),
            receipts_error=receipts_error)

    def _inspect_recovery_session(
        self, binding: HostedRoomBinding, task: Mapping[str, Any]) -> _RecoveryInspection:
        inspection = _NO_INSPECTION
        try:
            profile, transport = task["payload"]["target_profile"], self._transport_for(binding, task)
            with self.turn_lock(profile):
                session_id = self._resume_exact(transport, task["identity"].room_id, profile)
                if session_id is not None:
                    inspection = self._inspect_session(transport, task, session_id, read_history=True)
            return inspection
        finally:
            if inspection is _NO_INSPECTION and self.task_progress is not None:
                self.task_progress(
                    task["identity"].room_id, task["identity"].task_id, int(task["execution_generation"]), {})

    def _inspect_local_recovery_session(self, task: Mapping[str, Any]) -> _RecoveryInspection:
        """Check only live process state (no resume, no history) before explicit local recovery.

        A restart loses the in-process terminal callback identity and history cannot prove
        which attempt authored a row, so an inactive abandoned attempt stays indeterminate
        until the user retries it under a new fenced generation.
        """
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
            remote = self._remote(task)
            try:
                # Resolving a remote member's route replays its admission: do it once per cycle,
                # in the probe below, not just to learn the route is not in process.
                is_local = not remote and self._transport_for(binding, task) is self.rpc
            except Exception as exc:  # health: allow BLE001 -- a lost route must not hold the room past its deadline
                # A missing route or a failed admission replay (PeerRunsHTTPError) must not keep
                # the turn from reaching its deadline. Retry of the deferred turn recovers this
                # same generation first (_retry_deferred), so deferring never mints a new key.
                self._record_error(
                    f"task {task['identity'].task_id} recovery probe failed: {exc}")
                is_local = False
            deadline = self.indeterminate_defer_seconds + float(
                task.get("indeterminate_at") or task.get("updated_at") or task.get("created_at")
                or self.clock())
            if is_local and (attempt_key not in inspected or self.clock() >= deadline):
                inspection = self._inspect_local_recovery_session(task)
                inspected.add(attempt_key)
                # The probe can wait on the profile lock longer than the lease lasts.
                lease = self._ensure_lease(binding)
                if inspection.terminal is not None:
                    self._resolve_indeterminate(binding, task, lease, inspection.terminal)
                    inspected.discard(attempt_key)
                    continue
                if inspection.active:
                    # Deferring a running turn would lose its reply and its Stop.
                    if self._past_execution_deadline(task, inspection.progress_at):
                        self._expire_attempt_deadline(binding, task, lease)
                        inspected.discard(attempt_key)
                        continue
                    self._set_blocked(binding.room_id, True)
                    return True
            inspection, probe_failed = _NO_INSPECTION, False
            # A remote run may finish between probes, so it is observed every cycle (its client
            # owns the HTTP backoff); an in-process session is probed once, then at the deadline.
            if remote or attempt_key not in inspected or self.clock() >= deadline:
                try:
                    if remote or self._transport_for(binding, task) is not self.rpc:
                        inspection = self._inspect_recovery_session(binding, task)
                except Exception as exc:
                    probe_failed = True
                    self._record_error(
                        f"task {task['identity'].task_id} recovery probe failed: {exc}")
                inspected.add(attempt_key)
                # A probe that timed out can outlast the lease every outcome below commits under.
                lease = self._ensure_lease(binding)
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
            if inspection.active and self._past_execution_deadline(task, inspection.progress_at):
                # Recovery applies the same inactivity limit and exact Stop as a watched turn.
                self._expire_attempt_deadline(binding, task, lease)
                inspected.discard(attempt_key)
                continue
            if self.clock() < deadline or inspection.active:
                # A peer run that is still going keeps the room waiting, as a local one does:
                # deferring it would release the room while the bot is still working.
                self._set_blocked(binding.room_id, True)
                return True
            self._give_up_on_attempt(binding, task, inspection, probe_failed=probe_failed)
            inspected.discard(attempt_key)
        self._set_blocked(binding.room_id, False)
        return False

    def _give_up_on_attempt(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], inspection: _RecoveryInspection,
        *, probe_failed: bool) -> None:
        """Resolve a lost head before deferring its turn or handing its message to the lead."""
        lost = None
        if inspection.status == "unknown":
            try:
                transport, profile, session_id = self._open_session(binding, task)
                if session_id is not None and callable(getattr(transport, "discard", None)):
                    lost = self._settle_lost_attempt(
                        transport, profile, session_id, task, transport.info(**_session_kw(profile, session_id)))
            except Exception as exc:  # health: allow BLE001 -- unresolved: its Retry or the next turn tries again
                self._record_error(f"task {task['identity'].task_id} recovery could not resolve its lost run: {exc}")
        if isinstance(lost, _TerminalReceipt):
            self._resolve_indeterminate(binding, task, self._ensure_lease(binding), lost)
            return
        # A failed probe or unread receipts say nothing about whether the turn is lost.
        lost_routing = (not probe_failed and inspection.receipts_error is None
                        and self._lead_takes_lost_routing(binding, task))
        lease = self._ensure_lease(binding)  # the confirmation can outlast the lease
        if lost_routing:
            given_up = self._fenced(
                state.resolve_indeterminate_cancellation, None, task, lease,
                cancel_id=f"{LOST_ROUTING_CANCEL_PREFIX}{task['execution_generation']}")
        else:
            given_up = self._fenced(
                state.defer_indeterminate_task, None, task, lease, reason="member_unavailable")
        self._publish(binding, given_up)

    def _lead_takes_lost_routing(self, binding: HostedRoomBinding, task: Mapping[str, Any]) -> bool:
        """Whether a lead's lost routing turn is given up rather than deferred: only once nothing
        can still run it, as for Stop of a deferred turn. Its message then goes to the lead alone
        (``first_round_members``), and no Retry or late answer can route it a second time. While the
        lead's computer cannot confirm, the turn is deferred and the message waits for its answer."""
        if task["payload"].get("lead_routing") is not True:
            return False
        try:
            return self.confirm_deferred_stop is None or self.confirm_deferred_stop(binding.room_id, task)
        except Exception as exc:  # health: allow BLE001 -- unconfirmed: the turn is deferred instead
            self._record_error(f"task {task['identity'].task_id} routing stop is unconfirmed: {exc}")
            return False

    def _past_execution_deadline(
        self, task: Mapping[str, Any], progress_at: float | None = None) -> bool:
        """Whether the attempt shows no activity for ``turn_timeout_seconds``. Activity is a change
        in the progress time its bot reports, the first one seen included, timed by this home's
        clock: the bot's own clock may be far off. Until then the limit counts from the later of the
        start, which a reopened attempt resets, and the submit."""
        started = task.get("started_at")
        if started is None:
            return False
        key = _attempt_key(task)
        with self._status_lock:
            reported, changed_at = self._attempt_progress.get(key, (None, 0.0))
            if self._remote(task) and progress_at is None and reported is None:
                return False  # older peers cannot distinguish activity from silence; they enforce their own limit
            if progress_at is not None and progress_at != reported:
                changed_at = self.clock()
                self._attempt_progress[key] = (progress_at, changed_at)
        return self.clock() >= max(float(started), changed_at) + self.turn_timeout_seconds

    def _forget_ended_attempts(self, binding: HostedRoomBinding) -> None:
        with self._status_lock:
            remembered = {key for key in self._attempt_progress if key[0] == binding.room_id}
        if remembered:
            live = {_attempt_key(task) for status in ("running", "indeterminate", "stopping")
                    for task in self._tasks(binding, status)}
            with self._status_lock:
                for key in remembered - live:
                    self._attempt_progress.pop(key, None)

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
    """Return the canonical hidden session title for one hosted room."""
    return f"Group: {room_id}"


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


def _receipt_from(
    receipt: Mapping[str, Any], identity: state.TaskIdentity, execution_generation: int) -> _TerminalReceipt:
    return _TerminalReceipt(
        status="settled" if receipt.get("status") == "settled" else "failed",
        settlement_id=receipt.get("settlement_id") or f"reply:{identity.task_id}:{execution_generation}",
        result=_bounded_terminal_result(receipt))


def discarded_receipt(result: Mapping[str, Any], task: Mapping[str, Any]) -> _TerminalReceipt | None:
    """The lost attempt's own receipt when its Discard committed a retained answer rather than
    cancelling it (``resolve_unknown`` keeps a captured result), else None."""
    if result.get("status") not in {"settled", "failed"}:
        return None
    return _receipt_from(result, task["identity"], int(task["execution_generation"]))


def _bounded_terminal_result(receipt: Mapping[str, Any]) -> dict[str, Any]:
    text, truncated = _truncate_utf8(receipt.get("text", ""), max_bytes=MAX_TERMINAL_TEXT_BYTES)
    error, error_truncated = _truncate_utf8(receipt.get("error", ""), max_bytes=4096)
    return {
        "message_id": receipt.get("message_id"), "text": text,
        **({"error": error} if error else {}),
        **({"truncated": True} if truncated or error_truncated else {})}


def _find_terminal_receipt(
    history: Sequence[Mapping[str, Any]], identity: state.TaskIdentity, execution_generation: int
) -> _TerminalReceipt | None:
    for message in reversed(history):
        status = message.get("status")
        if (
            message.get("task_id") != identity.task_id
            or message.get("execution_generation") != execution_generation
            or message.get("role") != "assistant" or status not in {"settled", "failed"}):
            continue
        receipt_id = message.get("message_id")
        if not isinstance(receipt_id, str) or not receipt_id:
            receipt_id = f"reply:{identity.task_id}:{execution_generation}"
        return _TerminalReceipt(
            status=cast(state.TerminalStatus, status), settlement_id=receipt_id,
            result=_bounded_terminal_result(
                {"message_id": receipt_id, "text": message.get("content", "")}))
    return None


def _info_active(info: Mapping[str, Any]) -> bool:
    return bool(info.get("active", info.get("running", False)))


def _attempt_key(task: Mapping[str, Any]) -> tuple[str, str, int]:
    return task["identity"].room_id, task["identity"].task_id, int(task["execution_generation"])


def _attempt_progress_at(info: Mapping[str, Any], task: Mapping[str, Any]) -> float | None:
    """Use only a finite frame timestamp belonging to this attempt, never the polling time."""
    value = info.get("progress_at")
    if (info.get("task_id") in (None, task["identity"].task_id)
            and info.get("execution_generation") in (None, task["execution_generation"])
            and type(value) in (int, float) and math.isfinite(value)):
        return float(value)
    return None


def _info_is_active_for(
    info: Mapping[str, Any], identity: state.TaskIdentity, *, require_exact: bool = False) -> bool:
    accepted = (identity.task_id,) if require_exact else (None, identity.task_id)
    return _info_active(info) and info.get("task_id") in accepted
