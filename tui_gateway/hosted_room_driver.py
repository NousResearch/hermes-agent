"""Runtime adapter for gateway-owned hosted room turns.

The durable state machine lives in :mod:`gateway.hosted_room_driver`; this module owns the
process-local worker and an injected session adapter, never the gateway server or agents.
One bounded supervisor schedules independent room workers: profile turn locks serialize Bots
sharing a profile while a room waiting for approval cannot stall unrelated rooms. Member
sessions reuse ``Group: <room_id>`` so a local-to-hosted migration keeps one transcript.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ContextManager, Protocol, cast

from gateway import hosted_room_driver as state
from gateway.hosted_room_owner_probe import probe_owner_incarnation

_CANCEL_ROUTE_RETRIES = 8
_STOP_ACK_STATUSES = {"cancelled", "interrupted"}

ROOM_SESSION_SOURCE = "bot_room"
MAX_TERMINAL_TEXT_BYTES = 64 * 1024
# Per staged clip; the assembled turn is then held to the planner's own prompt budget below.
MAX_STAGED_MEDIA_TEXT_BYTES = 8 * 1024
_STAGED_MEDIA_HEADER = "\n\nTranscribed from the attached audio:\n"
_STAGED_MEDIA_TRUNCATED = "…[transcript truncated]"
_TERMINAL_TRUNCATION_NOTICE = (
    "\n\n[Reply truncated. Ask the Bot to share the full result as a file.]")
_STOP_PENDING = "stop retry remains pending: {exc}"


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
        execution_generation: int, on_terminal: Callable[[Mapping[str, Any]], None],
    ) -> Mapping[str, Any]: ...
    def history(
        self, *, profile: str, session_id: str, source: str) -> Sequence[Mapping[str, Any]]: ...
    def info(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]: ...
    def interrupt(
        self, *, profile: str, session_id: str, source: str, expected_task_id: str,
        expected_execution_generation: int | None = None,
    ) -> Mapping[str, Any] | None: ...


    def rollback_attachment_staging(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        execution_generation: int,
    ) -> None:
        """Restore pending session attachments after a pre-submit failure."""


    def commit_attachment_staging(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        execution_generation: int,
    ) -> None:
        """Accept staged attachments after crossing the submit boundary."""


    def begin_attachment_staging(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        execution_generation: int,
    ) -> None:
        """Snapshot pending session attachments for one fenced attempt."""


    def stage_attachment(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        attachment: Mapping[str, Any],
        data: bytes,
        execution_generation: int,
    ) -> Mapping[str, Any]:
        """Stage one verified canonical blob into the local member session."""


MemberTransportResolver = Callable[["HostedRoomBinding", Mapping[str, Any]], InternalSessionRPC]


def _with_staged_media_text(prompt: str, notes: Sequence[str]) -> str:
    """Append staged-clip transcripts, bounded per clip and by the assembled prompt budget.

    Per-clip truncation cannot hold the total: one turn can carry several voice messages, so the
    whole tail is measured against the planner's own ``MAX_PROMPT_BYTES``.  When the mandatory
    prompt and its opaque ``@file:`` refs leave no room for any derived content this raises
    before admission — the canonical attachment bytes stay on the event, so an explicit Retry
    still works — rather than silently dropping what the user said or submitting an oversized turn.
    """

    from gateway.hosted_room_discussion import _truncate_utf8_text

    if not notes:
        return prompt
    body = "\n".join(
        _truncate_utf8_text(note, max_bytes=MAX_STAGED_MEDIA_TEXT_BYTES, suffix=_STAGED_MEDIA_TRUNCATED)
        for note in notes)
    available = state.MAX_PROMPT_BYTES - len((prompt + _STAGED_MEDIA_HEADER).encode("utf-8"))
    if available <= len(_STAGED_MEDIA_TRUNCATED.encode("utf-8")):
        raise RuntimeError(
            "staged voice transcripts do not fit this turn's prompt budget")
    return prompt + _STAGED_MEDIA_HEADER + _truncate_utf8_text(
        body, max_bytes=available, suffix=_STAGED_MEDIA_TRUNCATED)


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
AttachmentLoader = Callable[["HostedRoomBinding", Mapping[str, Any]], Iterable[tuple[Mapping[str, Any], bytes]]]
HistoryContextLoader = Callable[
    ["HostedRoomBinding", Mapping[str, Any]], tuple[str, Mapping[str, Any], bytes] | None]


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
        attachment_loader: AttachmentLoader | None = None,
        history_context_loader: HistoryContextLoader | None = None,
        clock: Callable[[], float] = time.time,
        lease_ttl_seconds: float = 30.0, poll_interval_seconds: float = 5.0,
        active_poll_interval_seconds: float = 0.25, turn_timeout_seconds: float = 1830.0,
        indeterminate_defer_seconds: float = 60.0, max_concurrent_rooms: int = 4,
        unavailable_retry_min_seconds: float = 1.0, unavailable_retry_max_seconds: float = 30.0,
        process_generation: str | None = None) -> None:
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
        self.attachment_loader = attachment_loader
        self.history_context_loader = history_context_loader
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
        self._unavailable_route_retries: dict[tuple[str, str], dict[str, float]] = {}
        self._blocked_rooms: set[str] = set()
        self._status_lock, self._current_tasks = threading.Lock(), {}
        # Attempts this process has started but not yet submitted. A Stop that lands in that
        # window must not read "no active session" as proof the turn stopped.
        self._unadmitted_attempts: dict[str, state.TaskIdentity] = {}
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
    def cancel(self, identity: state.TaskIdentity, *, cancel_id: str) -> dict[str, Any]:
        """Persist a stop intent, then commit cancellation after acknowledgement.

        The worker transitions tasks concurrently, so the status read is only a routing
        hint: a fast-path fence failure re-reads and re-routes instead of surfacing it.
        """
        for _ in range(_CANCEL_ROUTE_RETRIES):
            before = state.get_task(self.db_path, identity)
            if before["status"] == "cancelled":
                return before
            if before["status"] in state.TERMINAL_STATUSES:
                raise state.InvalidTaskTransitionError(
                    f"cannot cancel task in state '{before['status']}'")
            # Only work no transport ever accepted can be cancelled outright. A deferred attempt
            # that passed its admission fence may still be executing somewhere, so it takes the
            # acknowledged two-phase route like any other admitted attempt.
            direct = before["status"] == "queued" or (
                before["status"] == "deferred" and before["admitted_at"] is None)
            try:
                result = (state.cancel_task if direct else state.begin_task_cancel)(
                    self.db_path, identity, cancel_id=cancel_id,
                    expected_cancel_generation=before["cancel_generation"], clock=self.clock)
            except (state.InvalidTaskTransitionError, state.StaleTaskError):
                continue  # lost the race with the worker (settled or re-queued); re-route
            if not direct:
                binding = self._binding_for_room(identity.room_id)
                try:
                    if binding is not None:
                        lease = self._ensure_lease(binding)
                        if self._peer_stop_acknowledged(binding, result):
                            self._complete_cancel(result, cancel_id=cancel_id)
                        else:
                            # Same resolution the worker uses, so a stop requested here and one
                            # retried there agree on what counts as proof that the run is over.
                            self._finish_stop(binding, result, lease)
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

    def retry_indeterminate(self, identity: state.TaskIdentity) -> dict[str, Any]:
        """Explicitly retry one uncertain attempt under the current room lease."""
        task = state.get_task(self.db_path, identity)
        if task["status"] not in {"indeterminate", "deferred"}:
            raise state.InvalidTaskTransitionError(f"cannot retry task in state '{task['status']}'")
        binding = self._binding_for_room(identity.room_id)
        if binding is None:
            raise state.RoomUnavailableError("hosted room is unavailable")
        lease = self._ensure_lease(binding)
        uncertain = str(task["status"])
        # Explicit Retry may resume the exact stored session; the automatic abandoned-attempt
        # scan stays non-resuming for local sessions. A deferral only fences publication, so a
        # deferred attempt is probed the same way, and fails closed when the probe cannot answer:
        # an unreachable member cannot establish that its attempt stopped.
        try:
            inspection = self._inspect_recovery_session(binding, task)
        except Exception as exc:
            if uncertain != "deferred":
                raise
            self._set_blocked(identity.room_id, True)
            raise state.InvalidTaskTransitionError(
                "cannot confirm the deferred task attempt stopped; retry once its member "
                f"answers ({exc})") from exc
        # Harvest the exact completion before considering any replay: a receipt for this attempt's
        # own generation is proof the execution ended, in either uncertain state and whichever
        # process ran it.
        if inspection.terminal is not None:
            return self._resolve_indeterminate(
                binding, task, lease, inspection.terminal, from_status=uncertain)
        if inspection.status == "cancelled":
            return self._fenced(
                state.resolve_indeterminate_cancellation, binding, task, lease,
                cancel_id=f"remote-cancel:{task['execution_generation']}", from_status=uncertain)
        if inspection.active:
            self._set_blocked(identity.room_id, True)
            raise state.InvalidTaskTransitionError(
                "cannot retry while the original task attempt is still active")
        # Owner death is the LAST authority consulted: an exact receipt, a transport-reported
        # cancellation and an observably active attempt all decide above this line, exactly as
        # before. It only answers the case those cannot -- an admitted attempt whose owner can no
        # longer speak for itself.
        proof = self._owner_death_proof(task)
        if proof is not None and self._stop_intent_recorded(task):
            # This attempt was stopped and its owner is gone: resolving the stop it already
            # carries is the truthful outcome, and replaying work the user stopped would not be.
            return self._resolve_dead_owner_stop(binding, task, lease, proof, uncertain)
        requeue = (
            state.requeue_deferred_task if uncertain == "deferred"
            else state.requeue_indeterminate_task)
        return self._requeue(
            requeue, task, lease, identity.room_id, owner_death_proof=proof)

    def _resolve_dead_owner_stop(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease,
        proof: state.OwnerDeathProof, from_status: str) -> dict[str, Any]:
        """Terminalize an uncertain attempt that was stopped and whose owner is proven ended.

        Same outcome rule as the ``stopping`` route: a deadline stop keeps its explicit failure and
        its reason, a user Stop stays a cancellation. Both validate the proof in their own
        transaction; neither has an unguarded fallback."""
        death = self._death_route(proof)
        if str(task.get("cancel_id") or "").startswith("deadline:"):
            return self._fenced(
                state.resolve_indeterminate_task, binding, task, lease, from_status=from_status,
                settlement_id=f"deadline:{int(task['execution_generation'])}", status="failed",
                result={
                    "error": "This Group Chat turn exceeded its configured time limit and was stopped.",
                    "reason_code": "turn_deadline_exceeded",
                    "timeout_seconds": self.turn_timeout_seconds}, **death)
        return self._fenced(
            state.resolve_indeterminate_cancellation, binding, task, lease,
            cancel_id=str(task["cancel_id"]), from_status=from_status, **death)

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
        lease: state.DriverLease, room_id: str, **extra: Any) -> dict[str, Any]:
        retried = self._fenced(requeue, None, task, lease, **extra)
        self._set_blocked(room_id, False)
        self.wakeup()
        return retried

    def _complete_cancel(
        self, task: Mapping[str, Any], *, cancel_id: str | None = None, **death: Any
    ) -> dict[str, Any]:
        return state.complete_task_cancel(
            self.db_path, task["identity"], clock=self.clock,
            cancel_id=task["cancel_id"] if cancel_id is None else cancel_id,
            expected_cancel_generation=task["cancel_generation"], **death)

    @staticmethod
    def _death_route(proof: state.OwnerDeathProof | None) -> dict[str, Any]:
        """The keywords that turn one terminal transition into its death-authorized form."""
        return {} if proof is None else {
            "death_authorized": True, "owner_death_proof": proof}

    def _stop_intent_recorded(self, task: Mapping[str, Any]) -> bool:
        """Whether THIS attempt was really asked to stop, not merely handed an inherited counter.

        The storage guard re-checks this inside the transaction; the runtime uses it only to choose
        between resolving a stop and replaying the turn."""
        descriptor = task.get("owner_descriptor")
        if not str(task.get("cancel_id") or "") or not isinstance(descriptor, dict):
            return False
        admitted = descriptor.get("cancel_generation_at_admission")
        return isinstance(admitted, int) and int(task["cancel_generation"]) > admitted

    def _resolve_indeterminate(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease,
        terminal: _TerminalReceipt, *, publish: bool = True,
        from_status: str = "indeterminate") -> dict[str, Any]:
        return self._fenced(
            state.resolve_indeterminate_task, binding, task, lease, publish=publish,
            from_status=from_status, **asdict(terminal))

    def _finish_stop(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease
    ) -> bool:
        """Terminalize a stopping task from its receipt or an acknowledged interrupt."""
        if self._settle_stopping_completion(binding, task, lease):
            return True
        if self._interrupt_stopping_task(binding, task):
            self._complete_acknowledged_stop(binding, task, lease)
            return True
        if self._dispatch_cannot_be_excluded(task):
            if (proof := self._owner_death_proof(task)) is not None:
                # Owner death is an ADDITIONAL cessation authority, reached only after the receipt
                # and native-acknowledgement routes above declined. It establishes that the run is
                # over; it does not choose the outcome, which stays whatever the recorded stop
                # identity says (a deadline stop is still `failed` with its own reason, a user Stop
                # still `cancelled`).
                self._complete_acknowledged_stop(binding, task, lease, owner_death_proof=proof)
                return True
            # Neither a receipt nor an interruption: an attempt admitted by another process may
            # still dispatch, so this resolves as uncertain rather than claiming it stopped. The
            # ordinary indeterminate path then owns it (probe, explicit retry, no auto-resubmit),
            # and a Stop repeated once the session is observably running interrupts it for real.
            self._fenced(state.mark_stop_uncertain, binding, task, lease, publish=False)
            self._set_blocked(binding.room_id, True)
            return True
        return False

    def _dispatch_cannot_be_excluded(self, task: Mapping[str, Any]) -> bool:
        """Whether an admitted attempt belongs to a run this process cannot speak for."""
        return (
            task.get("admitted_at") is not None
            and task.get("run_process_generation") != self.process_generation)

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
            and _info_identifies_attempt(info, task))

    def _interrupt_stopping_task(self, binding: HostedRoomBinding, task: Mapping[str, Any]) -> bool:
        # An idle or absent session is only evidence of cessation for an attempt whose dispatch
        # this process can account for; interrupting an observably running one is proof for
        # anybody, so the probes below still run.
        idle_is_proof = not self._stop_must_stay_provisional(task)
        transport, profile, session_id = self._open_session(binding, task)
        if session_id is None:
            # A local turn cannot survive without its canonical session, so an authoritative
            # absence is a safe Stop acknowledgement (errors raise); a peer stays uncertain.
            return idle_is_proof and transport is not None and transport is self.rpc
        info = transport.info(**_session_kw(profile, session_id))
        if not _info_active(info):
            # History was checked just before this probe: an inactive exact session cannot
            # keep executing, and after a restart its process-local task marker is absent.
            return idle_is_proof
        if not _info_is_active_for(info, task["identity"], require_exact=True):
            return False
        result = transport.interrupt(
            **_session_kw(profile, session_id), expected_task_id=task["identity"].task_id,
            expected_execution_generation=int(task["execution_generation"]))
        if result is None or not (
            result.get("interrupted") is True
            or str(result.get("status") or "") in _STOP_ACK_STATUSES):
            return False
        if str(result.get("status") or "") == "cancelled":
            return True  # a terminal run status is the transport's proof that the run is over
        # "interrupted" only reports that the request was accepted: the native run thread can
        # still be alive. Re-probe the exact session, and while it keeps running this task the
        # Stop stays provisional, so a later cycle finds its receipt or a settled session.
        return not _info_is_active_for(
            transport.info(**_session_kw(profile, session_id)), task["identity"], require_exact=True)

    def _settle_stopping_completion(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease
    ) -> bool:
        """Publish a terminal receipt that arrived before Stop was acknowledged."""
        transport, profile, session_id = self._open_session(binding, task)
        if session_id is None:
            return False
        receipt = self._terminal_from_history(transport, profile, session_id, task)
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
            if self._stop.is_set():
                return
            if retry is not None and self.clock() < retry["next_attempt_at"]:
                continue
            lease = self._renew_lease_if_needed(lease)
            attempt = state.start_task(
                self.db_path, task["identity"], lease,
                expected_cancel_generation=task["cancel_generation"], clock=self.clock)
            self._execute_attempt(binding, task, attempt)
            current = state.get_task(self.db_path, task["identity"])
            if current["status"] not in state.TERMINAL_STATUSES:
                retry = self._unavailable_route_retries.get((task["identity"].room_id, _member_id(task)))
                if current["status"] == "queued" and retry is not None and self.clock() < retry["next_attempt_at"]:
                    continue
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
        self, binding: HostedRoomBinding, task: Mapping[str, Any], attempt: state.TaskAttempt
    ) -> None:
        profile, submit_attempted = task["payload"]["target_profile"], False
        transport = None
        attachment_staging_active = False
        attachment_session_id = None
        with self._status_lock:
            self._current_tasks[binding.room_id] = attempt.identity
            self._unadmitted_attempts[binding.room_id] = attempt.identity
        try:
            transport = self._transport_for(binding, task)
            with self.turn_lock(profile):
                session = self._resolve_or_create(transport, profile, binding.room_id)
                # Resolution is slow (profile lock, session resume) and a Stop can commit inside
                # it. This is the final admission boundary: one fenced write on the same row the
                # cancellation writes, so whichever commits first decides. A Stop that got there
                # first raises here and no prompt is submitted; the unadmitted marker above keeps
                # a Stop that arrives after it from calling this window a stopped execution.
                if self._stop.is_set():
                    return
                session_id = _session_id(session)
                prompt = str(task["payload"]["prompt"])
                manifests = task["payload"].get("attachments") or []
                history = self.history_context_loader(binding, task) if self.history_context_loader else None
                if manifests or history:
                    if manifests and self.attachment_loader is None:
                        raise RuntimeError("hosted attachments are unavailable for this member transport")
                    transport.begin_attachment_staging(
                        **_session_kw(profile, session_id), execution_generation=attempt.execution_generation)
                    attachment_staging_active = True
                    attachment_session_id = session_id
                    expected_ids = [str(attachment.get("attachment_id") or "") for attachment in manifests]
                    from itertools import chain
                    loaded = (self.attachment_loader(binding, task)
                              if manifests and self.attachment_loader is not None else ())
                    if history:
                        from gateway.hosted_room_attachments import validate_task_manifest
                        validate_task_manifest([*manifests, dict(history[1])])
                        expected_ids.append(history[1]["attachment_id"])
                        loaded = chain(loaded, [(history[1], history[2])])
                    file_refs = []
                    history_ref = ""
                    media_notes = []
                    loaded_count = 0
                    for loaded_count, (attachment, data) in enumerate(loaded, start=1):
                        if (loaded_count > len(expected_ids)
                                or str(attachment.get("attachment_id") or "") != expected_ids[loaded_count - 1]):
                            raise RuntimeError("hosted attachment ownership did not match the task manifest")
                        staged = transport.stage_attachment(
                            **_session_kw(profile, session_id), attachment=attachment, data=data,
                            execution_generation=attempt.execution_generation)
                        if history and attachment["attachment_id"] == history[1]["attachment_id"]:
                            from tui_gateway.hosted_room_history_context import history_file_reference
                            history_ref = history_file_reference(staged)
                            continue  # Do not inline history through native @file preprocessing.
                        if attachment.get("kind") == "file":
                            ref = str(staged.get("ref_text") or "").strip()
                            if not ref and transport is self.rpc:
                                raise RuntimeError("hosted file attachment returned no staged reference")
                            if ref:
                                file_refs.append(f"{attachment['name']}: {ref}")
                            derived = str(staged.get("derived_text") or "").strip()
                            if derived:
                                media_notes.append(f"{attachment['name']}: {derived}")
                    if loaded_count != len(expected_ids):
                        raise RuntimeError("hosted attachment ownership did not match the task manifest")
                    if file_refs:
                        prompt = f"{prompt}\n\nAttached files staged in your session workspace:\n" + "\n".join(file_refs)
                    prompt = _with_staged_media_text(prompt, media_notes)
                    if history:
                        from tui_gateway.hosted_room_history_context import MAX_HISTORY_INDEX_BYTES
                        if len(history[0].encode("utf-8")) > MAX_HISTORY_INDEX_BYTES:
                            raise RuntimeError("legacy history index exceeds its byte limit")
                        prompt = f"{prompt}\n\n{history[0]}\n{history_ref}"
                        if len(prompt.encode("utf-8")) > state.MAX_PROMPT_BYTES:
                            raise RuntimeError("legacy history references exceed the remaining prompt budget")
                # Captured before the fence and written BY it: the descriptor and `admitted_at`
                # land in the same CAS, so a rejected admission stores neither and submits
                # nothing. An invalid target raises here, still before any dispatch.
                owner_descriptor = self._capture_owner_descriptor(transport, profile, session_id)
                try:
                    state.fence_task_admission(
                        self.db_path, attempt, clock=self.clock,
                        owner_descriptor=owner_descriptor)
                except state.StaleTaskError:
                    # This `return` leaves the enclosing block without reaching the outer handlers,
                    # so staged media is released HERE or it leaks into the next accepted attempt.
                    if attachment_staging_active:
                        self._finish_attachment_staging_after_error(
                            transport=transport, profile=profile, session_id=session_id,
                            execution_generation=attempt.execution_generation,
                            submit_attempted=False)
                        attachment_staging_active = False
                    self._record_task_error(
                        attempt, "was stopped before admission; no prompt was submitted")
                    return
                # A submit should fail before admission or return after it; an unexpected
                # exception at that boundary is ambiguous, never a proven failure.
                submit_attempted = True
                deadline_monotonic = time.monotonic() + self.turn_timeout_seconds
                transport.submit(
                    **_session_kw(profile, session_id), prompt=prompt,
                    task=attempt.identity, execution_generation=attempt.execution_generation,
                    on_terminal=lambda receipt: self._on_terminal(binding, attempt, receipt))
                if attachment_staging_active:
                    transport.commit_attachment_staging(
                        **_session_kw(profile, session_id), execution_generation=attempt.execution_generation)
                    attachment_staging_active = False
                self._clear_unadmitted(binding.room_id, attempt)
                self._unavailable_route_retries.pop(
                    (task["identity"].room_id, _member_id(task)), None)
                receipt = self._wait_for_terminal(
                    binding, profile=profile, session_id=session_id, attempt=attempt,
                    transport=transport, deadline_monotonic=deadline_monotonic)
                if receipt is None:
                    return
                state.settle_task(self.db_path, attempt, **asdict(receipt), clock=self.clock)
        except (state.StaleLeaseError, state.StaleTaskError) as exc:
            if transport is not None and attachment_staging_active and attachment_session_id is not None:
                self._finish_attachment_staging_after_error(
                    transport=transport, profile=profile, session_id=attachment_session_id,
                    execution_generation=attempt.execution_generation, submit_attempted=submit_attempted)
            self._drop_lease(binding.room_id)
            self._record_task_error(attempt, f"fenced: {exc}")
        except Exception as exc:
            if transport is not None and attachment_staging_active and attachment_session_id is not None:
                self._finish_attachment_staging_after_error(
                    transport=transport, profile=profile, session_id=attachment_session_id,
                    execution_generation=attempt.execution_generation, submit_attempted=submit_attempted,
                    not_admitted=bool(getattr(exc, "not_admitted", False)))
            if bool(getattr(exc, "not_admitted", False)):
                try:
                    if task.get("payload", {}).get("target_member_id"):
                        deferred = state.defer_not_admitted_task(
                            self.db_path, attempt, reason="member_unavailable", clock=self.clock)
                    else:
                        deferred = None
                        state.requeue_not_admitted_task(self.db_path, attempt, clock=self.clock)
                except (state.StaleLeaseError, state.StaleTaskError) as fence_exc:
                    self._mark_ambiguous(binding, attempt)
                    self._record_task_error(
                        attempt, f"not-admitted proof lost its fence: {fence_exc}")
                else:
                    delay = self._defer_unavailable_route(task)
                    if deferred is not None and self.publish_terminal is not None:
                        self.publish_terminal(binding, deferred)
                    self._record_task_error(
                        attempt, "was not admitted; " + (f"member deferred for {delay:g}s"
                                                        if deferred is not None else f"queued for retry in {delay:g}s"))
            elif submit_attempted:
                self._mark_ambiguous(binding, attempt)
                self._record_task_error(attempt, f"observation failed after submit: {exc}")
            else:
                self._settle_failure_if_current(attempt, exc)
        finally:
            self._clear_unadmitted(binding.room_id, attempt)
            with self._status_lock:
                self._current_tasks.pop(binding.room_id, None)
                # The task may have published a reply or exposed the next turn while this
                # thread held its slot: schedule exactly one follow-up after it leaves
                # (idle room scans never set this marker).
                self._rooms_needing_reschedule.add(binding.room_id)

    def _capture_owner_descriptor(
        self, transport: InternalSessionRPC, profile: str, session_id: str
    ) -> Mapping[str, Any] | None:
        """The native half of this attempt's owner descriptor, or ``None``.

        The capability is discovered at this one call site rather than declared on
        ``InternalSessionRPC``: a Protocol member is not optional just because it may return
        ``None``, and requiring it would break every peer transport and test double that satisfies
        the contract structurally today.

        ``None`` means "no descriptor", which costs only death-recovery eligibility -- peers and
        unsupported platforms keep submitting exactly as they do now. An INVALID target (no such
        record, or a profile that is not this member's) is different: the adapter raises, and that
        propagates before the admission fence, so nothing is dispatched.
        """
        capture = getattr(transport, "execution_identity", None)
        if not callable(capture):
            return None
        return capture(profile=profile, session_id=session_id)

    def _owner_death_proof(self, task: Mapping[str, Any]) -> state.OwnerDeathProof | None:
        """This successor's own proof that ``task``'s recorded owner incarnation ended.

        Never a client value and never a runtime predicate: idle sessions, absent process-local
        markers, released active-session locks, expired leases and process-generation equality are
        all excluded by construction, because only the strict incarnation probe can answer
        ``ended``.
        """
        return state.owner_death_proof(task, probe_owner_incarnation(task.get("owner_descriptor")))

    def _clear_unadmitted(self, room_id: str, attempt: state.TaskAttempt) -> None:
        with self._status_lock:
            if self._unadmitted_attempts.get(room_id) == attempt.identity:
                self._unadmitted_attempts.pop(room_id, None)

    def _owns_pre_submit_window(self, task: Mapping[str, Any]) -> bool:
        """Whether this process holds ``task`` between its admission fence and its submit call."""
        identity = task["identity"]
        with self._status_lock:
            return self._unadmitted_attempts.get(identity.room_id) == identity

    def _stop_must_stay_provisional(self, task: Mapping[str, Any]) -> bool:
        """Whether an idle or absent session is no proof that this attempt stopped.

        An attempt that never passed its admission fence handed nothing to a transport, so its
        absence is a safe acknowledgement. Once admitted, an idle session only proves the run is
        over for the process that owns the run: inside our own pre-submit window the dispatch has
        not happened yet, and a foreign owner may be in exactly that window too -- a lease that
        expired says its holder should stop, never that its thread already did.
        """
        return task.get("admitted_at") is not None and (
            self._owns_pre_submit_window(task)
            or task.get("run_process_generation") != self.process_generation)

    def _mark_ambiguous(self, binding: HostedRoomBinding, attempt: state.TaskAttempt) -> None:
        self._drop_lease(binding.room_id)
        self._ambiguous_rooms[binding.room_id] = attempt.lease.expires_at

    def _on_terminal(
        self, binding: HostedRoomBinding, attempt: state.TaskAttempt, receipt: Mapping[str, Any]
    ) -> None:
        """Durably commit one in-process terminal receipt for ``attempt``."""
        status = receipt.get("status")
        if status == "cancelled":
            self.wakeup()
            return
        terminal = _TerminalReceipt(
            status="settled" if status == "settled" else "failed",
            settlement_id=receipt.get("settlement_id")
            or f"reply:{attempt.identity.task_id}:{attempt.execution_generation}",
            result=_bounded_terminal_result(receipt))
        try:
            self._publish(
                binding,
                state.settle_task(self.db_path, attempt, **asdict(terminal), clock=self.clock))
        except state.StaleTaskError:
            with suppress(state.StaleLeaseError, state.StaleTaskError):
                current = state.get_task(self.db_path, attempt.identity)
                if current["status"] == "stopping":
                    self._fenced(
                        state.settle_stopping_task, binding, current, attempt.lease,
                        **asdict(terminal),
                        expected_execution_generation=attempt.execution_generation)
        except state.StaleLeaseError:
            # Cancellation, disband, or authority transfer won the durable race: the model
            # result is discarded rather than turning a correct fence into a thread exception.
            pass
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
                self._wake.wait(self.active_poll_interval_seconds)
                self._wake.clear()
                continue
            if time.monotonic() >= deadline_monotonic:
                self._expire_attempt_deadline(binding, task, lease)
                return None
            lease = self._renew_lease_if_needed(lease)
            receipt = self._terminal_from_history(transport, profile, session_id, task)
            if receipt is not None:
                return receipt
            info = transport.info(**_session_kw(profile, session_id))
            self._report_pending_action(task, session_id=session_id, info=info)
            remaining = max(0.0, deadline_monotonic - time.monotonic())
            self._wake.wait(min(self.active_poll_interval_seconds, remaining))
            self._wake.clear()
        return None

    def _complete_acknowledged_stop(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], lease: state.DriverLease,
        *, owner_death_proof: state.OwnerDeathProof | None = None) -> dict[str, Any]:
        """Terminalize an acknowledged Stop: deadline stops publish an explicit failure.

        The branch is on the RECORDED stop identity, so how cessation was established -- a native
        acknowledgement or a proven-dead owner -- never relabels a timeout as a user cancellation
        or the reverse."""
        death = self._death_route(owner_death_proof)
        if not str(task.get("cancel_id") or "").startswith("deadline:"):
            return self._complete_cancel(task, **death)
        return self._fenced(
            state.settle_stopping_task, binding, task, lease, **death,
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
        # A user Stop that won the race keeps its own cancellation semantics.
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
        # A reported terminal status only says something about THIS attempt when the transport
        # names it: the peer transport scopes its status request and answers with the exact
        # task and execution generation. An unbound or mismatched status is not proof.
        status = str(info.get("status") or "") or None
        return _RecoveryInspection(
            terminal=receipt, active=_info_is_active_for(info, task["identity"]),
            status=status if status is not None and _info_identifies_attempt(info, task) else None)

    def _inspect_recovery_session(
        self, binding: HostedRoomBinding, task: Mapping[str, Any]) -> _RecoveryInspection:
        profile, transport = task["payload"]["target_profile"], self._transport_for(binding, task)
        with self.turn_lock(profile):
            session_id = self._resume_exact(transport, task["identity"].room_id, profile)
            if session_id is None:
                return _NO_INSPECTION
            return self._inspect_session(transport, task, session_id, read_history=True)

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
            # Deferring means "this member is not working on it": re-establish that now. The
            # inspection above is recorded once per attempt, so without this check an execution
            # that was observed ALIVE at recovery would be deferred later purely because its
            # deadline passed -- and a deferred task reads as work nobody is running.
            if inspection.active or (is_local and self._inspect_local_recovery_session(task).active):
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
                status="failed", result={"error": "Group Chat member turn failed."}, clock=self.clock)
        self._record_task_error(attempt, "failed")

    def _record_task_error(self, attempt: state.TaskAttempt, message: str) -> None:
        self._record_error(f"task {attempt.identity.task_id} {message}")

    def _record_error(self, message: str) -> None:
        with self._status_lock:
            self._last_error = message


    def _finish_attachment_staging_after_error(
        self,
        *,
        transport: InternalSessionRPC,
        profile: str,
        session_id: str,
        execution_generation: int,
        submit_attempted: bool,
        not_admitted: bool = False,
    ) -> None:
        """Close staging without masking the task's original failure.

        An unclassified submit exception remains ambiguous and keeps staged
        bytes available to a turn that may have been accepted. Positive
        ``not_admitted`` proof rolls back just like a pre-submit failure, so a
        refused turn cannot leak attachments into the next canonical prompt.
        """

        try:
            finalizer = (
                transport.commit_attachment_staging
                if submit_attempted and not not_admitted
                else transport.rollback_attachment_staging
            )
            finalizer(
                profile=profile,
                session_id=session_id,
                source=ROOM_SESSION_SOURCE,
                execution_generation=execution_generation,
            )
        except Exception as cleanup_error:
            self._record_error(
                "attachment staging cleanup failed for "
                f"session {session_id}: {cleanup_error}"
            )


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


def _info_is_active_for(
    info: Mapping[str, Any], identity: state.TaskIdentity, *, require_exact: bool = False) -> bool:
    accepted = (identity.task_id,) if require_exact else (None, identity.task_id)
    return _info_active(info) and info.get("task_id") in accepted


def _info_identifies_attempt(info: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    """Whether a transport's report names this exact task attempt and its execution generation."""
    return (
        str(info.get("task_id") or "") == task["identity"].task_id
        and int(info.get("execution_generation") or 0) == int(task["execution_generation"]))


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


class MemberTransportUnavailable(RuntimeError):
    """A member route rejected work before any remote admission occurred."""

    not_admitted = True
