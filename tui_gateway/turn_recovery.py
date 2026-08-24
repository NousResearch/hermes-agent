"""Turn-recovery v2 journal and capability advertisement.

Background
----------
The Hermes Android client can survive being backgrounded mid-turn: it keeps a
``client_turn_id``, and on resume it asks the server to replay everything that
happened while it was away. That handshake only engages when the gateway
advertises a ``turn_recovery`` capability in its ``gateway.ready`` frame.

The client's parser is deliberately *fail-closed*
(``lib/core/models/gateway_turn_contract.dart``): if any field is missing,
mistyped, or internally inconsistent, it silently disables recovery and falls
back to the legacy transport rather than surfacing an error. That makes a
malformed advertisement strictly worse than no advertisement at all — the app
would negotiate a route the server cannot honour, and the failure would be
invisible. Everything in this module is therefore built to be exact.

Scope
-----
This module owns the bounded in-process journal and exact capability dict. The
separate :mod:`tui_gateway.turn_recovery_runtime` adapter wires those primitives
into ``gateway.ready`` and the JSON-RPC method table only after every advertised
method has been installed, so the client never negotiates a partial route.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from collections import deque

__all__ = [
    "ATTACHMENT_DETACH_METHOD",
    "CAPABILITY_VERSION",
    "CORE_METHODS",
    "EXECUTION_ROUTE",
    "PROMPT_SUBMIT_VERSION",
    "PROTOCOL_MAJOR",
    "PROTOCOL_NAME",
    "TERMINAL_STATUSES",
    "TURN_STATUSES",
    "TurnJournal",
    "TurnHandle",
    "TurnLimits",
    "TurnRecoveryError",
    "build_protocol_frame",
    "build_turn_recovery_capability",
]

# --- Wire constants -------------------------------------------------------
# These mirror the client's compile-time constants exactly. Changing any of
# them without a matching client release silently disables recovery.

PROTOCOL_NAME = "hermes-jsonrpc"
PROTOCOL_MAJOR = 2
CAPABILITY_VERSION = 2
PROMPT_SUBMIT_VERSION = 2
EXECUTION_ROUTE = "single_process_in_process"
UUID_FORMAT = "canonical_lowercase_uuid"

SESSION_OPEN_METHOD = "session.open"
RECONCILE_METHOD = "turn.reconcile"
INTERRUPT_METHOD = "turn.interrupt"
ATTACHMENT_DETACH_METHOD = "attachment.detach@2"
PROMPT_SUBMIT_APPLIES_TO = "prompt.submit@2"

CORE_METHODS: Tuple[str, ...] = (
    SESSION_OPEN_METHOD,
    RECONCILE_METHOD,
    INTERRUPT_METHOD,
)

TURN_STATUSES = frozenset(
    {"accepted", "running", "waiting_input", "completed", "failed", "interrupted"}
)
TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted"})

_EVENT_TYPES = frozenset(
    {"message.start", "message.delta", "message.complete", "turn.status"}
)

_EMPTY_MANIFEST_DIGEST = "0" * 64
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class TurnRecoveryError(Exception):
    """Raised when a caller violates the turn-recovery contract."""


@dataclass(frozen=True)
class TurnLimits:
    """Byte/count/retention budgets advertised to the client and enforced here.

    The inequalities below are not stylistic: each one corresponds to a
    rejection branch in the client's capability parser. Validating them at
    construction means an operator misconfiguration fails loudly at startup
    instead of silently disabling recovery on every phone.
    """

    event_retention_seconds: int = 900
    turn_retention_seconds: int = 3600
    max_event_bytes: int = 64 * 1024
    max_turn_bytes: int = 4 * 1024 * 1024
    terminal_event_reserve_bytes: int = 32 * 1024
    max_prompt_bytes: int = 256 * 1024
    reconcile_max_events: int = 250
    reconcile_max_page_bytes: int = 512 * 1024
    max_attachments: int = 8
    max_file_attachment_bytes: int = 16 * 1024 * 1024
    max_image_attachment_bytes: int = 16 * 1024 * 1024
    max_pdf_attachment_bytes: int = 16 * 1024 * 1024
    max_attachment_registry_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        positive = (
            "event_retention_seconds",
            "turn_retention_seconds",
            "max_event_bytes",
            "max_turn_bytes",
            "terminal_event_reserve_bytes",
            "max_prompt_bytes",
            "reconcile_max_events",
            "reconcile_max_page_bytes",
            "max_attachments",
            "max_file_attachment_bytes",
            "max_image_attachment_bytes",
            "max_pdf_attachment_bytes",
            "max_attachment_registry_bytes",
        )
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

        if self.turn_retention_seconds < self.event_retention_seconds:
            raise ValueError(
                "turn_retention_seconds must be >= event_retention_seconds"
            )
        if self.terminal_event_reserve_bytes > self.max_event_bytes:
            raise ValueError(
                "terminal_event_reserve_bytes must be <= max_event_bytes"
            )
        if self.reconcile_max_page_bytes < self.max_event_bytes:
            raise ValueError("reconcile_max_page_bytes must be >= max_event_bytes")
        if self.max_turn_bytes <= self.terminal_event_reserve_bytes:
            raise ValueError(
                "max_turn_bytes must exceed terminal_event_reserve_bytes"
            )


def build_protocol_frame() -> Dict[str, Any]:
    """Return the ``protocol`` block for ``gateway.ready``."""

    return {"name": PROTOCOL_NAME, "major": PROTOCOL_MAJOR}


def build_turn_recovery_capability(
    *,
    limits: Optional[TurnLimits] = None,
    attachments_supported: bool = False,
) -> Dict[str, Any]:
    """Return the ``capabilities.turn_recovery`` block for ``gateway.ready``.

    ``attachments_supported`` must reflect a route that is genuinely
    implemented. The client treats the attachment method as all-or-nothing:
    advertising it without honouring ``attachment.detach@2`` breaks attachment
    sends rather than degrading them.
    """

    limits = limits or TurnLimits()

    methods: List[str] = list(CORE_METHODS)
    applies_to: List[str] = list(CORE_METHODS) + [PROMPT_SUBMIT_APPLIES_TO]
    if attachments_supported:
        methods.append(ATTACHMENT_DETACH_METHOD)
        applies_to.append(ATTACHMENT_DETACH_METHOD)

    capability: Dict[str, Any] = {
        "version": CAPABILITY_VERSION,
        "prompt_submit_version": PROMPT_SUBMIT_VERSION,
        "execution_route": EXECUTION_ROUTE,
        "mobile_session_id_format": UUID_FORMAT,
        "client_turn_id_format": UUID_FORMAT,
        # Recovery replays; it never re-runs a prompt on the user's behalf.
        "shadow_only": False,
        "automatic_resubmit": False,
        "methods": methods,
        "applies_to": applies_to,
        "event_retention_seconds": limits.event_retention_seconds,
        "turn_retention_seconds": limits.turn_retention_seconds,
        "max_event_bytes": limits.max_event_bytes,
        "max_turn_bytes": limits.max_turn_bytes,
        "terminal_event_reserve_bytes": limits.terminal_event_reserve_bytes,
        "max_prompt_bytes": limits.max_prompt_bytes,
        "reconcile_max_events": limits.reconcile_max_events,
        "reconcile_max_page_bytes": limits.reconcile_max_page_bytes,
    }
    if attachments_supported:
        capability.update(
            {
                "max_attachments": limits.max_attachments,
                "max_file_attachment_bytes": limits.max_file_attachment_bytes,
                "max_image_attachment_bytes": limits.max_image_attachment_bytes,
                "max_pdf_attachment_bytes": limits.max_pdf_attachment_bytes,
                "max_attachment_registry_bytes": limits.max_attachment_registry_bytes,
            }
        )
    return capability


@dataclass
class _Event:
    seq: int
    turn_id: str
    message_id: str
    type: str
    payload: Dict[str, Any]
    created_at: float
    nbytes: int

    def to_wire(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "seq": self.seq,
            "message_id": self.message_id,
            "type": self.type,
            "payload": dict(self.payload),
        }


@dataclass
class _Turn:
    turn_id: str
    session_id: str
    client_turn_id: str
    created_at: float
    updated_at: float
    status: str = "accepted"
    last_seq: int = 0
    next_seq: int = 1
    events: Deque[_Event] = field(default_factory=deque)
    total_bytes: int = 0
    assistant_message_id: Optional[str] = None
    assistant_text: str = ""
    final_message_ref: int = 1
    attachment_manifest_digest: str = _EMPTY_MANIFEST_DIGEST

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def earliest_seq(self) -> int:
        # With no retained events the window is empty; report the first seq the
        # client could still receive so ``after_seq >= earliest_seq - 1`` holds.
        return self.events[0].seq if self.events else self.last_seq + 1


@dataclass(frozen=True)
class TurnHandle:
    """Result of :meth:`TurnJournal.open_turn`."""

    turn_id: str
    client_turn_id: str
    status: str
    last_seq: int
    created: bool


@dataclass(frozen=True)
class EventHandle:
    """Result of :meth:`TurnJournal.append_event`."""

    turn_id: str
    seq: int


@dataclass(frozen=True)
class InterruptResult:
    """Result of :meth:`TurnJournal.interrupt`."""

    turn_id: str
    status: str
    last_seq: int


def _validate_payload(event_type: str, payload: Any) -> None:
    """Mirror the client's per-type payload validator exactly."""

    if event_type not in _EVENT_TYPES:
        raise TurnRecoveryError(f"unknown event type {event_type!r}")
    if not isinstance(payload, dict):
        raise TurnRecoveryError("event payload must be a mapping")

    if event_type == "message.start":
        if payload:
            raise TurnRecoveryError("message.start payload must be empty")
        return

    if event_type == "message.delta":
        if set(payload) != {"text"} or not isinstance(payload["text"], str):
            raise TurnRecoveryError(
                "message.delta payload must be exactly {'text': str}"
            )
        return

    if event_type == "message.complete":
        if set(payload) != {"text", "status"}:
            raise TurnRecoveryError(
                "message.complete payload must be exactly {'text', 'status'}"
            )
        if not isinstance(payload["text"], str):
            raise TurnRecoveryError("message.complete text must be a str")
        if payload["status"] not in TERMINAL_STATUSES:
            raise TurnRecoveryError(
                "message.complete status must be terminal"
            )
        return

    # turn.status
    if set(payload) != {"status"} or payload["status"] not in TURN_STATUSES:
        raise TurnRecoveryError(
            "turn.status payload must be exactly {'status': <known status>}"
        )


def _wire_bytes(event: Dict[str, Any]) -> int:
    return len(json.dumps(event, separators=(",", ":")).encode("utf-8"))


class TurnJournal:
    """In-memory, size- and time-bounded journal of recoverable turns.

    Thread-safe: the gateway appends events from agent worker threads while RPC
    handlers read from the socket thread.
    """

    def __init__(
        self,
        *,
        clock: Optional[Callable[[], float]] = None,
        limits: Optional[TurnLimits] = None,
    ) -> None:
        import time

        self._clock = clock or time.monotonic
        self._limits = limits or TurnLimits()
        self._lock = threading.RLock()
        self._turns: Dict[Tuple[str, str], _Turn] = {}

    @property
    def limits(self) -> TurnLimits:
        return self._limits

    # -- lookup ------------------------------------------------------------

    def _require(self, session_id: str, client_turn_id: str) -> _Turn:
        turn = self._turns.get((session_id, client_turn_id))
        if turn is None:
            raise TurnRecoveryError(
                f"unknown turn for session={session_id!r} "
                f"client_turn_id={client_turn_id!r}"
            )
        return turn

    def get_turn(self, session_id: str, client_turn_id: str) -> _Turn:
        with self._lock:
            return self._require(session_id, client_turn_id)

    def resolve_turn(
        self,
        session_id: str,
        *,
        client_turn_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> _Turn:
        """Resolve one turn by either client id or server id, scoped to session."""

        with self._lock:
            if bool(client_turn_id) == bool(turn_id):
                raise TurnRecoveryError(
                    "provide exactly one of client_turn_id or turn_id"
                )
            if client_turn_id:
                return self._require(session_id, client_turn_id)
            for (candidate_session_id, _), turn in self._turns.items():
                if candidate_session_id == session_id and turn.turn_id == turn_id:
                    return turn
            raise TurnRecoveryError(
                f"unknown turn for session={session_id!r} turn_id={turn_id!r}"
            )

    def discard_turn(self, session_id: str, client_turn_id: str) -> None:
        """Forget a submission that the underlying prompt handler rejected."""

        with self._lock:
            self._turns.pop((session_id, client_turn_id), None)

    # -- mutation ----------------------------------------------------------

    def open_turn(self, *, session_id: str, client_turn_id: str) -> TurnHandle:
        """Create a turn, or return the existing one for this client turn id.

        Idempotency is the whole point: a client that resends ``prompt.submit``
        after a reconnect must land on the same turn instead of running the
        prompt twice.
        """

        with self._lock:
            key = (session_id, client_turn_id)
            existing = self._turns.get(key)
            if existing is not None:
                return TurnHandle(
                    turn_id=existing.turn_id,
                    client_turn_id=client_turn_id,
                    status=existing.status,
                    last_seq=existing.last_seq,
                    created=False,
                )
            now = self._clock()
            turn = _Turn(
                turn_id=str(uuid.uuid4()),
                session_id=session_id,
                client_turn_id=client_turn_id,
                created_at=now,
                updated_at=now,
            )
            self._turns[key] = turn
            return TurnHandle(
                turn_id=turn.turn_id,
                client_turn_id=client_turn_id,
                status=turn.status,
                last_seq=turn.last_seq,
                created=True,
            )

    def append_event(
        self,
        session_id: str,
        client_turn_id: str,
        event_type: str,
        message_id: str,
        payload: Dict[str, Any],
    ) -> EventHandle:
        with self._lock:
            turn = self._require(session_id, client_turn_id)
            if turn.is_terminal:
                raise TurnRecoveryError(
                    f"turn {turn.turn_id} is terminal ({turn.status}); "
                    "no further events may be appended"
                )
            # Validate before consuming a sequence number so a rejected event
            # cannot punch a hole in the client's contiguity check.
            _validate_payload(event_type, payload)
            if not isinstance(message_id, str) or not message_id:
                raise TurnRecoveryError("message_id must be a non-empty str")

            seq = turn.next_seq
            wire = {
                "turn_id": turn.turn_id,
                "seq": seq,
                "message_id": message_id,
                "type": event_type,
                "payload": dict(payload),
            }
            nbytes = _wire_bytes(wire)
            if nbytes > self._limits.max_event_bytes:
                raise TurnRecoveryError(
                    f"event of {nbytes} bytes exceeds max_event_bytes "
                    f"({self._limits.max_event_bytes})"
                )

            is_terminal_event = event_type == "message.complete"
            self._evict_for(turn, nbytes, is_terminal_event=is_terminal_event)

            now = self._clock()
            event = _Event(
                seq=seq,
                turn_id=turn.turn_id,
                message_id=message_id,
                type=event_type,
                payload=dict(payload),
                created_at=now,
                nbytes=nbytes,
            )
            turn.events.append(event)
            turn.total_bytes += nbytes
            turn.next_seq = seq + 1
            turn.last_seq = seq
            turn.updated_at = now

            # Maintain the snapshot projection so a turn stays answerable once
            # its individual events age out.
            if event_type in {"message.start", "message.delta", "message.complete"}:
                if turn.assistant_message_id != message_id:
                    turn.assistant_message_id = message_id
                    turn.assistant_text = ""
                if event_type == "message.delta":
                    turn.assistant_text += payload["text"]
                elif event_type == "message.complete":
                    turn.assistant_text = payload["text"]

            return EventHandle(turn_id=turn.turn_id, seq=seq)

    def _evict_for(
        self, turn: _Turn, incoming_bytes: int, *, is_terminal_event: bool
    ) -> None:
        """Drop oldest events until ``incoming_bytes`` fits the turn budget.

        A slice of ``max_turn_bytes`` is reserved so the terminal event always
        fits. Without it a chatty turn could evict its own completion frame and
        leave the client waiting forever on a turn that already finished.
        """

        budget = self._limits.max_turn_bytes
        if not is_terminal_event:
            budget -= self._limits.terminal_event_reserve_bytes
        while turn.events and turn.total_bytes + incoming_bytes > budget:
            oldest = turn.events.popleft()
            turn.total_bytes -= oldest.nbytes

    def set_status(self, session_id: str, client_turn_id: str, status: str) -> None:
        with self._lock:
            turn = self._require(session_id, client_turn_id)
            if status not in TURN_STATUSES:
                raise TurnRecoveryError(f"unknown turn status {status!r}")
            if turn.is_terminal and status != turn.status:
                raise TurnRecoveryError(
                    f"turn {turn.turn_id} already terminal ({turn.status})"
                )
            turn.status = status
            turn.updated_at = self._clock()

    def set_final_message_ref(
        self, session_id: str, client_turn_id: str, ref: int
    ) -> None:
        with self._lock:
            turn = self._require(session_id, client_turn_id)
            if not isinstance(ref, int) or isinstance(ref, bool) or ref <= 0:
                raise TurnRecoveryError("final_message_ref must be a positive int")
            turn.final_message_ref = ref

    def set_attachment_manifest_digest(
        self, session_id: str, client_turn_id: str, digest: str
    ) -> None:
        with self._lock:
            turn = self._require(session_id, client_turn_id)
            if not isinstance(digest, str) or not _HEX_DIGEST_RE.match(digest):
                raise TurnRecoveryError(
                    "attachment_manifest_digest must be 64 lowercase hex chars"
                )
            turn.attachment_manifest_digest = digest

    def interrupt(self, session_id: str, client_turn_id: str) -> InterruptResult:
        """Interrupt a running turn; idempotent once the turn is terminal."""

        with self._lock:
            turn = self._require(session_id, client_turn_id)
            if turn.is_terminal:
                return InterruptResult(
                    turn_id=turn.turn_id,
                    status=turn.status,
                    last_seq=turn.last_seq,
                )
            message_id = turn.assistant_message_id or turn.turn_id
            self.append_event(
                session_id,
                client_turn_id,
                "turn.status",
                message_id,
                {"status": "interrupted"},
            )
            turn.status = "interrupted"
            turn.updated_at = self._clock()
            return InterruptResult(
                turn_id=turn.turn_id,
                status=turn.status,
                last_seq=turn.last_seq,
            )

    # -- retention ---------------------------------------------------------

    def prune(self) -> None:
        """Drop expired events and turns. Safe to call repeatedly."""

        with self._lock:
            now = self._clock()
            event_cutoff = now - self._limits.event_retention_seconds
            turn_cutoff = now - self._limits.turn_retention_seconds

            for key in list(self._turns):
                turn = self._turns[key]
                if turn.updated_at < turn_cutoff:
                    del self._turns[key]
                    continue
                while turn.events and turn.events[0].created_at < event_cutoff:
                    oldest = turn.events.popleft()
                    turn.total_bytes -= oldest.nbytes

    # -- reconcile ---------------------------------------------------------

    def reconcile(
        self, session_id: str, client_turn_id: str, *, after_seq: int
    ) -> Dict[str, Any]:
        """Return one replay page for the client's cursor.

        Two shapes are possible. ``events`` replays the retained tail. When the
        cursor points at events that have already aged out, the only honest
        answer for a finished turn is a ``snapshot`` of its final state; for a
        still-running turn there is no honest answer at all, so we raise rather
        than fabricate one.
        """

        with self._lock:
            turn = self._require(session_id, client_turn_id)
            if (
                not isinstance(after_seq, int)
                or isinstance(after_seq, bool)
                or after_seq < 0
            ):
                raise TurnRecoveryError("after_seq must be a non-negative int")
            if after_seq > turn.last_seq:
                raise TurnRecoveryError(
                    f"after_seq {after_seq} is ahead of last_seq {turn.last_seq}"
                )

            earliest = turn.earliest_seq
            if after_seq < earliest - 1:
                if not turn.is_terminal:
                    raise TurnRecoveryError(
                        f"turn {turn.turn_id} lost events before seq {earliest} "
                        "and is not terminal; cannot reconcile honestly"
                    )
                return self._snapshot_page(turn, earliest)

            events: List[Dict[str, Any]] = []
            page_bytes = 0
            for event in turn.events:
                if event.seq <= after_seq:
                    continue
                if len(events) >= self._limits.reconcile_max_events:
                    break
                if (
                    events
                    and page_bytes + event.nbytes
                    > self._limits.reconcile_max_page_bytes
                ):
                    # Always emit at least one event when the cursor is behind:
                    # an empty page with has_more would stall the client.
                    break
                events.append(event.to_wire())
                page_bytes += event.nbytes

            next_after_seq = events[-1]["seq"] if events else after_seq
            return {
                "mode": "events",
                "turn_id": turn.turn_id,
                "status": turn.status,
                "earliest_seq": earliest,
                "last_seq": turn.last_seq,
                "next_after_seq": next_after_seq,
                "has_more": next_after_seq < turn.last_seq,
                "events": events,
                "automatic_resubmit": False,
            }

    def _snapshot_page(self, turn: _Turn, earliest: int) -> Dict[str, Any]:
        return {
            "mode": "snapshot",
            "earliest_seq": earliest,
            "last_seq": turn.last_seq,
            "next_after_seq": turn.last_seq,
            "has_more": False,
            "automatic_resubmit": False,
            "snapshot": {
                "turn_id": turn.turn_id,
                "client_turn_id": turn.client_turn_id,
                "status": turn.status,
                "last_seq": turn.last_seq,
                "assistant": {
                    "message_id": turn.assistant_message_id or turn.turn_id,
                    "text": turn.assistant_text,
                    "complete": True,
                },
                "attachment_manifest_digest": turn.attachment_manifest_digest,
                "final_message_ref": turn.final_message_ref,
            },
        }
