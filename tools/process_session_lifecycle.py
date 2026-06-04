"""
Process-session lifecycle prototype — Hermes-native, tmux-inspired.

Status: PROTOTYPE. Disabled-default. Purely additive — does not import or
modify the existing `tools/process_registry.py`. Intended for review and
incremental adoption, not for runtime enablement.

Provenance:
  Inspired by the tmux source spike
  (/home/filip/spearhead-execution/20260528-source-spikes/tmux-tmux/source-spike.md).
  Concepts adopted: separate "process exited" vs "stream drained" lifecycle
  states (tmux job.c JOB_RUNNING/JOB_DEAD/JOB_CLOSED), retain-on-failure dead
  pane summaries (tmux server-fn.c remain-on-exit), idempotent subscriber
  semantics (tmux cmd-pipe-pane.c -o "open if absent"), and backpressure
  accounting on async sinks (tmux control mode "too far behind").

  NO tmux source code was copied. The C structures and libevent details are
  not relevant to Python/asyncio/threaded Hermes; only the lifecycle invariant
  ("completion requires both exit_code captured AND stream drained") is
  reused, expressed in Python idioms.

Hard non-goals (out of scope by task definition):
  - No terminal UI / pane / layout clone.
  - No automatic respawn with side effects. Respawn is documented as a future
    spike that must come with idempotency keys and approval gates.
  - No production wiring. This module exposes a state machine, a summary
    artifact, and an idempotent subscriber registry only.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


class LifecycleState(str, Enum):
    """Explicit, monotonic process-session lifecycle states.

    Order is monotonic: a session can only move forward through this sequence.
    Backward transitions are an error (LifecycleTransitionError).

    States:
      RUNNING        — OS process is alive; no exit status captured.
      EXITED         — OS process has terminated; exit_code (and optional
                       signal) captured. Stream buffers may still hold bytes.
      STREAM_DRAINED — stdout/stderr reader has observed EOF (or has been
                       explicitly abandoned via abandon_stream()). No further
                       output will be appended.
      CLOSED         — Session finalized. A DeadSessionSummary has been
                       emitted (or retention policy intentionally skipped it).
                       Subscribers can no longer be registered.

    The split between EXITED and STREAM_DRAINED is the core tmux-derived
    invariant: tmux's job.c only fires the completion callback once BOTH
    the child process has been reaped AND the stdout bufferevent has
    drained. Conflating the two — the failure mode this prototype guards
    against — loses either the exit code (if cleanup happens first) or
    final output (if exit happens first).
    """

    RUNNING = "running"
    EXITED = "exited"
    STREAM_DRAINED = "stream_drained"
    CLOSED = "closed"


_ALLOWED_FORWARD: Dict[LifecycleState, Tuple[LifecycleState, ...]] = {
    LifecycleState.RUNNING: (LifecycleState.EXITED,),
    LifecycleState.EXITED: (LifecycleState.STREAM_DRAINED,),
    LifecycleState.STREAM_DRAINED: (LifecycleState.CLOSED,),
    LifecycleState.CLOSED: (),
}


class LifecycleTransitionError(RuntimeError):
    """Raised on an illegal lifecycle state transition.

    Examples that raise:
      - mark_stream_drained() before mark_exited()
      - mark_exited() after CLOSED
      - mark_exited() called twice with conflicting exit_code

    Idempotent calls (same target state, same payload) are NOT errors; they
    return silently. This mirrors tmux pipe-pane -o ("open if absent") and
    keeps reader-thread / reconciler races safe.
    """


# ---------------------------------------------------------------------------
# Dead-session summary artifact (retain-on-failure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeadSessionSummary:
    """Compact post-mortem artifact for a closed process session.

    Inspired by tmux's `remain-on-exit-format`: enough context to render a
    "dead pane" line in a dashboard / Kanban card / CLI listing without
    requiring log archaeology. NOT a full log dump — `output_tail` is a
    bounded slice, and `log_pointer` is a hint to the full record (e.g. a
    file path or storage key) for callers that want more.

    Fields are intentionally serializable (no thread locks, no Popen handles)
    so this can be persisted into checkpoint files, Kanban metadata, or
    structured logs.
    """

    session_id: str
    command: str
    exit_code: Optional[int]
    exit_signal: Optional[int]
    exit_reason: str          # "exited", "killed", "abandoned", "lost"
    started_at: float
    exited_at: Optional[float]
    closed_at: float
    output_tail: str          # last N bytes of captured output, may be ""
    output_tail_bytes: int
    log_pointer: Optional[str] = None
    subscriber_backpressure: Optional[Dict[str, int]] = None
    notes: Tuple[str, ...] = ()

    @property
    def is_failure(self) -> bool:
        """A session is a failure if it exited non-zero, was killed, or its
        runtime handle was lost. Used by retain-on-failure policy."""
        if self.exit_reason in ("killed", "lost", "abandoned"):
            return True
        if self.exit_code is None:
            # Unknown exit code — treat as failure for retention purposes.
            return True
        return self.exit_code != 0


class RetentionPolicy(str, Enum):
    """When to retain the DeadSessionSummary after CLOSED.

    RETAIN_ON_FAILURE is the recommended default (tmux's remain-on-exit
    "failed" mode). Keeping every successful summary forever is wasteful;
    discarding everything makes failure triage harder.
    """

    NEVER = "never"
    RETAIN_ON_FAILURE = "retain_on_failure"
    ALWAYS = "always"


# ---------------------------------------------------------------------------
# Idempotent subscriber registry with backpressure accounting
# ---------------------------------------------------------------------------


@dataclass
class SubscriberBackpressure:
    """Per-subscriber backpressure accounting.

    Tracks how many events were discarded due to a slow consumer. Once
    `discarded` exceeds `too_far_behind_threshold`, the subscriber is
    marked "too far behind" — callers may then choose to disconnect it
    (tmux control mode's "too far behind" exit reason).
    """

    discarded: int = 0
    bytes_discarded: int = 0
    last_discard_at: float = 0.0
    too_far_behind_threshold: int = 1000
    too_far_behind: bool = False

    def record_discard(self, byte_count: int = 0) -> None:
        self.discarded += 1
        self.bytes_discarded += byte_count
        self.last_discard_at = time.time()
        if self.discarded >= self.too_far_behind_threshold:
            self.too_far_behind = True

    def snapshot(self) -> Dict[str, int]:
        return {
            "discarded": self.discarded,
            "bytes_discarded": self.bytes_discarded,
            "too_far_behind": int(self.too_far_behind),
        }


SubscriberCallback = Callable[[str, str], None]
"""Subscriber callback: (event_kind, payload) -> None.

event_kind examples: "stdout_chunk", "lifecycle_transition", "summary".
The callback must not block; backpressure accounting is the caller's
responsibility to feed via record_discard() when a sink is slow.
"""


@dataclass
class _SubscriberHandle:
    subscriber_id: str
    callback: SubscriberCallback
    registered_at: float
    backpressure: SubscriberBackpressure = field(default_factory=SubscriberBackpressure)
    read_only: bool = False
    # Optional notes; useful when registering the same id twice and wanting
    # to surface why the original wins.
    label: str = ""


class IdempotentSubscriberRegistry:
    """Idempotent log/event subscriber registry.

    `register(subscriber_id, ...)` returns an existing handle if the id is
    already registered (tmux pipe-pane -o semantics). This prevents the
    "two competing watchers on the same session" bug that the existing
    process_registry's watch_patterns code already guards against via
    rate-limit strikes; the idempotent registry attacks the same problem
    earlier — at subscription time — so duplicate sinks never form.

    Thread-safe. Designed to be embedded in ProcessSessionLifecycle but
    independently testable.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handles: Dict[str, _SubscriberHandle] = {}
        self._closed = False

    def register(
        self,
        subscriber_id: str,
        callback: SubscriberCallback,
        *,
        read_only: bool = False,
        label: str = "",
        replace: bool = False,
    ) -> _SubscriberHandle:
        """Register a subscriber id.

        If `subscriber_id` is already registered:
          - replace=False (default): return the existing handle. Idempotent.
          - replace=True: drop the old handle and install the new one.

        If the registry is closed (lifecycle CLOSED), raises RuntimeError —
        callers should not attach new sinks to a finalized session.
        """
        if not subscriber_id:
            raise ValueError("subscriber_id must be a non-empty string")
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "Subscriber registry is closed — session has reached CLOSED state"
                )
            existing = self._handles.get(subscriber_id)
            if existing is not None and not replace:
                return existing
            handle = _SubscriberHandle(
                subscriber_id=subscriber_id,
                callback=callback,
                registered_at=time.time(),
                read_only=read_only,
                label=label,
            )
            self._handles[subscriber_id] = handle
            return handle

    def unregister(self, subscriber_id: str) -> bool:
        """Remove a subscriber. Returns True if it existed, False otherwise.

        Idempotent: unregistering an absent subscriber is not an error.
        """
        with self._lock:
            return self._handles.pop(subscriber_id, None) is not None

    def get(self, subscriber_id: str) -> Optional[_SubscriberHandle]:
        with self._lock:
            return self._handles.get(subscriber_id)

    def list_ids(self) -> List[str]:
        with self._lock:
            return list(self._handles.keys())

    def broadcast(self, event_kind: str, payload: str) -> None:
        """Best-effort fan-out. Slow / raising sinks do not block siblings;
        their backpressure counter is incremented instead.
        """
        with self._lock:
            handles = list(self._handles.values())
        for handle in handles:
            try:
                handle.callback(event_kind, payload)
            except Exception:
                # Sink failure counts as backpressure — keeps a flaky
                # subscriber from being silently considered healthy.
                handle.backpressure.record_discard(byte_count=len(payload))

    def backpressure_snapshot(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {
                sid: h.backpressure.snapshot()
                for sid, h in self._handles.items()
            }

    def _close(self) -> None:
        """Internal: lifecycle hook — refuse further registrations."""
        with self._lock:
            self._closed = True


# ---------------------------------------------------------------------------
# Process-session lifecycle FSM
# ---------------------------------------------------------------------------


@dataclass
class _LifecycleSnapshot:
    """Read-only snapshot returned by ProcessSessionLifecycle.snapshot()."""

    session_id: str
    command: str
    state: LifecycleState
    exit_code: Optional[int]
    exit_signal: Optional[int]
    exit_reason: str
    started_at: float
    exited_at: Optional[float]
    stream_drained_at: Optional[float]
    closed_at: Optional[float]
    subscriber_ids: Tuple[str, ...]


class ProcessSessionLifecycle:
    """Hermes-native, tmux-inspired process-session lifecycle.

    Owns:
      - the state machine (RUNNING → EXITED → STREAM_DRAINED → CLOSED)
      - exit status capture at EXITED (preserved regardless of stream cleanup)
      - the idempotent subscriber registry
      - dead-session summary generation under a configurable retention policy

    Does NOT own:
      - subprocess.Popen / pty handles
      - the actual output buffer (callers feed `output_tail` at close time)
      - notification delivery (subscribers + caller's queue)
      - respawn (out of scope; see "Respawn" note at bottom of file)

    This separation lets the FSM be tested without spawning real processes,
    and lets the existing process_registry adopt it incrementally without a
    big rewrite.
    """

    def __init__(
        self,
        session_id: str,
        command: str,
        *,
        started_at: Optional[float] = None,
        retention: RetentionPolicy = RetentionPolicy.RETAIN_ON_FAILURE,
        output_tail_limit_bytes: int = 4096,
    ) -> None:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        self.session_id = session_id
        self.command = command
        self._state = LifecycleState.RUNNING
        self._state_lock = threading.RLock()
        self.started_at = started_at if started_at is not None else time.time()
        self.exited_at: Optional[float] = None
        self.stream_drained_at: Optional[float] = None
        self.closed_at: Optional[float] = None
        self.exit_code: Optional[int] = None
        self.exit_signal: Optional[int] = None
        self.exit_reason: str = "running"
        self.subscribers = IdempotentSubscriberRegistry()
        self._retention = retention
        self._output_tail_limit = output_tail_limit_bytes
        self._summary: Optional[DeadSessionSummary] = None
        self._notes: List[str] = []

    # ----- State queries -----

    @property
    def state(self) -> LifecycleState:
        with self._state_lock:
            return self._state

    def snapshot(self) -> _LifecycleSnapshot:
        with self._state_lock:
            return _LifecycleSnapshot(
                session_id=self.session_id,
                command=self.command,
                state=self._state,
                exit_code=self.exit_code,
                exit_signal=self.exit_signal,
                exit_reason=self.exit_reason,
                started_at=self.started_at,
                exited_at=self.exited_at,
                stream_drained_at=self.stream_drained_at,
                closed_at=self.closed_at,
                subscriber_ids=tuple(self.subscribers.list_ids()),
            )

    def summary(self) -> Optional[DeadSessionSummary]:
        with self._state_lock:
            return self._summary

    def add_note(self, note: str) -> None:
        """Attach a free-form note that will be folded into the summary.

        Useful for callers to record context like "killed via session reset"
        or "watcher disabled after 3 strikes" without overloading exit_reason.
        """
        with self._state_lock:
            if self._state == LifecycleState.CLOSED:
                # Notes after CLOSED would mutate the summary; refuse.
                raise LifecycleTransitionError(
                    "Cannot add note after CLOSED — summary is frozen"
                )
            self._notes.append(note)

    # ----- Transitions -----

    def mark_exited(
        self,
        exit_code: Optional[int],
        *,
        exit_signal: Optional[int] = None,
        exit_reason: str = "exited",
        at: Optional[float] = None,
    ) -> bool:
        """Record that the OS process exited.

        Idempotent: calling twice with the same exit_code/signal is a no-op.
        Conflicting payloads (different exit_code on the second call) raise
        LifecycleTransitionError — that's a bug, not a race.

        Returns True if this call performed the transition, False if it was
        an idempotent re-entry.

        Valid sources of `exit_reason`:
          "exited"   — normal child reap
          "killed"   — terminated by signal / kill_process
          "lost"     — runtime handle gone (detached + pid not alive)
          "abandoned" — caller explicitly gave up on this session
        """
        with self._state_lock:
            if self._state == LifecycleState.RUNNING:
                self._state = LifecycleState.EXITED
                self.exit_code = exit_code
                self.exit_signal = exit_signal
                self.exit_reason = exit_reason
                self.exited_at = at if at is not None else time.time()
                self._broadcast_transition()
                return True
            # Idempotent re-entry — verify payload consistency.
            if (
                self.exit_code != exit_code
                or self.exit_signal != exit_signal
                or self.exit_reason != exit_reason
            ):
                raise LifecycleTransitionError(
                    f"mark_exited called twice with conflicting payload: "
                    f"existing=(code={self.exit_code}, signal={self.exit_signal}, "
                    f"reason={self.exit_reason!r}), new=(code={exit_code}, "
                    f"signal={exit_signal}, reason={exit_reason!r})"
                )
            if self._state in (
                LifecycleState.EXITED,
                LifecycleState.STREAM_DRAINED,
                LifecycleState.CLOSED,
            ):
                # Already past or at EXITED with consistent payload — no-op.
                return False
            # Should be unreachable.
            raise LifecycleTransitionError(
                f"Invalid state for mark_exited: {self._state}"
            )

    def mark_stream_drained(self, *, at: Optional[float] = None) -> bool:
        """Record that the output stream has been fully read (EOF observed).

        Must be called only after mark_exited(). Calling before is the
        canonical "premature cleanup" bug this lifecycle is designed to
        prevent — it raises LifecycleTransitionError.

        Idempotent. Returns True for the actual transition, False for
        idempotent re-entry.
        """
        with self._state_lock:
            if self._state == LifecycleState.RUNNING:
                raise LifecycleTransitionError(
                    "Cannot drain stream before recording exit. "
                    "Premature cleanup would lose exit status."
                )
            if self._state == LifecycleState.EXITED:
                self._state = LifecycleState.STREAM_DRAINED
                self.stream_drained_at = at if at is not None else time.time()
                self._broadcast_transition()
                return True
            if self._state in (
                LifecycleState.STREAM_DRAINED,
                LifecycleState.CLOSED,
            ):
                return False
            raise LifecycleTransitionError(
                f"Invalid state for mark_stream_drained: {self._state}"
            )

    def abandon_stream(self, *, reason: str = "abandoned", at: Optional[float] = None) -> bool:
        """Force STREAM_DRAINED without observing EOF.

        Escape hatch for the "orphaned descendant holds stdout open" case
        (Hermes issue #17327). The lifecycle records the abandonment as a
        note so the dead-session summary can explain why no final bytes
        appeared.

        Must be called after mark_exited(). Same idempotency rules as
        mark_stream_drained().
        """
        with self._state_lock:
            if self._state == LifecycleState.RUNNING:
                raise LifecycleTransitionError(
                    "Cannot abandon stream before recording exit"
                )
            if self._state == LifecycleState.EXITED:
                self._notes.append(f"stream abandoned: {reason}")
                self._state = LifecycleState.STREAM_DRAINED
                self.stream_drained_at = at if at is not None else time.time()
                self._broadcast_transition()
                return True
            if self._state in (
                LifecycleState.STREAM_DRAINED,
                LifecycleState.CLOSED,
            ):
                return False
            raise LifecycleTransitionError(
                f"Invalid state for abandon_stream: {self._state}"
            )

    def close(
        self,
        *,
        output_tail: str = "",
        log_pointer: Optional[str] = None,
        at: Optional[float] = None,
    ) -> Optional[DeadSessionSummary]:
        """Finalize the session.

        Must be called only after mark_stream_drained() (or abandon_stream()).
        Idempotent. Returns the DeadSessionSummary if one was retained per
        policy, otherwise None.

        After CLOSED, the subscriber registry refuses new registrations.
        """
        with self._state_lock:
            if self._state in (LifecycleState.RUNNING, LifecycleState.EXITED):
                raise LifecycleTransitionError(
                    f"Cannot close from state {self._state.value}: stream must "
                    "be drained first (call mark_stream_drained or abandon_stream)"
                )
            if self._state == LifecycleState.CLOSED:
                return self._summary

            self._state = LifecycleState.CLOSED
            self.closed_at = at if at is not None else time.time()

            # Bound output tail per configured policy.
            tail = output_tail or ""
            if len(tail.encode("utf-8", errors="replace")) > self._output_tail_limit:
                # Trim from the front so we keep the most recent bytes.
                encoded = tail.encode("utf-8", errors="replace")
                tail = encoded[-self._output_tail_limit:].decode("utf-8", errors="replace")

            candidate = DeadSessionSummary(
                session_id=self.session_id,
                command=self.command,
                exit_code=self.exit_code,
                exit_signal=self.exit_signal,
                exit_reason=self.exit_reason,
                started_at=self.started_at,
                exited_at=self.exited_at,
                closed_at=self.closed_at,
                output_tail=tail,
                output_tail_bytes=len(tail.encode("utf-8", errors="replace")),
                log_pointer=log_pointer,
                subscriber_backpressure=self.subscribers.backpressure_snapshot(),
                notes=tuple(self._notes),
            )

            if self._retention == RetentionPolicy.ALWAYS:
                self._summary = candidate
            elif self._retention == RetentionPolicy.RETAIN_ON_FAILURE:
                self._summary = candidate if candidate.is_failure else None
            else:
                self._summary = None

            self.subscribers._close()
            self._broadcast_transition()
            return self._summary

    # ----- Internals -----

    def _broadcast_transition(self) -> None:
        """Notify subscribers of the new lifecycle state.

        Called with self._state_lock held. Subscribers MUST NOT call back
        into this lifecycle synchronously, or they will deadlock — that's
        an intentional ergonomic foot-gun guard. The fan-out itself takes
        the registry's own lock independently.
        """
        # Build the payload outside the broadcast so subscribers see a
        # consistent snapshot even if they take their time.
        payload = (
            f"{self.session_id}:{self._state.value}:"
            f"exit_code={self.exit_code}:reason={self.exit_reason}"
        )
        self.subscribers.broadcast("lifecycle_transition", payload)


# ---------------------------------------------------------------------------
# Respawn note (intentionally NOT implemented here)
# ---------------------------------------------------------------------------
#
# The tmux spike calls out spawn.c's respawn semantics as a useful idea:
# reuse a logical pane/session identity, refuse to respawn if still active
# unless a kill flag is set, and persist enough launch metadata (cwd, argv,
# env deltas, shell) to support it.
#
# For Hermes agents, blind respawn is DANGEROUS — repeating an LLM tool
# call can duplicate side effects (file writes, API calls, notifications).
# Respawn therefore requires:
#   - an idempotency key supplied by the caller, with deduplication enforced
#     by the registry, not by best-effort hashing;
#   - an explicit approval gate ("NEEDS FILIP APPROVAL" / Kanban review-
#     required) before re-running anything that touches production;
#   - a checkpoint of "what was already done" so respawn can resume rather
#     than restart.
#
# This module deliberately offers no respawn() method to make the absence
# obvious. A future, separately-approved card should implement it.
# ---------------------------------------------------------------------------
