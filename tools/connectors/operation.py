"""One backend-owned connection operation per ``manage_connections`` call. Pure data, no I/O."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional

from tools.connectors.contract import RESOLVED_STATES, Actor, LegState, SettleReason, allowed
from tools.operations import Owner

# Not a config key: a user-tunable wait with clamp rails was a foot-gun (PR1 shipped one, unmerged).
OPERATION_DEADLINE_SECONDS = 300.0


class IllegalTransition(ValueError):
    pass


@dataclass
class Leg:
    name: str
    kind: str
    action: str
    state: LegState = LegState.pending
    detail: str = ""
    connect_url: Optional[str] = None
    # The vendor account a managed mint created or observed. Not the desktop transport's connection id.
    connection_id: Optional[str] = None
    # Opaque per-attempt handle when the gateway mints one (absent today; the status route adds it).
    attempt: Optional[str] = None
    # Earliest the watcher may read this target's account again; set from a 429's Retry-After so a
    # rate-limited route is not hammered once per second.
    next_read_at: float = 0.0
    # The credentials an MCP install still needs ({name, prompt, required}); the card draws a
    # field per entry and holds its verb until every required one has text.
    required_env: List[Dict[str, Any]] = field(default_factory=list)
    # Fields a transition passes through to the model (``tools`` on a connected MCP leg).
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.state in RESOLVED_STATES

    def snapshot(self, *, with_url: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "kind": self.kind, "action": self.action, "state": self.state.value}
        if self.detail:
            out["detail"] = self.detail
        if with_url and self.connect_url:
            out["connect_url"] = self.connect_url
        if self.connection_id:
            out["connection_id"] = self.connection_id
        if self.attempt:
            out["attempt"] = self.attempt
        if self.required_env:
            out["required_env"] = self.required_env
        out.update(self.extra)
        return out


@dataclass
class ConnectionOperation:
    # The gateway installs its ``connection.update`` emitter here once; pure data otherwise.
    on_change: ClassVar[
        Optional[Callable[["ConnectionOperation", Optional[Dict[str, Any]], Dict[str, Any]], None]]
    ] = None

    legs: List[Leg]
    session_key: str = ""
    # Stamped by ``operations.open`` from the opening thread's profile unless the caller names one.
    owner: Optional[Owner] = None
    # The model's id for the call that opened the operation; the card binds to that tool row only.
    tool_call_id: Optional[str] = None
    op_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    deadline_at: float = 0.0
    settled_at: Optional[float] = None
    _settled_by: Optional[SettleReason] = field(default=None, repr=False)
    # Monotonic write counter. Every frame carries the seq of the snapshot it was built from, so a
    # renderer that keeps the highest seq per op can drop a frame that arrives after a newer one.
    seq: int = 0
    # Set on every transition and on settle; the waiting loop sleeps on it.
    wake: threading.Event = field(default_factory=threading.Event, repr=False)
    _settled_snapshot: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.deadline_at:
            self.deadline_at = self.created_at + OPERATION_DEADLINE_SECONDS

    @property
    def key(self) -> str:
        return self.session_key

    def leg(self, name: str) -> Optional[Leg]:
        return next((leg for leg in self.legs if leg.name == name), None)

    def transition(
        self, name: str, to: LegState, actor: Actor, *, detail: Optional[str] = None,
        connect_url: Optional[str] = None, connection_id: Optional[str] = None, attempt: Optional[str] = None,
        **extra: Any,
    ) -> Optional[Dict[str, Any]]:
        """Move one leg when the contract permits ``actor`` to do so."""
        leg = self.leg(name)
        if leg is None:
            raise IllegalTransition(f"unknown leg {name!r}")
        with self._lock:
            if leg.state == to:
                return None
            if allowed(leg.kind, leg.state, to) != actor:
                raise IllegalTransition(f"{leg.kind} {name}: {leg.state.value} -> {to.value} by {actor.value}")
            change = {"leg": name, "from": leg.state.value, "to": to.value, "actor": actor.value}
            leg.state = to
            if detail is not None:
                leg.detail = detail
            change["detail"] = leg.detail
            if connect_url is not None:
                leg.connect_url = connect_url
            if connection_id is not None:
                leg.connection_id = connection_id
            if attempt is not None:
                leg.attempt = attempt
            if extra:
                leg.extra = dict(extra)
            snapshot = self._bump_locked()
        self.wake.set()
        self._changed(change, snapshot)
        return change

    def refresh(self, name: str, *, connect_url: Optional[str], detail: str, actor: Actor = Actor.user) -> None:
        """Replace a leg's link and detail without a state change.

        ``actor`` says who produced the new text: a second failure of a backend attempt is the
        backend's report, not the user's move, and the frame must not claim otherwise."""
        leg = self.leg(name)
        if leg is None:
            raise IllegalTransition(f"unknown leg {name!r}")
        with self._lock:
            leg.connect_url = connect_url
            leg.detail = detail
            change = {"leg": name, "from": leg.state.value, "to": leg.state.value, "actor": actor.value,
                      "detail": detail}
            snapshot = self._bump_locked()
        self.wake.set()
        self._changed(change, snapshot)

    def _bump_locked(self) -> Dict[str, Any]:
        """Advance the write counter and take the snapshot that frame carries. Both happen under
        ``_lock`` so a second writer cannot backdate this frame with its own state.

        After settlement the frozen snapshot is the frame, seq included: a later write changes
        nothing the renderer can see, so it must not leave the operation naming a seq no frame
        carried (the resume snapshot would wait for a frame that never comes)."""
        if self._settled_snapshot is None:
            self.seq += 1
        return self._result_locked()

    def _changed(self, change: Optional[Dict[str, Any]], snapshot: Dict[str, Any]) -> None:
        hook = type(self).on_change
        if hook is not None:
            hook(self, change, snapshot)

    @property
    def all_resolved(self) -> bool:
        return bool(self.legs) and all(leg.resolved for leg in self.legs)

    @property
    def settled(self) -> bool:
        return self.settled_at is not None

    @property
    def settled_by(self) -> Optional[str]:
        return self._settled_by.value if self._settled_by else None

    @property
    def settle_reason(self) -> Optional[SettleReason]:
        return self._settled_by

    def remaining_seconds(self, now: Optional[float] = None) -> float:
        return max(0.0, self.deadline_at - (time.time() if now is None else now))

    def settle(self, by: SettleReason | str, now: Optional[float] = None) -> bool:
        """Compare-and-set: the first caller freezes the result."""
        with self._lock:
            if self.settled_at is not None:
                return False
            self.settled_at = time.time() if now is None else now
            self._settled_by = SettleReason(by)
            for leg in self.legs:
                if not leg.resolved:
                    leg.state = LegState.not_connected
            self._settled_snapshot = self._bump_locked()
        self.wake.set()
        self._changed(None, self._settled_snapshot)
        return True

    def settle_if_all_resolved(self) -> bool:
        return self.all_resolved and self.settle(SettleReason.all_resolved)

    def _snapshot_locked(self, *, with_urls: bool = True) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "seq": self.seq,
            "deadline_at": self.deadline_at,
            "settled_at": self.settled_at,
            "settled_by": self.settled_by,
            "legs": [leg.snapshot(with_url=with_urls) for leg in self.legs],
        }

    def _result_locked(self, *, with_urls: bool = True) -> Dict[str, Any]:
        if self._settled_snapshot is not None:
            legs = [dict(leg) for leg in self._settled_snapshot["legs"]]
            if not with_urls:
                for leg in legs:
                    leg.pop("connect_url", None)
            return dict(self._settled_snapshot, legs=legs)
        return self._snapshot_locked(with_urls=with_urls)

    def result(self, *, with_urls: bool = True) -> Dict[str, Any]:
        """The settled result (frozen at settle time), or the live snapshot before settlement."""
        with self._lock:
            return self._result_locked(with_urls=with_urls)

    def request_payload(self) -> Dict[str, Any]:
        """Return the ``connection.request`` payload and resume snapshot."""
        with self._lock:
            legs = [leg.snapshot() for leg in self.legs]
            seq = self.seq
        payload: Dict[str, Any] = {
            "op_id": self.op_id,
            "seq": seq,
            "deadline_at": self.deadline_at,
            "timeout_seconds": OPERATION_DEADLINE_SECONDS,
            "legs": legs,
        }
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload

    def snapshot(self) -> Dict[str, Any]:
        return self.request_payload()
