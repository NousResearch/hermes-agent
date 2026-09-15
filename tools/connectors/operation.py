"""One backend-owned connection operation per ``manage_connections`` call. Pure data, no I/O."""

from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional

from tools.connectors.contract import RESOLVED_STATES, Actor, SettleReason, TargetState, allowed

# Not a config key: a user-tunable wait with clamp rails was a foot-gun (PR1 shipped one, unmerged).
OPERATION_DEADLINE_SECONDS = 300.0


class IllegalTransition(ValueError):
    pass


@dataclass
class Target:
    name: str
    kind: str
    action: str
    state: TargetState = TargetState.pending
    detail: str = ""
    connect_url: Optional[str] = None
    # Opaque per-attempt handle when the gateway mints one (absent today; the status route adds it).
    attempt: Optional[str] = None
    # `reconnect force` on a connected account: the list reports the OLD account `active` until the user
    # signs in again, so `connected` is not believed until the row has read as anything else once.
    awaiting_new_attempt: bool = False
    # Renderer-reported fields passed through to the model (``tools`` after MCP OAuth).
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
        if self.attempt:
            out["attempt"] = self.attempt
        out.update(deepcopy(self.extra))
        return out


@dataclass
class ConnectionOperation:
    # The gateway installs its ``connection.update`` emitter here once; pure data otherwise.
    on_change: ClassVar[Optional[Callable[["ConnectionOperation", Optional[Dict[str, Any]]], None]]] = None

    targets: List[Target]
    session_key: str = ""
    # The model's id for the call that opened the operation; the card binds to that tool row only.
    tool_call_id: Optional[str] = None
    op_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    deadline_at: float = 0.0
    settled_at: Optional[float] = None
    settled_by: Optional[SettleReason] = None
    # Set on every transition and on settle; the waiting loop sleeps on it.
    wake: threading.Event = field(default_factory=threading.Event, repr=False)
    _settled_snapshot: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.deadline_at:
            self.deadline_at = self.created_at + OPERATION_DEADLINE_SECONDS

    def target(self, name: str) -> Optional[Target]:
        return next((t for t in self.targets if t.name == name), None)

    def transition(
        self, name: str, to: TargetState, actor: Actor, *, detail: Optional[str] = None,
        connect_url: Optional[str] = None, attempt: Optional[str] = None, **extra: Any,
    ) -> Optional[Dict[str, Any]]:
        """Move one target; the contract decides whether ``actor`` may. Returns the change, or None
        when the target is already in ``to``. Allowed after settlement: the frozen result stays."""
        target = self.target(name)
        if target is None:
            raise IllegalTransition(f"unknown target {name!r}")
        with self._lock:
            if target.state == to:
                return None
            if allowed(target.kind, target.state, to) != actor:
                raise IllegalTransition(f"{target.kind} {name}: {target.state.value} -> {to.value} by {actor.value}")
            change = {"target": name, "from": target.state.value, "to": to.value, "actor": actor.value}
            target.state = to
            if detail is not None:
                target.detail = detail
            change["detail"] = target.detail
            if connect_url is not None:
                target.connect_url = connect_url
            if attempt is not None:
                target.attempt = attempt
            if extra:
                target.extra = dict(extra)
        self.wake.set()
        self._changed(change)
        return change

    def refresh(self, name: str, *, connect_url: Optional[str], detail: str) -> None:
        """Replace a target's link and detail without a state change (a repeated failure)."""
        target = self.target(name)
        if target is None:
            raise IllegalTransition(f"unknown target {name!r}")
        with self._lock:
            target.connect_url = connect_url
            target.detail = detail
            change = {"target": name, "from": target.state.value, "to": target.state.value, "actor": Actor.user.value,
                      "detail": detail}
        self.wake.set()
        self._changed(change)

    def _changed(self, change: Optional[Dict[str, Any]]) -> None:
        hook = type(self).on_change
        if hook is not None:
            hook(self, change)

    @property
    def all_resolved(self) -> bool:
        return bool(self.targets) and all(t.resolved for t in self.targets)

    @property
    def settled(self) -> bool:
        return self.settled_at is not None

    def remaining_seconds(self, now: Optional[float] = None) -> float:
        return max(0.0, self.deadline_at - (time.time() if now is None else now))

    def settle(self, by: SettleReason, now: Optional[float] = None) -> bool:
        """Compare-and-set: the first caller freezes the result."""
        with self._lock:
            if self.settled_at is not None:
                return False
            self.settled_at = time.time() if now is None else now
            self.settled_by = by
            for target in self.targets:
                if not target.resolved:
                    target.state = TargetState.not_connected
            self._settled_snapshot = self._snapshot_locked()
        self.wake.set()
        self._changed(None)
        return True

    def settle_if_all_resolved(self) -> bool:
        return self.all_resolved and self.settle(SettleReason.all_resolved)

    def _snapshot_locked(self, *, with_urls: bool = True) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "deadline_at": self.deadline_at,
            "settled_at": self.settled_at,
            "settled_by": self.settled_by.value if self.settled_by else None,
            "targets": [t.snapshot(with_url=with_urls) for t in self.targets],
        }

    def result(self, *, with_urls: bool = True) -> Dict[str, Any]:
        """The settled result (frozen at settle time), or the live snapshot before settlement."""
        with self._lock:
            if self._settled_snapshot is not None:
                snapshot = deepcopy(self._settled_snapshot)
                if not with_urls:
                    for t in snapshot["targets"]:
                        t.pop("connect_url", None)
                return snapshot
            return self._snapshot_locked(with_urls=with_urls)

    def request_payload(self) -> Dict[str, Any]:
        """The ``connection.request`` payload and the resume snapshot: identity, live target snapshots
        (links included, the panel owns them), server-owned deadline."""
        with self._lock:
            targets = [t.snapshot() for t in self.targets]
        payload: Dict[str, Any] = {
            "op_id": self.op_id,
            "deadline_at": self.deadline_at,
            "timeout_seconds": OPERATION_DEADLINE_SECONDS,
            "targets": targets,
        }
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload
