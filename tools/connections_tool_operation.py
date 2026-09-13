"""One backend-owned connection operation per ``manage_connections`` call: targets, a
server-set deadline, exactly-once settlement. Pure data, no I/O."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# config.yaml ``connections.wait_timeout_seconds``; floor only, no ceiling.
WAIT_TIMEOUT_DEFAULT_SECONDS = 120.0
WAIT_TIMEOUT_FLOOR_SECONDS = 5.0

# Target states. connected/skipped/unavailable are resolved; failed keeps the operation open.
PENDING = "pending"
CONNECTED = "connected"
SKIPPED = "skipped"
FAILED = "failed"
UNAVAILABLE = "unavailable"
# Stamped on unresolved targets at settlement; ``detail`` carries the settle reason.
NOT_CONNECTED = "not_connected"
RESOLVED_STATES = frozenset({CONNECTED, SKIPPED, UNAVAILABLE})

# How an operation settled.
SETTLED_ALL_RESOLVED = "all_resolved"
SETTLED_CONTINUE = "continue"
SETTLED_DEADLINE = "deadline"
SETTLED_INTERRUPT = "interrupt"
SETTLED_UNAVAILABLE = "unavailable"


def resolve_wait_timeout(config: Optional[Dict[str, Any]] = None) -> float:
    """``connections.wait_timeout_seconds`` from config.yaml, floored, default 120. Reads only
    that key; the executor and clarify budgets never proxy for it."""
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly() or {}
        except Exception:
            config = {}
    section = config.get("connections") if isinstance(config, dict) else None
    raw = section.get("wait_timeout_seconds") if isinstance(section, dict) else None
    try:
        value = WAIT_TIMEOUT_DEFAULT_SECONDS if raw is None or isinstance(raw, bool) else float(raw)
    except (TypeError, ValueError):
        value = WAIT_TIMEOUT_DEFAULT_SECONDS
    if value != value:  # NaN
        value = WAIT_TIMEOUT_DEFAULT_SECONDS
    return max(WAIT_TIMEOUT_FLOOR_SECONDS, value)


@dataclass
class Target:
    name: str
    kind: str  # "mcp" | "connector"
    action: str
    state: str = PENDING
    detail: str = ""
    # Renderer-reported outcome fields worth relaying (e.g. ``tools`` after OAuth).
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.state in RESOLVED_STATES

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "kind": self.kind, "action": self.action, "state": self.state}
        if self.detail:
            out["detail"] = self.detail
        out.update(self.extra)
        return out


@dataclass
class ConnectionOperation:
    targets: List[Target]
    session_key: str = ""
    wait_seconds: float = WAIT_TIMEOUT_DEFAULT_SECONDS
    op_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    deadline_at: float = 0.0
    settled_at: Optional[float] = None
    settled_by: Optional[str] = None
    _settled_snapshot: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.deadline_at:
            self.deadline_at = self.created_at + float(self.wait_seconds)

    # -- targets -------------------------------------------------------------

    def target(self, name: str) -> Optional[Target]:
        return next((t for t in self.targets if t.name == name), None)

    def record_target(self, name: str, state: str, detail: str = "", **extra: Any) -> bool:
        """Update a target's live state; False for an unknown name. Allowed after settlement:
        the frozen ``result()`` does not change."""
        target = self.target(name)
        if target is None:
            return False
        with self._lock:
            target.state = state
            target.detail = detail or ""
            target.extra = dict(extra)
        return True

    @property
    def all_resolved(self) -> bool:
        return bool(self.targets) and all(t.resolved for t in self.targets)

    @property
    def settled(self) -> bool:
        return self.settled_at is not None

    def remaining_seconds(self, now: Optional[float] = None) -> float:
        return max(0.0, self.deadline_at - (time.time() if now is None else now))

    # -- settlement ----------------------------------------------------------

    def settle(self, by: str, now: Optional[float] = None) -> bool:
        """Compare-and-set: first caller freezes the result, later callers get False."""
        with self._lock:
            if self.settled_at is not None:
                return False
            self.settled_at = time.time() if now is None else now
            self.settled_by = by
            # Unresolved targets freeze as not_connected so the static card has no pending row.
            for target in self.targets:
                if not target.resolved:
                    reason = target.detail or by
                    target.state = NOT_CONNECTED
                    target.detail = reason
            self._settled_snapshot = self._snapshot_locked()
            return True

    def settle_if_all_resolved(self) -> bool:
        return self.all_resolved and self.settle(SETTLED_ALL_RESOLVED)

    def _snapshot_locked(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "deadline_at": self.deadline_at,
            "settled_at": self.settled_at,
            "settled_by": self.settled_by,
            "targets": [t.snapshot() for t in self.targets],
        }

    def result(self) -> Dict[str, Any]:
        """The settled result (frozen at settle time), or the live snapshot before settlement."""
        with self._lock:
            if self._settled_snapshot is not None:
                return dict(self._settled_snapshot, targets=[dict(t) for t in self._settled_snapshot["targets"]])
            return self._snapshot_locked()

    def request_payload(self, reason: str = "") -> Dict[str, Any]:
        """The ``connection.request`` payload: identity, targets, server-owned deadline."""
        return {
            "op_id": self.op_id,
            "deadline_at": self.deadline_at,
            "timeout_seconds": float(self.wait_seconds),
            "reason": reason or "",
            "targets": [
                {"name": t.name, "kind": t.kind, "action": t.action} for t in self.targets
            ],
        }
