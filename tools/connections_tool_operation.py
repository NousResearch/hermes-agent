"""The connection operation: one backend-owned record per ``manage_connections`` call.

A call names targets; each target resolves on its own (connected, skipped, failed); the
operation settles exactly once. The first of {every target resolved, Continue, deadline,
interrupt} wins, and later events change the current per-target state only, never the
settled result. The deadline is set here, at creation, and nothing on the client side
(navigation, remount, desktop restart) can move it.

Pure data + settlement rules; no I/O. ``tools/connections_tool.py`` drives it.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# config.yaml ``connections.wait_timeout_seconds``: how long one operation may stay open.
# The floor keeps the card on screen long enough to be clicked; there is no ceiling — the
# configured value bounds the wait directly.
WAIT_TIMEOUT_DEFAULT_SECONDS = 120.0
WAIT_TIMEOUT_FLOOR_SECONDS = 5.0

# Per-target states. ``connected`` and ``skipped`` are RESOLVED (the user decided);
# ``failed`` is recoverable and keeps the operation open for Try again / Continue / deadline;
# ``unavailable`` is resolved too — no surface exists to ask on.
PENDING = "pending"
CONNECTED = "connected"
SKIPPED = "skipped"
FAILED = "failed"
UNAVAILABLE = "unavailable"
# Assigned at settlement to every target still pending/failed: the card's static
# "Not connected · <reason>" row, with ``detail`` carrying how the operation settled.
NOT_CONNECTED = "not_connected"
RESOLVED_STATES = frozenset({CONNECTED, SKIPPED, UNAVAILABLE})

# How an operation settled.
SETTLED_ALL_RESOLVED = "all_resolved"
SETTLED_CONTINUE = "continue"
SETTLED_DEADLINE = "deadline"
SETTLED_INTERRUPT = "interrupt"
SETTLED_UNAVAILABLE = "unavailable"


def resolve_wait_timeout(config: Optional[Dict[str, Any]] = None) -> float:
    """``connections.wait_timeout_seconds`` from config.yaml, floored, default 120.

    Reads only that key: the executor batch guard (``HERMES_CONCURRENT_TOOL_TIMEOUT_S``) and
    the clarify timeout are separate budgets and never proxy for this one.
    """
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
        """Update one target's CURRENT state. Returns False for an unknown target.

        Allowed after settlement on purpose: the live card may keep reporting, but the
        settled result (``result()``) was frozen at settle time and does not change.
        """
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
        """Compare-and-set: the first caller settles and freezes the result; every later
        caller gets False and changes nothing."""
        with self._lock:
            if self.settled_at is not None:
                return False
            self.settled_at = time.time() if now is None else now
            self.settled_by = by
            # Freeze: anything the user never decided is "not connected" with the settle
            # reason, so the static card never shows a live-looking "pending" row.
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
        """What the UI bridge sends: the operation identity, its targets and the server-owned deadline."""
        return {
            "op_id": self.op_id,
            "deadline_at": self.deadline_at,
            "timeout_seconds": float(self.wait_seconds),
            "reason": reason or "",
            "targets": [
                {"name": t.name, "kind": t.kind, "action": t.action} for t in self.targets
            ],
        }
