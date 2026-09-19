"""Connection contract: the states a target moves through, who may cause each move, and the
reasons an operation settles. ``operation.py`` enforces it; the desktop reads a generated copy.

A `str` enum so payloads serialise to the bare value and the TypeScript side sees literal unions."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple


class TargetState(str, Enum):
    pending = "pending"
    initiated = "initiated"
    connected = "connected"
    skipped = "skipped"
    failed = "failed"
    expired = "expired"
    unavailable = "unavailable"
    # Stamped by settle() on every unresolved target; never a transition target.
    not_connected = "not_connected"


class Actor(str, Enum):
    user = "user"
    renderer_flow = "renderer_flow"
    backend_watcher = "backend_watcher"
    clock = "clock"


class SettleReason(str, Enum):
    all_resolved = "all_resolved"
    continue_ = "continue"
    deadline = "deadline"
    interrupt = "interrupt"
    unavailable = "unavailable"


KINDS: Tuple[str, ...] = ("connector", "mcp")

RESOLVED_STATES = frozenset({TargetState.connected, TargetState.skipped, TargetState.unavailable})

_S, _A = TargetState, Actor

# (kind, from) -> {to: the only actor allowed to cause it}. A managed `connected` is witnessed by the
# gateway alone; an MCP `connected` (auth completed) is reported by the renderer that ran the flow.
TRANSITIONS: Dict[Tuple[str, TargetState], Dict[TargetState, Actor]] = {
    ("connector", _S.pending): {_S.initiated: _A.backend_watcher, _S.failed: _A.backend_watcher, _S.skipped: _A.user},
    ("connector", _S.initiated): {
        _S.connected: _A.backend_watcher, _S.failed: _A.backend_watcher, _S.expired: _A.clock, _S.skipped: _A.user,
    },
    ("connector", _S.failed): {_S.initiated: _A.user, _S.skipped: _A.user},
    ("connector", _S.expired): {_S.initiated: _A.user, _S.skipped: _A.user},
    ("mcp", _S.pending): {_S.initiated: _A.renderer_flow, _S.failed: _A.renderer_flow, _S.skipped: _A.user},
    ("mcp", _S.initiated): {_S.connected: _A.renderer_flow, _S.failed: _A.renderer_flow, _S.skipped: _A.user},
    ("mcp", _S.failed): {_S.initiated: _A.user, _S.skipped: _A.user},
}


def allowed(kind: str, current: TargetState, to: TargetState) -> Optional[Actor]:
    return TRANSITIONS.get((kind, current), {}).get(to)
