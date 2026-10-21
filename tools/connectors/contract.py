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
    # Stamped by settle() on every unresolved target; never a transition target.
    not_connected = "not_connected"


class Actor(str, Enum):
    user = "user"
    backend_watcher = "backend_watcher"
    clock = "clock"


class SettleReason(str, Enum):
    all_resolved = "all_resolved"
    continue_ = "continue"
    deadline = "deadline"
    interrupt = "interrupt"


KINDS: tuple[str, ...] = ("connector", "mcp", "plugin", "skill")

RESOLVED_STATES = frozenset({TargetState.connected, TargetState.skipped})
# A failed catalog row is done as well: the host already ran the install and the row carries the
# reason. Counting it as open held the model's turn until the deadline while nobody clicked. Try
# again still works while another row keeps the operation open, and the failed row keeps its state
# and reason when the operation settles.
_RESOLVED_BY_KIND = {kind: RESOLVED_STATES | {TargetState.failed} for kind in ("plugin", "skill")}


def resolves(kind: str, state: TargetState) -> bool:
    return state in _RESOLVED_BY_KIND.get(kind, RESOLVED_STATES)

_S, _A = TargetState, Actor

# (kind, from) -> {to: the only actor allowed to cause it}. Every `connected` is witnessed by the
# backend: the gateway's account list for a managed target, the install / enable / OAuth worker for
# an MCP one. The card can only skip a target, or ask for a failed one to be run again.
TRANSITIONS: dict[tuple[str, TargetState], dict[TargetState, Actor]] = {
    ("connector", _S.pending): {_S.initiated: _A.backend_watcher, _S.failed: _A.backend_watcher, _S.skipped: _A.user},
    ("connector", _S.initiated): {
        _S.connected: _A.backend_watcher, _S.failed: _A.backend_watcher, _S.expired: _A.clock, _S.skipped: _A.user,
    },
    ("connector", _S.failed): {_S.initiated: _A.user, _S.skipped: _A.user},
    ("connector", _S.expired): {_S.initiated: _A.user, _S.skipped: _A.user},
    ("mcp", _S.pending): {_S.initiated: _A.backend_watcher, _S.failed: _A.backend_watcher, _S.skipped: _A.user},
    ("mcp", _S.initiated): {_S.connected: _A.backend_watcher, _S.failed: _A.backend_watcher, _S.skipped: _A.user},
    ("mcp", _S.failed): {_S.initiated: _A.user, _S.skipped: _A.user},
}
# ``manage_catalog`` rows install like an MCP entry: approve starts the host's install, the install
# worker witnesses ``connected``, Try again re-runs a failed row.
for _kind in ("plugin", "skill"):
    for (_k, _from), _edges in list(TRANSITIONS.items()):
        if _k == "mcp":
            TRANSITIONS[(_kind, _from)] = dict(_edges)


def allowed(kind: str, current: TargetState, to: TargetState) -> Optional[Actor]:
    return TRANSITIONS.get((kind, current), {}).get(to)
