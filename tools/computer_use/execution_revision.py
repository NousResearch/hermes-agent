"""Internal execution-revision contract for computer_use (Phase 0B of #112734).

``admit at epoch N → do work → validate epoch N → publish / commit``, extracted from the Bot Screen
lease-epoch mechanics (#108914) into one internal record. The lease epoch comes from the vendored
#108914 abstraction (``tools/bot_desktop/lease.py``) — populated today, so ``validate()`` already
grows teeth on CONTROL; the driver snapshot id stays None and validates open. No scheduler, no
speculative mutations, no model-facing change (#112734 §K).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Callable, Dict, FrozenSet, Mapping, Optional

# Dependency names a revision can vouch for. Each maps to independent typed fields: a display rebind
# must not read as a control-epoch change (#112734 §A "Do not conflate revisions").
DISPLAY = "display"
BACKEND = "backend"
CONTROL = "control"
TARGET = "target"
SNAPSHOT = "snapshot"

_ALL_DEPS: FrozenSet[str] = frozenset({DISPLAY, BACKEND, CONTROL, TARGET, SNAPSHOT})

# Operation → the facts it relies on (#112734 §A). Capture fences on display/backend/target; input
# additionally needs the control epoch and a fresh driver snapshot. The epoch is provable now (the
# vendored #108914 lease bumps it on every control transition); the snapshot id stays None until
# cua-driver exposes one, and validates open meanwhile.
CAPTURE_DEPS: FrozenSet[str] = frozenset({DISPLAY, BACKEND, TARGET})
INPUT_DEPS: FrozenSet[str] = frozenset({CONTROL, DISPLAY, BACKEND, TARGET, SNAPSHOT})

_INVALIDATION_REASON: Dict[str, str] = {
    CONTROL: "control_epoch_changed",
    DISPLAY: "display_changed",
    BACKEND: "backend_replaced",
    TARGET: "target_changed",
    SNAPSHOT: "snapshot_stale",
}


@dataclass(frozen=True)
class ExecutionRevision:
    """Immutable facts one operation was admitted under. Fields the runtime cannot prove today stay None;
    ``validate()`` treats an unprovable fact as un-invalidatable, never as evidence of staleness."""
    profile_key: str
    display_identity: Optional[str] = None  # e.g. DISPLAY=":21"; None where the OS exposes none
    backend_generation: Optional[int] = None  # per-session install counter owned by tool.py
    control_epoch: Optional[int] = None  # lease epoch from the vendored #108914 abstraction
    app: Optional[str] = None  # sticky target app at admission
    pid: Optional[int] = None
    window_id: Optional[str] = None
    snapshot_id: Optional[str] = None  # driver snapshot token; None: cua-driver exposes none on main


@dataclass(frozen=True)
class RevisionVerdict:
    ok: bool
    reason: Optional[str] = None  # invalidation reason when not ok

    def describe(self) -> str:
        return "revision valid" if self.ok else f"revision invalidated: {self.reason}"


def _target_changed(rev: ExecutionRevision, current: Mapping[str, object]) -> bool:
    """Any admitted target fact that is provable on both sides and differs means the target moved."""
    for field in ("app", "pid", "window_id"):
        expected, actual = getattr(rev, field), current.get(field)
        if expected is not None and actual is not None and expected != actual:
            return True
    return False


class ExecutionState:
    """Per-profile revision bookkeeper. ``read_facts`` returns today's ground truth for the provable
    fields; ``admit()`` snapshots it, ``validate()`` diffs only the required dependencies."""

    def __init__(self, profile_key: str, read_facts: Callable[[], Mapping[str, object]]) -> None:
        self._profile_key = profile_key
        self._read_facts = read_facts

    def admit(self, *, app: Optional[str] = None, pid: Optional[int] = None,
              window_id: Optional[str] = None, snapshot_id: Optional[str] = None) -> ExecutionRevision:
        facts = self._read_facts()
        return ExecutionRevision(
            profile_key=self._profile_key,
            display_identity=facts.get("display_identity"),  # type: ignore[arg-type]
            backend_generation=facts.get("backend_generation"),  # type: ignore[arg-type]
            control_epoch=facts.get("control_epoch"),  # type: ignore[arg-type]
            app=app, pid=pid, window_id=window_id, snapshot_id=snapshot_id,
        )

    def validate(self, rev: ExecutionRevision, requires: AbstractSet[str] = _ALL_DEPS) -> RevisionVerdict:
        if rev.profile_key != self._profile_key:
            return RevisionVerdict(False, "profile_changed")
        current = self._read_facts()
        for dep in requires:
            if dep == TARGET:
                if _target_changed(rev, current):
                    return RevisionVerdict(False, _INVALIDATION_REASON[dep])
                continue
            field = {"display": "display_identity", "backend": "backend_generation",
                     "control": "control_epoch", "snapshot": "snapshot_id"}[dep]
            expected, actual = getattr(rev, field), current.get(field)
            if expected is None or actual is None:
                continue  # unprovable on one side: cannot call it stale
            if expected != actual:
                return RevisionVerdict(False, _INVALIDATION_REASON[dep])
        return RevisionVerdict(True, None)


def target_mismatch(admitted_app: Optional[str], requested_app: str) -> Optional[str]:
    """The sticky-target rule, against the revision's admitted target: a provable mismatch (both known,
    neither a substring of the other — 'Google-chrome' vs 'chrome') returns the admitted app, else None.
    Unknown target fails open; the verify ladder catches what this cannot prove."""
    current, wanted = (admitted_app or "").strip().lower(), requested_app.strip().lower()
    return None if not current or not wanted or wanted in current or current in wanted else admitted_app
