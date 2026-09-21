"""Fail-closed authorization snapshot for computer-use operations."""
from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any

@dataclass(frozen=True)
class ExecutionRevision:
    profile_key: str
    display_identity: str | None
    backend_generation: int
    control_epoch: int | None
    pid: Any = None
    window_id: Any = None
    snapshot_id: str | None = None

class StaleRevision(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)

def _display():
    if wayland := os.environ.get("WAYLAND_DISPLAY"):
        return f"wayland:{wayland}"
    if display := os.environ.get("DISPLAY"):
        return f"x11:{display}"
    return None

def _epoch(backend):
    provider = getattr(backend, "control_epoch_provider", None)
    return provider() if callable(provider) else getattr(backend, "control_epoch", None)

def _snapshot(backend):
    tokens = getattr(backend, "_snapshot_tokens", None)
    if not tokens:
        return None
    return hashlib.sha256(json.dumps(sorted(map(str, tokens.values()))).encode()).hexdigest()

def admit(backend, action, args):
    from hermes_constants import hermes_home_key
    return ExecutionRevision(hermes_home_key(), _display(), int(getattr(backend, "backend_generation", 1)),
                            _epoch(backend), getattr(backend, "_active_pid", None),
                            getattr(backend, "_active_window_id", None), _snapshot(backend))

def validate(rev, backend, *, deps):
    from hermes_constants import hermes_home_key
    checks = (("profile", rev.profile_key, hermes_home_key(), "profile_changed"),
              ("display", rev.display_identity, _display(), "display_changed"),
              ("backend", rev.backend_generation, int(getattr(backend, "backend_generation", 1)), "backend_changed"),
              ("epoch", rev.control_epoch, _epoch(backend), "epoch_changed"),
              ("target", (rev.pid, rev.window_id), (getattr(backend, "_active_pid", None), getattr(backend, "_active_window_id", None)), "target_changed"),
              ("snapshot", rev.snapshot_id, _snapshot(backend), "snapshot_changed"))
    for dep, old, new, reason in checks:
        if dep in deps and old != new:
            raise StaleRevision(reason)
