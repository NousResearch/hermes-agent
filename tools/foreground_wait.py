"""Cooperative handoff for tool calls that can leave foreground waits running.

The agent owns a registry; the tool executor binds one slot to the worker thread
before invoking a supported blocking tool.  A forced durable steer requests a
handoff through that slot instead of interrupting or killing the underlying
work.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_current = threading.local()


@dataclass
class ForegroundWaitSlot:
    tool_call_id: str
    kind: str
    background_requested: threading.Event = field(default_factory=threading.Event)
    _resolved: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _status: str = "waiting"
    _handoff: Optional[Dict[str, Any]] = None
    _error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete_background(self, handoff: Dict[str, Any]) -> None:
        """Resolve this wait as successfully moved to background."""
        with self._lock:
            if self._resolved.is_set():
                return
            self._status = "backgrounded"
            self._handoff = dict(handoff)
            self._resolved.set()

    def complete_naturally(self) -> None:
        """Resolve a force/completion race where the foreground work finished."""
        with self._lock:
            if self._resolved.is_set():
                return
            self._status = "completed"
            self._resolved.set()

    def fail_background(self, error: str) -> None:
        with self._lock:
            if self._resolved.is_set():
                return
            self._status = "failed"
            self._error = str(error or "foreground handoff failed")
            self._resolved.set()

    def wait_for_resolution(self, timeout: float) -> Dict[str, Any]:
        if not self._resolved.wait(timeout):
            return {
                "status": "failed",
                "error": "timed out while moving foreground wait to background",
            }
        with self._lock:
            result: Dict[str, Any] = {"status": self._status}
            if self._handoff is not None:
                result["handoff"] = dict(self._handoff)
            if self._error:
                result["error"] = self._error
            return result


class ForegroundWaitRegistry:
    """Thread-safe set of supported foreground waits owned by one agent."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._slots: Dict[str, ForegroundWaitSlot] = {}

    def register(self, tool_call_id: str, kind: str) -> ForegroundWaitSlot:
        slot = ForegroundWaitSlot(str(tool_call_id), str(kind))
        with self._lock:
            self._slots[slot.tool_call_id] = slot
        return slot

    def unregister(self, slot: ForegroundWaitSlot) -> None:
        with self._lock:
            if self._slots.get(slot.tool_call_id) is slot:
                self._slots.pop(slot.tool_call_id, None)
        slot.complete_naturally()

    def snapshot(self) -> List[ForegroundWaitSlot]:
        with self._lock:
            return list(self._slots.values())

    def request_background(
        self,
        slots: Optional[List[ForegroundWaitSlot]] = None,
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        slots = list(slots) if slots is not None else self.snapshot()
        kinds = sorted({slot.kind for slot in slots})
        if not slots:
            return {"status": "no_wait", "wait_kinds": []}

        for slot in slots:
            slot.background_requested.set()

        handoffs: List[Dict[str, Any]] = []
        failures: List[str] = []
        for slot in slots:
            outcome = slot.wait_for_resolution(timeout)
            if outcome["status"] == "backgrounded":
                handoff = outcome.get("handoff")
                if isinstance(handoff, dict):
                    handoffs.append(handoff)
            elif outcome["status"] != "completed":
                failures.append(str(outcome.get("error") or "foreground handoff failed"))

        if failures:
            return {
                "status": "failed",
                "wait_kinds": kinds,
                "errors": failures,
                "handoffs": handoffs,
            }
        return {
            "status": "backgrounded",
            "wait_kinds": kinds,
            "handoffs": handoffs,
        }


def set_current_foreground_wait(slot: Optional[ForegroundWaitSlot]) -> None:
    _current.slot = slot


def current_foreground_wait() -> Optional[ForegroundWaitSlot]:
    slot = getattr(_current, "slot", None)
    return slot if isinstance(slot, ForegroundWaitSlot) else None


def process_handoff(session_id: str) -> Dict[str, str]:
    """Return the canonical recovery contract for one adopted process."""
    return {
        "kind": "process",
        "session_id": session_id,
        "continue": f'process(action="wait", session_id="{session_id}")',
        "inspect": f'process(action="log", session_id="{session_id}")',
        "stop": f'process(action="kill", session_id="{session_id}")',
    }


def supported_wait_kind(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    if tool_name == "terminal" and not bool(args.get("background", False)):
        return "terminal"
    if tool_name == "delegate_task" and str(args.get("action") or "").lower() == "wait":
        return "delegation"
    return None


@contextmanager
def track_foreground_wait(
    agent, tool_call_id: str, tool_name: str, args: Dict[str, Any]
):
    """Bind a supported wait slot around one registry-dispatched tool call."""
    kind = supported_wait_kind(tool_name, args)
    registry = getattr(agent, "_foreground_waits", None)
    if not kind or registry is None:
        yield None
        return

    slot = registry.register(tool_call_id, kind)
    previous = current_foreground_wait()
    set_current_foreground_wait(slot)
    try:
        yield slot
    finally:
        set_current_foreground_wait(previous)
        registry.unregister(slot)
