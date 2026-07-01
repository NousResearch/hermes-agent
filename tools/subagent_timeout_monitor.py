#!/usr/bin/env python3
"""
Subagent Timeout Monitor

Monitors subagent activity during delegation and handles timeouts,
zombie processes, and hung sessions. Integrates with the existing
heartbeat staleness monitor in delegate_tool.py but adds:

1. Active interruption of stale subagents (not just stopping heartbeats)
2. Configurable timeout thresholds per subagent
3. Structured event logging for all timeout/interrupt events
4. Automatic retry logic with configurable max retries and backoff

Usage:
    monitor = SubagentTimeoutMonitor(
        stale_threshold_seconds=300,  # 5 min idle between turns -> interrupt
        in_tool_threshold_seconds=900,  # 15 min stuck on same tool -> interrupt
        max_retries=2,
    )

    # In _run_single_child heartbeat loop:
    monitor.check_activity(child, task_index)

    # After child completes or times out:
    result = monitor.on_child_exit(child_result, task_index)

Config keys (in delegation section of config.yaml):
    stale_threshold_seconds: seconds before idle subagent is interrupted (default 300)
    in_tool_threshold_seconds: seconds before in-tool subagent is interrupted (default 900)
    max_retries: how many times to retry a timed-out subagent (default 0 = no retry)
    auto_interrupt: whether to actively interrupt stale subagents (default true)
"""

import enum
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TimeoutEventKind(str, enum.Enum):
    """Types of timeout events the monitor can emit."""

    STALE_DETECTED = "stale_detected"
    AUTO_INTERRUPTED = "auto_interrupted"
    TIMEOUT_RETRY = "timeout_retry"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    MANUAL_INTERRUPT = "manual_interrupt"


class TimeoutEvent:
    """Structured event emitted when a subagent timeout or interrupt occurs."""

    def __init__(
        self,
        kind: TimeoutEventKind,
        task_index: int,
        subagent_id: Optional[str],
        duration_seconds: float,
        api_calls: int,
        current_tool: Optional[str],
        stale_cycles: int,
        retry_count: int = 0,
        max_retries: int = 0,
        details: Optional[str] = None,
    ):
        self.kind = kind
        self.task_index = task_index
        self.subagent_id = subagent_id
        self.duration_seconds = duration_seconds
        self.api_calls = api_calls
        self.current_tool = current_tool
        self.stale_cycles = stale_cycles
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.details = details
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "task_index": self.task_index,
            "subagent_id": self.subagent_id,
            "duration_seconds": round(self.duration_seconds, 2),
            "api_calls": self.api_calls,
            "current_tool": self.current_tool,
            "stale_cycles": self.stale_cycles,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "details": self.details,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        return (
            f"TimeoutEvent({self.kind.value}, task={self.task_index}, "
            f"sida={self.subagent_id}, duration={self.duration_seconds:.1f}s, "
            f"api_calls={self.api_calls}, retries={self.retry_count}/{self.max_retries})"
        )


class SubagentTimeoutMonitor:
    """Monitors subagent activity and handles timeouts with auto-interrupt and retry.

    Tracks per-subagent state (iteration count, current tool, stale cycle count)
    and fires events when thresholds are exceeded. Optionally interrupts hung
    subagents automatically and tracks retry counts for failed tasks.

    Thread-safe: all state is protected by an internal lock.
    """

    def __init__(
        self,
        stale_threshold_seconds: Optional[float] = None,
        in_tool_threshold_seconds: Optional[float] = None,
        max_retries: int = 0,
        auto_interrupt: bool = True,
        heartbeat_interval: float = 30.0,
        event_callback: Optional[Callable[[TimeoutEvent], None]] = None,
    ):
        """Initialize the timeout monitor.

        Args:
            stale_threshold_seconds: Seconds of idle time (no iteration advance,
                no tool change) before considering a subagent stale. Default 300s.
            in_tool_threshold_seconds: Seconds of time stuck on the same tool before
                considering a subagent stale. Default 900s.
            max_retries: Maximum number of retries for a timed-out subagent.
                Default 0 (no retry).
            auto_interrupt: Whether to actively interrupt stale subagents.
                Default True.
            heartbeat_interval: Expected interval between heartbeat checks in seconds.
                Used to calculate stale cycle counts. Default 30s.
            event_callback: Optional callback invoked for each timeout event.
                Receives a TimeoutEvent instance.
        """
        self.stale_threshold_seconds = stale_threshold_seconds or 300.0
        self.in_tool_threshold_seconds = in_tool_threshold_seconds or 900.0
        self.max_retries = max(0, max_retries)
        self.auto_interrupt = auto_interrupt
        self.heartbeat_interval = heartbeat_interval

        # stale_threshold_cycles and in_tool_threshold_cycles are derived from
        # the threshold seconds so the heartbeat loop can compare cycle counts.
        self.stale_threshold_cycles = max(1, int(self.stale_threshold_seconds / self.heartbeat_interval))
        self.in_tool_threshold_cycles = max(1, int(self.in_tool_threshold_seconds / self.heartbeat_interval))

        self.event_callback = event_callback

        # Per-subagent state: task_index -> {iter, tool, stale_count, interrupted}
        self._state_lock = threading.Lock()
        # task_index -> state dict
        self._subagent_state: Dict[int, Dict[str, Any]] = {}

        # Retry tracking
        self._retry_lock = threading.Lock()
        # task_index -> retry count
        self._retry_counts: Dict[int, int] = {}

        # Event log (thread-safe list)
        self._events_lock = threading.Lock()
        self._events: List[TimeoutEvent] = []

    def _get_state(self, task_index: int) -> Dict[str, Any]:
        """Get or create state for a subagent. Must be called with lock held."""
        if task_index not in self._subagent_state:
            self._subagent_state[task_index] = {
                "last_iter": 0,
                "last_tool": None,
                "stale_count": 0,
                "interrupted": False,
                "start_time": time.monotonic(),
            }
        return self._subagent_state[task_index]

    def _emit_event(self, event: TimeoutEvent) -> None:
        """Emit a timeout event to the log and optional callback."""
        with self._events_lock:
            self._events.append(event)

        logger.warning(
            "Subagent timeout event: %s",
            repr(event),
        )

        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as exc:
                logger.debug("Timeout event callback failed: %s", exc)

    def check_activity(
        self,
        child_agent: Any,
        task_index: int,
    ) -> Optional[TimeoutEvent]:
        """Check if a subagent has become stale and optionally interrupt it.

        Call this from the heartbeat loop on each tick. Returns a TimeoutEvent
        if the subagent was found stale and interrupted, or None otherwise.

        Args:
            child_agent: The child AIAgent instance (must have get_activity_summary).
            task_index: Integer index identifying this subagent.

        Returns:
            TimeoutEvent if stale and interrupted, None otherwise.
        """
        try:
            child_summary = child_agent.get_activity_summary()
        except Exception as exc:
            logger.debug("Failed to get child activity summary for task %d: %s", task_index, exc)
            return None

        child_iter = int(child_summary.get("api_call_count", 0) or 0)
        child_tool = child_summary.get("current_tool")

        with self._state_lock:
            state = self._get_state(task_index)

            # Skip if already interrupted — don't re-fire
            if state["interrupted"]:
                return None

            # Check for progress
            iter_advanced = child_iter > state["last_iter"]
            tool_changed = child_tool != state["last_tool"]

            if iter_advanced or tool_changed:
                # Progress detected — reset stale counter and update state
                state["last_iter"] = child_iter
                state["last_tool"] = child_tool
                state["stale_count"] = 0
                return None

            # No progress — increment stale counter
            state["stale_count"] += 1

            # Determine threshold based on whether child is inside a tool
            current_stale = state["stale_count"]
            threshold = (
                self.in_tool_threshold_cycles
                if child_tool
                else self.stale_threshold_cycles
            )

            if current_stale < threshold:
                return None

            # Stale threshold exceeded — mark as interrupted to prevent re-firing
            state["interrupted"] = True

        # Build event outside the lock (state is already marked)
        duration = time.monotonic() - self._subagent_state[task_index]["start_time"]
        subagent_id = getattr(child_agent, "_subagent_id", None)

        event = TimeoutEvent(
            kind=TimeoutEventKind.STALE_DETECTED,
            task_index=task_index,
            subagent_id=subagent_id,
            duration_seconds=duration,
            api_calls=child_iter,
            current_tool=child_tool,
            stale_cycles=current_stale,
            retry_count=self._retry_counts.get(task_index, 0),
            max_retries=self.max_retries,
            details=(
                f"Stale for {current_stale} heartbeat cycles "
                f"(threshold={threshold}, tool={child_tool or '<none>'})"
            ),
        )
        self._emit_event(event)

        # Auto-interrupt if enabled
        if self.auto_interrupt:
            try:
                tool_label = child_tool or "none"
                child_agent.interrupt(
                    f"Subagent interrupted by timeout monitor: stale for "
                    f"{current_stale} cycles ({duration:.0f}s), tool={tool_label}"
                )
            except Exception as exc:
                logger.warning(
                    "Failed to interrupt stale subagent %d: %s", task_index, exc
                )

            interrupt_event = TimeoutEvent(
                kind=TimeoutEventKind.AUTO_INTERRUPTED,
                task_index=task_index,
                subagent_id=subagent_id,
                duration_seconds=duration,
                api_calls=child_iter,
                current_tool=child_tool,
                stale_cycles=current_stale,
                retry_count=self._retry_counts.get(task_index, 0),
                max_retries=self.max_retries,
                details=f"Auto-interrupted after {duration:.0f}s of inactivity",
            )
            self._emit_event(interrupt_event)

        return event

    def on_child_exit(
        self,
        result: Dict[str, Any],
        task_index: int,
    ) -> Optional[TimeoutEvent]:
        """Handle a child exit (timeout or error) and manage retry state.

        Call this after _run_single_child returns with a timeout or error status.
        Increments the retry counter and emits events for retry decisions.

        Args:
            result: The result dict from _run_single_child.
            task_index: Integer index identifying this subagent.

        Returns:
            TimeoutEvent if a retry was triggered, None otherwise.
        """
        status = result.get("status", "")
        if status not in ("timeout", "error"):
            return None

        with self._state_lock:
            state = self._get_state(task_index)
            duration = time.monotonic() - state["start_time"]

        subagent_id = result.get("subagent_id")
        api_calls = int(result.get("api_calls", 0) or 0)

        # Get current retry count
        with self._retry_lock:
            retry_count = self._retry_counts.get(task_index, 0)

        if status == "timeout" and retry_count < self.max_retries:
            # Schedule a retry
            with self._retry_lock:
                self._retry_counts[task_index] = retry_count + 1

            event = TimeoutEvent(
                kind=TimeoutEventKind.TIMEOUT_RETRY,
                task_index=task_index,
                subagent_id=subagent_id,
                duration_seconds=duration,
                api_calls=api_calls,
                current_tool=result.get("current_tool"),
                stale_cycles=state.get("stale_count", 0),
                retry_count=retry_count + 1,
                max_retries=self.max_retries,
                details=f"Retry {retry_count + 1}/{self.max_retries} after timeout",
            )
            self._emit_event(event)
            return event

        if status == "timeout" and retry_count >= self.max_retries:
            # Max retries exceeded — mark as final failure
            event = TimeoutEvent(
                kind=TimeoutEventKind.MAX_RETRIES_EXCEEDED,
                task_index=task_index,
                subagent_id=subagent_id,
                duration_seconds=duration,
                api_calls=api_calls,
                current_tool=result.get("current_tool"),
                stale_cycles=state.get("stale_count", 0),
                retry_count=retry_count,
                max_retries=self.max_retries,
                details=f"Max retries ({self.max_retries}) exceeded after timeout",
            )
            self._emit_event(event)
            return event

        return None

    def should_retry(
        self,
        result: Dict[str, Any],
        task_index: int,
    ) -> bool:
        """Check if a timed-out subagent should be retried.

        Args:
            result: The result dict from _run_single_child.
            task_index: Integer index identifying this subagent.

        Returns:
            True if the subagent should be retried, False otherwise.
        """
        status = result.get("status", "")
        if status != "timeout":
            return False

        with self._retry_lock:
            retry_count = self._retry_counts.get(task_index, 0)

        return retry_count < self.max_retries

    def get_retry_count(self, task_index: int) -> int:
        """Get the current retry count for a subagent."""
        with self._retry_lock:
            return self._retry_counts.get(task_index, 0)

    def reset_state(self, task_index: int) -> None:
        """Reset monitoring state for a subagent (e.g., before a retry)."""
        with self._state_lock:
            if task_index in self._subagent_state:
                state = self._subagent_state[task_index]
                state["last_iter"] = 0
                state["last_tool"] = None
                state["stale_count"] = 0
                state["interrupted"] = False
                state["start_time"] = time.monotonic()

    def get_events(self) -> List[TimeoutEvent]:
        """Return a copy of all timeout events."""
        with self._events_lock:
            return list(self._events)

    def get_events_since(self, since_timestamp: float) -> List[TimeoutEvent]:
        """Return events emitted after a given timestamp."""
        with self._events_lock:
            return [e for e in self._events if e.timestamp > since_timestamp]

    def get_active_subagents(self) -> List[Dict[str, Any]]:
        """Return state for all currently monitored subagents."""
        with self._state_lock:
            result = []
            for idx, state in self._subagent_state.items():
                result.append({
                    "task_index": idx,
                    "last_iter": state["last_iter"],
                    "last_tool": state["last_tool"],
                    "stale_count": state["stale_count"],
                    "interrupted": state["interrupted"],
                    "duration_seconds": round(time.monotonic() - state["start_time"], 2),
                })
            return result

    def clear_events(self) -> int:
        """Clear the event log. Returns the number of events cleared."""
        with self._events_lock:
            count = len(self._events)
            self._events.clear()
        return count


# Module-level singleton for convenience. The delegate_tool heartbeat loop
# can import and use this directly without constructing a new instance each time.
_default_monitor: Optional[SubagentTimeoutMonitor] = None
_default_monitor_lock = threading.Lock()


def get_default_monitor() -> SubagentTimeoutMonitor:
    """Get or create the default timeout monitor instance.

    Reads configuration from the delegation section of config.yaml via
    _load_config() (same path as delegate_tool). Creates a singleton on
    first call.
    """
    global _default_monitor
    if _default_monitor is not None:
        return _default_monitor

    with _default_monitor_lock:
        if _default_monitor is not None:
            return _default_monitor

        # Read config — import here to avoid circular deps at module load
        try:
            from tools.delegate_tool import _load_config

            cfg = _load_config()
        except Exception:
            cfg = {}

        stale_seconds = cfg.get("stale_threshold_seconds")
        in_tool_seconds = cfg.get("in_tool_threshold_seconds")
        max_retries_val = cfg.get("max_retries", 0)
        auto_interrupt_val = cfg.get("auto_interrupt", True)

        try:
            stale_seconds = float(stale_seconds) if stale_seconds is not None else 300.0
        except (TypeError, ValueError):
            stale_seconds = 300.0

        try:
            in_tool_seconds = float(in_tool_seconds) if in_tool_seconds is not None else 900.0
        except (TypeError, ValueError):
            in_tool_seconds = 900.0

        try:
            max_retries_val = int(max_retries_val)
        except (TypeError, ValueError):
            max_retries_val = 0

        _default_monitor = SubagentTimeoutMonitor(
            stale_threshold_seconds=stale_seconds,
            in_tool_threshold_seconds=in_tool_seconds,
            max_retries=max(max_retries_val, 0),
            auto_interrupt=bool(auto_interrupt_val),
        )

    return _default_monitor


def reset_default_monitor() -> None:
    """Reset the default monitor singleton. Useful for testing."""
    global _default_monitor
    with _default_monitor_lock:
        _default_monitor = None
