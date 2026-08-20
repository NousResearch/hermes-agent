"""observer.py — Optional observability side-channel for Claude sessions.

Runs in a background thread, polls tmux output, and reports activity
to logs or callbacks. Completely independent from the core pipeline —
does not block send/wait_for_idle, and can be disabled without
affecting functionality.

Uses adaptive polling: interval varies by detected session state to
balance responsiveness with resource efficiency.
"""

import logging
import threading
import time
from typing import Callable, Optional

from tools.claude_session.idle import (
    SessionState, clean_lines, detect_state, detect_activity,
)
from tools.claude_session.tmux_interface import TmuxInterface
from tools.claude_session.output_buffer import OutputBuffer

logger = logging.getLogger(__name__)

# Adaptive polling intervals (seconds) keyed by session state.
# THINKING: Claude is generating — output changes slowly, low frequency ok.
# TOOL_CALL: Tools executing — state changes fast, need timely updates.
# PERMISSION: Waiting for user — moderate frequency.
# IDLE: Session quiet — minimal polling.
# UNKNOWN / fallback: Use default interval from constructor.
_STATE_INTERVALS = {
    "THINKING": 5.0,
    "TOOL_CALL": 2.0,
    "PERMISSION": 3.0,
    "INTERVIEW": 3.0,
    "IDLE": 15.0,
}


class SessionObserver:
    """Background observer for Claude Code session activity.

    Non-blocking, non-essential. Use for:
    - Progress monitoring / logging
    - Status callbacks to external systems (e.g. Telegram)
    - Stalled-session detection

    Polling is adaptive: interval adjusts based on detected session state.
    TOOL_CALL → 2s (fast feedback), THINKING → 180s (slow changes),
    IDLE → 15s (minimal), PERMISSION → 3s (moderate).

    Usage:
        observer = SessionObserver(tmux, buffer, on_update=my_callback)
        observer.start()
        # ... session runs ...
        observer.stop()
    """

    def __init__(
        self,
        tmux: TmuxInterface,
        buffer: OutputBuffer,
        on_update: Optional[Callable[[dict], None]] = None,
        poll_interval: float = 5.0,
    ):
        self._tmux = tmux
        self._buf = buffer
        self._on_update = on_update
        self._default_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()  # signaled by poll_now()

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="session-observer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def poll_now(self) -> None:
        """Force an immediate poll (also updates buffer).

        Signals the loop thread to wake up and poll immediately,
        rather than calling _poll_once directly (avoids concurrent
        buffer writes from two threads).
        """
        self._wake_event.set()

    def _get_interval(self, state_name: str) -> float:
        """Return adaptive poll interval for the given session state."""
        return _STATE_INTERVALS.get(state_name, self._default_interval)

    def _loop(self) -> None:
        current_interval = self._default_interval
        while not self._stop_event.is_set():
            state_name = None
            try:
                state_name = self._poll_once()
            except Exception as e:
                logger.warning("Observer poll error: %s", e)
            current_interval = self._get_interval(state_name) if state_name else self._default_interval
            # Wait for either timeout or wake signal from poll_now()
            self._wake_event.clear()
            self._wake_event.wait(timeout=current_interval)

    def _poll_once(self) -> Optional[str]:
        """Poll tmux once. Returns the detected state name, or None."""
        if not self._tmux.session_exists():
            return None

        raw = self._tmux.capture_pane()
        lines = clean_lines(raw)

        if lines:
            self._buf.append_batch(lines)

        # Always detect state (needed for adaptive polling even without callback)
        state_result = detect_state(lines)
        state_name = state_result.state

        if self._on_update:
            activity = detect_activity(lines)
            self._on_update({
                "state": state_result.state,
                "tool_name": state_result.tool_name,
                "tool_target": state_result.tool_target,
                "current_activity": activity["activity"],
                "activity_detail": activity["detail"],
                "output_lines": len(lines),
            })
        return state_name
