"""Opt-in interrupt/poll tracing for ``BaseEnvironment._wait_for_process``
(``HERMES_DEBUG_INTERRUPT=1``): loop entry/exit, interrupt/timeout detection and ~30s
heartbeats so "agent never sees the interrupt" reports can be diagnosed from agent.log."""

import threading
import time

from tools.interrupt import is_interrupted

_FMT = {
    "ENTER": "timeout=%ss activity_cb=%s initial_interrupt=%s",
    "INTERRUPT DETECTED": "iter=%d elapsed=%.1fs — killing process group",
    "TIMEOUT": "iter=%d timeout=%ss",
    "HEARTBEAT": "iter=%d elapsed=%.0fs interrupt=%s activity_cb=%s%s",
    "EXCEPTION_EXIT": "iter=%d elapsed=%.1fs — killing subprocess group before re-raise",
    "EXIT (natural)": "iter=%d elapsed=%.1fs returncode=%s"}


class _WaitTrace:
    """Per-wait debug tracer; every method is a no-op unless ``enabled``."""

    def __init__(self, proc, timeout, *, enabled: bool, logger):
        self.enabled = enabled
        self.iterations = 0
        if not enabled:
            return
        from tools.environments.base import get_activity_callback

        self._get_cb, self._log, self._timeout = get_activity_callback, logger, timeout
        self._tid = threading.current_thread().ident
        self._pid = getattr(proc, "pid", None)
        self._start = self._last_heartbeat = time.monotonic()
        self._cb_was_none = get_activity_callback() is None

    def _emit(self, event: str, *args) -> None:
        self._log.info(
            "[interrupt-debug] _wait_for_process %s tid=%s pid=%s " + _FMT[event],
            event, self._tid, self._pid, *args)

    def _elapsed(self) -> float:
        return time.monotonic() - self._start

    def enter(self) -> None:
        if self.enabled:
            self._emit("ENTER", self._timeout,
                       "set" if not self._cb_was_none else "MISSING", is_interrupted())

    def interrupted(self) -> None:
        if self.enabled:
            self._emit("INTERRUPT DETECTED", self.iterations, self._elapsed())

    def timed_out(self) -> None:
        if self.enabled:
            self._emit("TIMEOUT", self.iterations, self._timeout)

    def heartbeat(self) -> None:
        """Every ~30s: proves the loop is alive and reports whether the thread-local
        activity callback got clobbered by nested tool calls / executor thread reuse."""
        if not self.enabled or time.monotonic() - self._last_heartbeat < 30.0:
            return
        cb_now_none = self._get_cb() is None
        self._emit(
            "HEARTBEAT", self.iterations, self._elapsed(), is_interrupted(),
            "set" if not cb_now_none else "MISSING",
            " (LOST during run)" if cb_now_none and not self._cb_was_none else "")
        self._last_heartbeat = time.monotonic()
        self._cb_was_none = cb_now_none

    def exception_exit(self) -> None:
        if self.enabled:
            self._emit("EXCEPTION_EXIT", self.iterations, self._elapsed())

    def natural_exit(self, returncode) -> None:
        if self.enabled:
            self._emit("EXIT (natural)", self.iterations, self._elapsed(), returncode)
