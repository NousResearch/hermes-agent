"""Soft timeout for tracked background tasks (#110427).

A long-running task used to be bounded only by a hard timer: the timer fired, the
process was killed, and the model re-ran the same work from scratch with a bigger
budget — losing every minute already spent. The soft threshold keeps what the
timeout exists for (surface the fact that a task is blowing its budget) without
destroying the work:

* crossing the threshold queues a notification on ``completion_queue`` — the same
  path every other background notification uses, so the CLI/gateway/TUI wake the
  agent with it — and leaves the process ALONE;
* the agent decides: ``process(action='kill')`` ends it, ``process(action='continue')``
  (optionally ``seconds=<fresh budget>``) or simply leaving it runs on. The timer
  never decides;
* every further threshold crossing re-notifies, so a continuing task never loops
  silently.

Scope: tracked background TASKS (``notify_on_complete`` sessions — the ones whose
result the agent waits on: over-cap runs promoted to the background, foreground
commands yielded to the background, explicit ``terminal(background=true,
notify_on_complete=true)``). Long-lived servers/watchers spawn without notify and
are untouched.

Config: ``timeouts.process.soft_timeout`` (seconds) in config.yaml, else
``HERMES_PROCESS_SOFT_TIMEOUT``, else :data:`DEFAULT_SOFT_TIMEOUT_SECONDS`. 0 or
negative disables the gate (defaults stay unchanged).
"""

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Same 10-minute budget a long foreground command gets before it is promoted to a
# tracked background process (``terminal_tool.FOREGROUND_MAX_TIMEOUT``): the soft
# threshold defaults to the budget tasks already run under, so default behavior is
# not regressed — only the "and now die" part is.
DEFAULT_SOFT_TIMEOUT_SECONDS = 600.0
_SOFT_TIMEOUT_CONFIG_KEY = "process.soft_timeout"
_SOFT_TIMEOUT_ENV_VAR = "HERMES_PROCESS_SOFT_TIMEOUT"

# Ticker cadence: half the armed budget (so a test-sized budget is observable), floored
# and capped so a tiny configured budget can't spin and a big one isn't polled idle.
_MIN_TICK_SECONDS = 0.25
_MAX_TICK_SECONDS = 5.0
# ponytail: the ticker self-exits after this many idle ticks (no running sessions) and
# the next spawn restarts it; a steady stream of spawns keeps the same one thread.
_IDLE_TICKS_BEFORE_EXIT = 12


def soft_timeout_seconds() -> float:
    """Resolved soft threshold in seconds; 0.0 = gate disabled."""
    from agent.deadline import resolve_timeout

    resolved = resolve_timeout(
        _SOFT_TIMEOUT_CONFIG_KEY, default=DEFAULT_SOFT_TIMEOUT_SECONDS, env_var=_SOFT_TIMEOUT_ENV_VAR,
    )
    try:
        return max(float(resolved or 0.0), 0.0)
    except (TypeError, ValueError):
        return DEFAULT_SOFT_TIMEOUT_SECONDS


class ProcessSoftTimeoutMixin:
    """Soft-threshold gate + ticker for :class:`~tools.process_registry.ProcessRegistry`.

    Nothing here kills: the only method that touches a process is
    :meth:`continue_soft_timeout`, which extends its budget. Killing stays where it
    always was — the agent's ``process(action='kill')``.
    """

    # --- arming ---------------------------------------------------------------

    def arm_soft_timeout(self, session, *, budget: Optional[float] = None, now: Optional[float] = None) -> float:
        """Arm *session*'s first soft window; deadline anchored on its spawn time so a
        lazily-armed session still fires at ``budget`` seconds of uptime, not ``budget``
        seconds after the first tick. Returns the effective budget (0 = disabled)."""
        now = time.time() if now is None else now
        if budget is None:
            budget = soft_timeout_seconds()
        try:
            budget = max(float(budget), 0.0)
        except (TypeError, ValueError):
            budget = 0.0
        session.soft_armed = True
        session.soft_hits = 0
        session.soft_budget = budget
        session.soft_deadline = ((session.started_at or now) + budget) if budget > 0 else 0.0
        return budget

    # --- the gate -------------------------------------------------------------

    def soft_timeout_notice(self, session, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """The notice to queue when *session* has crossed its soft threshold, else None.

        Pure state transition, no kill: it bumps the hit counter, opens the next window
        (so each further crossing re-notifies) and returns the event dict."""
        now = time.time() if now is None else now
        if not getattr(session, "soft_armed", False) or session.soft_deadline <= 0.0:
            return None
        if session.exited or now < session.soft_deadline:
            return None
        session.soft_hits += 1
        session.soft_deadline = now + session.soft_budget
        return {
            **self._watch_event_base(session),
            "type": "soft_timeout",
            "hit": session.soft_hits,
            "budget_seconds": int(session.soft_budget),
            "elapsed_seconds": (int(now - session.started_at) if session.started_at else None),
        }

    def check_soft_timeouts(self, now: Optional[float] = None) -> int:
        """Arm + notify every running soft-timed task past its threshold. Returns the
        number of notices queued. Never kills and never blocks (tick-safe)."""
        now = time.time() if now is None else now
        with self._lock:
            sessions = list(self._running.values())
        queued = 0
        for session in sessions:
            # Only tasks the agent is waiting on carry a soft gate; a server/watcher
            # (no notify_on_complete) running for days must not nag the agent.
            if session.exited or not session.notify_on_complete:
                continue
            if not getattr(session, "soft_armed", False):
                self.arm_soft_timeout(session, now=now)
            evt = self.soft_timeout_notice(session, now)
            if evt is None:
                continue
            from tools.process_registry import _redact_process_result

            _redact_process_result(evt)
            self.completion_queue.put(evt)
            queued += 1
            logger.info(
                "Soft timeout for background process %s (notice #%d, %ds budget, %.0fs elapsed) — "
                "left running; the agent decides kill/continue",
                session.id, evt["hit"], evt["budget_seconds"], now - (session.started_at or now))
        return queued

    # --- the agent's explicit decisions ---------------------------------------

    def continue_soft_timeout(self, session_id: str, seconds: Optional[Any] = None) -> Dict[str, Any]:
        """``process(action='continue')``: keep the task running, optionally with a fresh
        budget. A decision, never a kill — an exited/unknown process is reported, not guessed."""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "session_id": session_id,
                    "error": f"no background process {session_id!r}"}
        if session.exited:
            return {"status": "exited", "session_id": session.id, "exit_code": session.exit_code,
                    "error": "the process already exited; read its result with process(action='log') "
                             "instead of continuing it."}
        try:
            budget = float(seconds) if seconds is not None else 0.0
        except (TypeError, ValueError):
            budget = 0.0
        if budget <= 0:
            budget = session.soft_budget or soft_timeout_seconds()
        if budget <= 0:
            return {"status": "error", "session_id": session.id,
                    "error": "soft timeout 'seconds' must be a positive number (or leave it out to reuse "
                             "the configured budget)."}
        session.soft_armed = True
        session.soft_budget = budget
        session.soft_deadline = time.time() + budget
        logger.info("Soft timeout for %s continued by the agent with a %ds budget", session.id, int(budget))
        return {
            "status": "continuing", "session_id": session.id, "command": session.command,
            "soft_timeout_seconds": int(budget),
            "note": (f"Still running; you'll be notified again if it passes {int(budget)}s more. "
                     f"Nothing was killed."),
        }

    # --- ticker ---------------------------------------------------------------

    def _ensure_soft_ticker(self) -> None:
        """Start the soft-timeout ticker once per registry (called on spawn)."""
        with self._soft_ticker_lock:
            if self._soft_ticker is not None and self._soft_ticker.is_alive():
                return
            self._soft_ticker = threading.Thread(
                target=self._soft_ticker_loop, daemon=True, name="proc-soft-timeout")
            self._soft_ticker.start()

    def _soft_tick_seconds(self) -> float:
        with self._lock:
            sessions = list(self._running.values())
        budgets = [s.soft_budget for s in sessions if getattr(s, "soft_armed", False) and s.soft_budget > 0]
        if not budgets:
            return _MAX_TICK_SECONDS
        return max(_MIN_TICK_SECONDS, min(_MAX_TICK_SECONDS, min(budgets) / 2.0))

    def _soft_ticker_loop(self) -> None:
        idle = 0
        while True:
            try:
                self.check_soft_timeouts()
                with self._lock:
                    running = len(self._running)
                tick = self._soft_tick_seconds()
            except Exception:
                logger.debug("soft-timeout tick failed", exc_info=True)
                running = 1  # keep ticking; a broken tick must not silently disable the gate
                tick = _MAX_TICK_SECONDS
            idle = 0 if running else idle + 1
            if idle >= _IDLE_TICKS_BEFORE_EXIT:
                return
            time.sleep(tick)
