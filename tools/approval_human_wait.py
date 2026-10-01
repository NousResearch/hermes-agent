"""Human-wait accounting for :mod:`tools.approval` (per session).

Tracks wall-clock time the agent spends verifiably blocked on a HUMAN prompt
(CLI approval prompt, gateway approval round-trip). The concurrent tool batch
deadline in agent/tool_executor.py excludes this time so a slow human answer
never times a batch out — but ONLY this time. Measuring at the source (rather
than residency in the authorization gate, which is arbitrary code) is what keeps
a wedged pre_tool_call plugin or a dead approval client from growing the
exclusion 1:1 with wall clock and defeating the deadline entirely. Keyed by
session so one gateway session's pending approval cannot extend a different
session's batch deadline; state is process-global like the rest of the approval
state, bounded by _HUMAN_WAIT_MAX_SESSIONS.
"""

import contextlib
import threading
import time


# ========================================================================= Human-wait accounting (per
# session) ========================================================================= Tracks the wall-clock
# time the agent spends verifiably blocked on a HUMAN prompt (CLI approval prompt, gateway approval
# round-trip). The concurrent tool batch deadline in agent/tool_executor.py excludes this time so a slow
# human answer never times a batch out — but ONLY this time. Measuring human waits at the source (rather
# than residency in the authorization gate, which is arbitrary code) is what keeps a wedged pre_tool_call
# plugin or a dead approval client from growing the exclusion 1:1 with wall clock and defeating the deadline
# entirely (#79719). Keyed by session so one gateway session's pending approval cannot extend a different
# session's batch deadline. State is process-global like the rest of this module's approval state; entries
# are bounded by _HUMAN_WAIT_MAX_SESSIONS.
class _HumanWaitState:
    __slots__ = ("pending", "window_started", "completed_seconds")

    def __init__(self) -> None:
        self.pending = 0
        self.window_started: float | None = None
        self.completed_seconds = 0.0


_human_wait_lock = threading.Lock()
_human_wait_states: dict[str, _HumanWaitState] = {}
_HUMAN_WAIT_MAX_SESSIONS = 256
# Margin added on top of approvals.timeout when clamping a window's contribution (read-side AND close-side) and when
# bounding the authorization gate's serialization-lock acquire in agent/tool_executor.py. One constant so the clamps
# can't drift apart.
HUMAN_WAIT_MARGIN_S = 60.0


def human_wait_ceiling() -> float:
    """Max seconds a single window may contribute: approvals.timeout + margin.
    Every legitimate human wait self-terminates at ``approvals.timeout`` (the CLI
    prompt join and the gateway poll loop both enforce it), so a window that
    overstays this ceiling is itself wedged and must not keep extending a batch
    deadline. Also the bound on the authorization gate's serialization-lock
    acquire in agent/tool_executor.py, so the two cannot drift. Never call while
    holding ``_human_wait_lock`` — it reads the config cache.
    ``_get_approval_timeout`` caps at ``agent.deadline.MAX_SAFE_TIMEOUT_S`` so the
    value is always safe for ``Lock.acquire(timeout=...)`` / ``Thread.join(timeout=...)``."""
    from tools import approval_context
    return float(approval_context._get_approval_timeout()) + HUMAN_WAIT_MARGIN_S


def _clamped_window_seconds(started: float, now: float, ceiling: float) -> float:
    """Seconds an open window contributes: elapsed, floored at 0, capped. Shared
    by the close-time accrual and the open-window read so the two clamps stay
    identical by construction."""
    return min(max(0.0, now - started), ceiling)


def _human_wait_state(session_key: str) -> _HumanWaitState:
    """Return (creating if needed) the wait state for *session_key*. Caller must
    hold ``_human_wait_lock``. Evicts idle entries (no pending waiter)
    insertion-order-first until the table is under the cap so an army of
    short-lived session keys cannot grow it without bound. Entries with an open
    window are never evicted (that would corrupt live accounting), so the cap is
    best-effort under 256+ concurrently-pending sessions."""
    state = _human_wait_states.get(session_key)
    if state is None:
        for key in list(_human_wait_states):
            if len(_human_wait_states) < _HUMAN_WAIT_MAX_SESSIONS:
                break
            if _human_wait_states[key].pending == 0:
                del _human_wait_states[key]
        state = _human_wait_states[session_key] = _HumanWaitState()
    return state


def _resolve_key(session_key: str | None) -> str:
    if session_key is not None:
        return session_key
    from tools import approval_context
    return approval_context.get_current_session_key()


def activity_heartbeat(label: str):
    """Callable that pings the agent's inactivity tracker (at most every ~10s)
    while a human wait is parked, so the gateway watchdog does not kill the agent
    while the user is still answering. No-op in minimal tool-only environments."""
    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover - minimal tool-only environments
        return lambda: None
    now = time.monotonic()
    state = {"last_touch": now, "start": now}
    return lambda: touch_activity_if_due(state, label)


@contextlib.contextmanager
def human_wait_window(session_key: str | None = None):
    """Mark the enclosed block as time spent blocked on a human prompt. Wrap ONLY
    code that is genuinely parked waiting for a user's answer (the CLI approval
    prompt, the gateway approval poll loop). The concurrent tool batch deadline
    excludes this time; wrapping anything else re-creates the hang where
    arbitrary wedged code pushes the deadline out forever. Overlapping windows
    for the same session coalesce (pending counter), so two serialized approval
    prompts don't double-count the same wall clock.

    See #79719.
    """
    key = _resolve_key(session_key)
    now = time.monotonic()
    with _human_wait_lock:
        state = _human_wait_state(key)
        if state.pending == 0:
            state.window_started = now
        state.pending += 1
    try:
        yield
    finally:
        now = time.monotonic()
        # Clamp the accrual too: a window that overstayed the ceiling was wedged —
        # record at most the ceiling, not the whole overstay.
        ceiling = human_wait_ceiling()
        with _human_wait_lock:
            state = _human_wait_states.get(key)
            if state is not None:
                state.pending -= 1
                if state.pending == 0:
                    if state.window_started is not None:
                        state.completed_seconds += _clamped_window_seconds(state.window_started, now, ceiling)
                    state.window_started = None


def human_wait_seconds(session_key: str | None = None) -> float:
    """Return total human-wait seconds recorded for the session: completed windows
    plus the currently open one (if any). Monotonically non-decreasing for the
    life of the process — except when an idle session's entry is evicted under
    cap pressure, which can only shrink a consumer's baseline delta to zero (the
    safe direction: the deadline fires sooner). Deadline consumers snapshot a
    baseline at batch start and use the delta. Each window's contribution is
    clamped to :func:`human_wait_ceiling` (belt-and-braces against the
    wedged-window hang).

    Each window's contribution is clamped to :func:`human_wait_ceiling`: every legitimate human wait
    self-terminates at ``approvals.timeout`` (both the CLI prompt join and the gateway poll loop enforce
    it), so a window that overstays that bound is itself wedged and must not keep extending a batch deadline
    (belt-and-braces for #79719).
    """
    key = _resolve_key(session_key)
    now = time.monotonic()
    # Resolve the clamp outside the lock: it reads the config cache, which must never nest under _human_wait_lock.
    ceiling = human_wait_ceiling()
    with _human_wait_lock:
        state = _human_wait_states.get(key)
        if state is None:
            return 0.0
        total = state.completed_seconds
        if state.window_started is not None:
            total += _clamped_window_seconds(state.window_started, now, ceiling)
        return total
