"""Per-turn auth/reconnect wait budget for MCP tool calls.

Recovery waits are each individually reasonable and collectively awful. One failing tool call can
burn, in sequence: up to 5s waiting for a session to reappear, 10s in the OAuth manager's
``handle_401``, 15s waiting on the reconnect it triggers, and — when that path declines — another
15s on the session-expired reconnect: 45 seconds of pure waiting before the retry RPC even starts,
per call, with nothing capping the total across several failing tools in one turn. From the
model's side that is indistinguishable from a hang.

So the waits share one budget, scoped to the turn. Each wait is clamped to what is left; once the
budget is spent, recovery waits are skipped and the handler returns its "reconnecting, back off"
error immediately. The recovery itself is NOT cancelled — the reconnect is still signalled and the
server task keeps rebuilding its transport in the background. Deliberately not charged: the retry
RPC itself (the tool's own ``tool_timeout``; work, not waiting). Calls outside a tool dispatch
(startup discovery, dashboard probe, CLI) have no turn to protect and are never capped."""

import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager

logger = logging.getLogger("tools.mcp_tool")

_TURN_RECOVERY_BUDGET_SEC = 15.0
_MAX_TRACKED_RECOVERY_TURNS = 64
_turn_recovery_spent: "OrderedDict[str, float]" = OrderedDict()
_budget_lock = threading.Lock()


def _current_recovery_turn_key() -> str:
    """Turn id bound around the active tool dispatch, or "" outside one."""
    from tools import approval_context
    return approval_context.get_current_turn_id()


def _recovery_budget_remaining() -> float:
    """Seconds of auth/reconnect waiting this turn may still spend (uncapped outside a turn)."""
    key = _current_recovery_turn_key()
    if not key:
        return _TURN_RECOVERY_BUDGET_SEC
    with _budget_lock:
        spent = _turn_recovery_spent.get(key, 0.0)
    return max(0.0, _TURN_RECOVERY_BUDGET_SEC - spent)


def _adjust_spent(key: str, delta: float) -> None:
    """Caller holds ``_budget_lock``."""
    _turn_recovery_spent[key] = max(0.0, _turn_recovery_spent.get(key, 0.0) + delta)
    _turn_recovery_spent.move_to_end(key)
    while len(_turn_recovery_spent) > _MAX_TRACKED_RECOVERY_TURNS:
        _turn_recovery_spent.popitem(last=False)


def _charge_recovery_budget(seconds: float) -> None:
    """Charge ``seconds`` of waiting to the active turn's budget."""
    key = _current_recovery_turn_key()
    if not key:
        return
    with _budget_lock:
        _adjust_spent(key, max(0.0, float(seconds)))


@contextmanager
def _recovery_wait(op: str, requested: float):
    """Clamp one recovery wait to the turn's remaining budget. Yields the seconds the caller may
    actually wait — possibly 0.0, which every call site treats as "skip the wait". The grant is
    RESERVED under the lock before the wait starts — tools of one turn run concurrently, and
    waits that all read the same remainder would each get the whole budget — and settled to the
    time actually spent on exit, including on an exception, so a raising wait cannot leak budget
    back to the turn."""
    key = _current_recovery_turn_key()
    requested = max(0.0, float(requested or 0.0))
    if not key:
        allowed = min(requested, _TURN_RECOVERY_BUDGET_SEC)
    else:
        with _budget_lock:
            allowed = min(requested, max(0.0, _TURN_RECOVERY_BUDGET_SEC - _turn_recovery_spent.get(key, 0.0)))
            _adjust_spent(key, allowed)
    if allowed <= 0.0:
        logger.info("MCP recovery budget for this turn is spent (%.0fs); skipping the %s wait and returning "
                    "to the model immediately. The reconnect continues in the background.",
                    _TURN_RECOVERY_BUDGET_SEC, op)
    started = time.monotonic()
    try:
        yield allowed
    finally:
        if key:
            with _budget_lock:
                _adjust_spent(key, (time.monotonic() - started) - allowed)


def _reset_recovery_budget_for_tests() -> None:
    with _budget_lock:
        _turn_recovery_spent.clear()
