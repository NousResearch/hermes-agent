"""Blocking gateway approval wait for :mod:`tools.approval`.

Mirrors the CLI's synchronous ``input()`` flow: the agent thread enqueues a
pending approval, the gateway notifies the user, and the thread blocks until
``/approve`` / ``/deny`` resolves it or the approval timeout elapses. Multiple
threads (parallel subagents, execute_code RPC handlers) can block concurrently
— each gets its own ``threading.Event``; ``/approve`` resolves the oldest,
``/approve all`` every pending entry. Queue state (``_gateway_queues``,
``_lock``) is owned by ``tools.approval`` and reached through that module at
call time.
"""

import logging
import threading
import time
import uuid

from tools.interrupt import get_interrupt_reason, is_interrupted
from tools import approval_context as _ctx
from tools.approval_human_wait import activity_heartbeat, human_wait_window

logger = logging.getLogger("tools.approval")


class _ApprovalEntry:
    """One pending dangerous-command approval inside a gateway session."""
    __slots__ = ("event", "data", "result", "reason", "acknowledged", "settle", "cancelled")

    def __init__(self, data: dict):
        self.event = threading.Event()
        self.data = dict(data)
        self.data.setdefault("request_id", uuid.uuid4().hex)
        self.acknowledged = False
        # Surface hook run once when the wait ends by ANY path (answer, timeout, interrupt, /approve from
        # another client): the tui_gateway withdraws its open server→client request through it.
        self.settle = None
        self.result: str | None = None  # "once"|"session"|"always"|"deny"
        # Free-text reason from ``/deny <reason>`` so the agent can adapt, not just hear "denied".
        self.reason: str | None = None
        # Why the prompt was withdrawn with nobody answering (interrupt cause, session teardown);
        # followers and teardown read it so a withdrawn prompt never renders as a user deny.
        self.cancelled: str | None = None


def _poll_event(event: threading.Event, session_key: str, *, interrupt_log: str) -> str:
    """Wait on *event* until it fires, the turn is interrupted, or approvals.timeout
    elapses; returns ``"set"`` | ``"interrupted"`` | ``"timeout"``. Polls in ~1s
    slices so activity heartbeats reach the agent's inactivity tracker every ~10s —
    otherwise the gateway watchdog kills the agent while the user is still
    responding (mirrors ``_wait_for_process()`` cadence). The loop is recorded as
    human-wait time so the concurrent batch deadline excludes it.

    Any interrupt (/stop, inactivity timeout, delegation teardown) ends the wait
    fail-closed: the command does not run. Who caused it is read from the
    per-thread interrupt-cause channel (``get_interrupt_reason()``, a trusted fixed
    category), never inferred from message text, so the caller can report a
    withdrawn prompt without inventing a user refusal."""
    deadline = time.monotonic() + max(_ctx._get_approval_timeout(), 0)
    heartbeat = activity_heartbeat("waiting for user approval")
    with human_wait_window(session_key):
        while True:
            # The poll loop below is verifiably blocked on a human answer (the user tapping approve/deny on
            # the gateway surface), bounded by the approval timeout. Record it as human-wait time so the
            # concurrent batch deadline excludes it (#79719).
            if is_interrupted():
                logger.info(interrupt_log, session_key)
                return "interrupted"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            if event.wait(timeout=min(1.0, remaining)):
                return "set"
            heartbeat()


def _cancel_cause(state: str, entry) -> str | None:
    """Why the wait ended with nobody answering: the turn was interrupted (cause from the
    per-thread channel — a user /stop or a parent's delegation teardown), the entry was
    withdrawn with a stamped cause (leader interrupted, session torn down), or the turn ended
    under the prompt (notifier unregistered, result never set). ``None`` for a real answer
    or a plain timeout."""
    if state == "interrupted":
        return get_interrupt_reason() or "turn interrupted"
    if state == "set" and entry.result is None:
        return entry.cancelled or "the turn ended before the prompt was answered"
    return None


def _finish(payload: dict, resolved: bool, choice: str | None, reason, **extra) -> dict:
    """Fire the post hook and build the decision dict. Unresolved (timeout) and
    a None choice both mean the user never answered; ``cancelled`` carries the
    cause when the prompt was withdrawn rather than answered."""
    if extra.get("cancelled"):
        hook_choice = "cancelled"
    else:
        hook_choice = "timeout" if not resolved else (choice or "timeout")
    _ctx._fire_approval_hook("post_approval_response", **payload, choice=hook_choice, **extra)
    return {"resolved": resolved, "choice": choice, "reason": reason, **extra}


def _await_coalesced_leader(session_key: str, leader, payload: dict):
    """Wait on an already-pending identical approval instead of re-prompting.
    Adopts the leader's decision: ``session``/``always`` → approval (same dict
    shape as a direct resolution; persistence stays the caller's and is
    idempotent across leader and followers); ``deny`` → denial carrying the
    leader's reason; leader timeout / our own deadline → unresolved. ``once``
    returns ``None``: single-use consent covers only the leader's execution,
    so the caller must issue a fresh prompt. Hooks fire with ``coalesced=True``
    so observers see the follower's lifecycle without a duplicate prompt."""
    _ctx._fire_approval_hook("pre_approval_request", **payload, coalesced=True)
    state = _poll_event(leader.event, session_key,
                        interrupt_log="Coalesced approval wait interrupted — "
                                      "returning deny for session %s")
    cancelled = _cancel_cause(state, leader)
    if state == "interrupted":
        # Deny only OUR follower; the leader thread handles its own signal.
        choice, resolved = "deny", True
    elif state == "timeout":
        choice, resolved = None, False
    else:
        choice = leader.result
        resolved = choice is not None
    if choice == "once":
        # The post hook fires for the fresh prompt's own lifecycle, not here.
        return None
    extra = {"cancelled": cancelled} if cancelled else {}
    return _finish(payload, resolved, choice, getattr(leader, "reason", None), coalesced=True, **extra)


def _await_gateway_decision(session_key: str, notify_cb, approval_data: dict, *, surface: str = "gateway") -> dict:
    """Enqueue *approval_data*, notify the user, and block until resolved or timed
    out. Shared by the terminal command guard, the execute_code guard, the plugin
    escalation gate, and MCP elicitation. Returns ``{"resolved", "choice",
    "reason"}`` or ``{"resolved": False, "choice": None, "notify_failed": True}``
    when the notify callback raised. Persisting the choice and building the
    tool-facing result stay with the caller.

    Identical concurrent approvals (same command text + pattern-key set) are
    coalesced: parallel tool calls would otherwise fire N identical prompts
    the user must /approve N times while the agent sits wedged. Followers adopt
    the leader's ``session``/``always``/``deny``/timeout; a ``once`` covers only
    the leader, so the follower falls through to a fresh prompt."""
    from tools import approval as _approval

    primary_key = approval_data.get("pattern_key", "")
    payload = {
        "command": approval_data.get("command", ""),
        "description": approval_data.get("description", ""),
        "pattern_key": primary_key,
        "pattern_keys": list(approval_data.get("pattern_keys", [primary_key])),
        "session_key": session_key, "surface": surface,
    }
    keys = list(approval_data.get("pattern_keys") or [])
    with _approval._lock:
        leader = next((e for e in _approval._gateway_queues.get(session_key, [])
                       if e.data.get("command") == approval_data.get("command")
                       and list(e.data.get("pattern_keys") or []) == keys), None)
    if leader is not None:
        adopted = _await_coalesced_leader(session_key, leader, payload)
        if adopted is not None:
            return adopted

    entry = _ApprovalEntry(approval_data)
    with _approval._lock:
        _approval._gateway_queues.setdefault(session_key, []).append(entry)

    def _drop_entry(reason: str) -> None:
        with _approval._lock:
            queue = _approval._gateway_queues.get(session_key, [])
            if entry in queue:
                queue.remove(entry)
            if not queue:
                _approval._gateway_queues.pop(session_key, None)
            settle, entry.settle = entry.settle, None
        if settle is not None:
            try:
                settle(reason)
            except Exception:
                logger.debug("approval settle hook failed", exc_info=True)

    # Plugins hear about the request before the gateway does (real-time observers).
    _ctx._fire_approval_hook("pre_approval_request", **payload)
    # Bridges sync agent thread → async gateway.
    try:
        notify_cb(dict(entry.data))
    except Exception as exc:
        logger.warning("Gateway approval notify failed: %s", exc)
        _drop_entry("notify_failed")
        _ctx._fire_approval_hook("post_approval_response", **payload, choice="notify_failed")
        return {"resolved": False, "choice": None, "notify_failed": True}

    state = _poll_event(entry.event, session_key,
                        interrupt_log="Approval wait interrupted — returning deny for session %s")
    cancelled = _cancel_cause(state, entry)
    choice = entry.result
    if state == "interrupted":
        # Our own decision stays a fail-closed deny; coalesced followers wake with the
        # cause instead of a deny nobody issued.
        choice, entry.cancelled = "deny", cancelled
        entry.event.set()
    _drop_entry("answered" if state == "set" else state)
    extra = {"cancelled": cancelled} if cancelled else {}
    return _finish(payload, state != "timeout", choice, entry.reason, **extra)
