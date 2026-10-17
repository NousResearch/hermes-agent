"""Stop repeated ``process_manage`` wait/poll calls on a silent running process.

Models re-issue byte-identical ``wait``/``poll`` calls on a background process that has printed
nothing new; each repeat is a full model turn. After ``_PROMOTE_AFTER`` consecutive wait/poll results
with no new output, this promotes the session to ``notify_on_complete`` (wiring gateway delivery when
the process was started silently) and tells the model to stop checking: the exit arrives as a message.

Only where a completion can actually reach this agent: one-shot runners (``hermes -z``), cron, Kanban
workers, stateless HTTP sessions and delegated subagents keep the old result, since nothing would
deliver the notice there. ``terminal.process_wait_auto_notify: false`` restores the old behaviour
everywhere.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_PROMOTE_AFTER = 2
_MAX_TRACKED = 512
# session id -> (total_output_chars at the last wait/poll, consecutive unchanged count)
_state: dict[str, tuple[int, int]] = {}
_state_lock = threading.Lock()

_STOP_NOTE = (
    "The process is still running and has printed nothing new since your last check. Hermes has turned on "
    "notify_on_complete for it: when it exits, a message with its exit code and final output starts a new turn "
    "automatically, even after you end this turn. Do NOT call process_manage wait or poll on {sid} again, and do "
    "not sleep or loop to wait for it. If you have other work, do it now; otherwise end your turn with a one-line "
    "status and report the result when the completion message arrives."
)


def _enabled() -> bool:
    """``terminal.process_wait_auto_notify`` (default true); true when config is unreadable."""
    from tools.process_registry import ProcessRegistry
    try:
        return bool(ProcessRegistry._config_value("terminal", "process_wait_auto_notify", True))
    except Exception:  # a broken config.yaml must not break process_manage
        logger.debug("process_wait_auto_notify unreadable; using default", exc_info=True)
        return True


def _forget(session_id: str) -> None:
    with _state_lock:
        _state.pop(session_id, None)


def _count_unchanged(session) -> int:
    """Record one still-running wait/poll observation; return the consecutive no-new-output count."""
    total = int(getattr(session, "total_output_chars", 0) or 0)
    with _state_lock:
        # Baseline is the spawn (0 chars observed): the first check on a silent process already counts.
        last_total, count = _state.get(session.id, (0, 0))
        count = count + 1 if total == last_total else 0
        _state[session.id] = (total, count)
        while len(_state) > _MAX_TRACKED:
            _state.pop(next(iter(_state)))
    return count


def _can_deliver_completion() -> bool:
    """Whether a completion notice for this session can reach the agent that is waiting.

    Finite turns (``hermes chat -q`` / ``-z``, as ``delegate_task`` decides inline execution) end
    the run when the turn ends, so a promised notice would never arrive."""
    from agent.delegation_context import is_delegated_child_context
    from gateway.session_context import async_delivery_supported, get_session_env
    from gateway.session_finite import finite_turn_required
    finite = finite_turn_required()
    if finite is True or (finite is None and get_session_env("HERMES_SINGLE_QUERY_SESSION") == "1"):
        return False
    return async_delivery_supported() and not is_delegated_child_context()


def _promote(registry, session) -> None:
    """Turn on completion delivery for a process spawned without it, on every surface.

    CLI, TUI/Desktop and ``hermes chat -q`` drain ``completion_queue``, which the exit path fills
    whenever the flag is set. The messaging gateway delivers only through a per-process watcher, so a
    process started silently also needs the chat routing stamped and a watcher armed."""
    from tools.terminal_tool_background import _register_completion_watcher, _stamp_gateway_routing
    from gateway.session_context import get_session_env
    session.notify_on_complete = True
    if not session.watcher_platform:
        _stamp_gateway_routing(session, get_session_env)
    # watcher_interval is set only by _register_completion_watcher: 0 means no watcher yet.
    if session.watcher_platform and not session.watcher_interval:
        _register_completion_watcher(registry, session, session.session_key)
    try:
        registry._write_checkpoint()
    except Exception:
        logger.debug("checkpoint after auto-notify promotion failed", exc_info=True)
    logger.info("process %s: repeated wait/poll with no new output; auto-enabled notify_on_complete", session.id)


def apply_wait_poll_policy(action: str, session_id: str, result: dict) -> dict:
    """Post-process one ``process_manage`` session action result (see module docstring)."""
    if not isinstance(result, dict) or not _enabled():
        return result
    from tools.process_registry import process_registry
    session = process_registry.get(session_id)
    if session is None:
        return result
    if action == "wait":
        running = result.get("status") == "timeout"
    elif action == "poll":
        running = result.get("status") == "running"
    else:
        running = False
    if not running:
        # Exit, any other action on the session, or a released/interrupted wait starts over.
        if action not in ("wait", "poll") or result.get("status") != "interrupted":
            _forget(session.id)
        return result
    if session.exited or _count_unchanged(session) < _PROMOTE_AFTER or not _can_deliver_completion():
        return result
    if not session.notify_on_complete:
        _promote(process_registry, session)
    note = _STOP_NOTE.format(sid=session.id)
    result["notify_on_complete"] = True
    if action == "wait":
        clamp = result.get("timeout_note", "")
        clamp = clamp.split(". Wait window", 1)[0] + ". " if clamp.startswith("Requested wait") else ""
        result["timeout_note"] = clamp + note
    else:
        result["note"] = note
    return result
