"""Stop repeated ``process_manage`` wait/poll calls on a silent running process.

Models re-issue byte-identical ``wait``/``poll`` calls on a background process that has printed
nothing new; each repeat is a full model turn. After ``_PROMOTE_AFTER`` consecutive wait/poll results
with no new output, this promotes the session to ``notify_on_complete`` (wiring gateway delivery when
the process was started silently) and tells the model to stop checking: the exit arrives as a message.

Only where a completion can actually reach this agent: one-shot runners (``hermes -z``), cron, Kanban
workers, stateless HTTP sessions, delegated subagents, profiles whose effective
``display.background_process_notifications`` is ``off``, and callers that do not own the process's
completion (another session's process, a subagent's un-handed-off process) keep the old result, since
nothing would start a turn for the notice there. ``terminal.process_wait_auto_notify: false`` restores the old behaviour
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


def _effective_config() -> dict:
    """The profile's effective user config (managed overlay, ``${VAR}`` expansion): the same reader the
    completion consumers use (``gateway.run._load_gateway_config``, ``tui_gateway.server._load_cfg``), so
    an admin-managed or env-expanded value decides promotion exactly as it decides delivery."""
    from hermes_cli.config_effective import load_user_config_effective
    return load_user_config_effective()


def _enabled() -> bool:
    """``terminal.process_wait_auto_notify`` (default true); true when config is unreadable."""
    from hermes_cli.config import cfg_get
    try:
        return cfg_get(_effective_config(), "terminal", "process_wait_auto_notify", default=True) is not False
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

    Finite turns (non-interactive ``hermes chat -q`` / ``-z``, as ``delegate_task`` decides inline execution) end
    the run when the turn ends, so a promised notice would never arrive."""
    from agent.delegation_context import is_delegated_child_context
    from gateway.session_context import async_delivery_supported, get_session_env
    from gateway.session_finite import finite_turn_required
    finite = finite_turn_required()
    if finite is True or (finite is None and get_session_env("HERMES_SINGLE_QUERY_SESSION") == "1"):
        return False
    if not async_delivery_supported() or is_delegated_child_context():
        return False
    # ``off`` drains the completion but never starts the agent turn (CLI, TUI and gateway alike). The
    # gateway also honours the HERMES_BACKGROUND_NOTIFICATIONS override; either one saying off is off.
    from hermes_cli.config import cfg_get
    try:
        from gateway.platforms._shared import platform_gate_env
        modes = (platform_gate_env("HERMES_BACKGROUND_NOTIFICATIONS"),
                 cfg_get(_effective_config(), "display", "background_process_notifications"))
    except Exception:
        logger.debug("background_process_notifications unreadable; assuming on", exc_info=True)
        return True
    return not any(m is False or str(m or "").strip().lower() == "off" for m in modes)


def _caller_receives_completion(session) -> bool:
    """Whether the completion this process would publish is routed to the CALLER, judged by the
    consumers' own predicates: ``ProcessRegistry._owns_event`` (session-key routing; another session's
    process stays with its owner) and ``should_surface_notification`` (a subagent-owned process is
    suppressed unless handed off or ``delegation.surface_child_process_notifications`` is on). Fails
    closed: an unbound caller cannot prove ownership of a keyed process."""
    from gateway.session_context import get_session_env
    from tools.process_registry import ProcessRegistry
    from tools.process_registry_notifications import should_surface_notification
    caller_key = get_session_env("HERMES_SESSION_KEY", "")
    evt = {"type": "completion", "session_id": session.id, "session_key": session.session_key,
           "task_id": session.task_id, "owner_task_id": session.owner_task_id}
    if session.session_key and not caller_key:
        return False
    return ProcessRegistry._owns_event(evt, caller_key, None, False) and should_surface_notification(evt)


def _claim_notify_if_running(registry, session) -> bool:
    """Set ``notify_on_complete`` unless the exit already won; False means it did.

    ``_move_to_finished`` claims the finish under ``registry._lock`` and reads the flag afterwards to
    decide whether to queue the completion, so flipping it under the same lock while the session is
    still in ``_running`` guarantees the exit publication sees it."""
    with registry._lock:
        if session._finish_claimed or session.id not in registry._running:
            return False
        session.notify_on_complete = True
        return True


def _promote(registry, session) -> None:
    """Wire completion delivery for a process just claimed by ``_claim_notify_if_running``.

    CLI, TUI/Desktop and ``hermes chat -q`` drain ``completion_queue``, which the exit path fills
    whenever the flag is set. The messaging gateway delivers only through a per-process watcher, so a
    process started silently also needs the chat routing stamped and a watcher armed."""
    from tools.terminal_tool_background import _register_completion_watcher, _stamp_gateway_routing
    from gateway.session_context import get_session_env
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
    if not _caller_receives_completion(session):
        return result  # the completion would reach its owner, not this caller: keep plain polling
    if not session.notify_on_complete:
        if not _claim_notify_if_running(process_registry, session):
            # Exited between the wait/poll and this check, before the flag was set: no completion
            # was queued, so hand back the exit itself instead of promising a message.
            _forget(session.id)
            return process_registry.poll(session_id) if action == "poll" else process_registry.wait(session_id, timeout=5)
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
