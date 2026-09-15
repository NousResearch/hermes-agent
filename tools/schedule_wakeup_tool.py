"""``schedule_wakeup`` — the agent defers its OWN continuation to a later time in THIS session.

The in-session counterpart of a one-shot cron job: instead of a fresh isolated session, the
scheduled prompt re-enters the current conversation as a plain user turn when it is idle and the
time has passed (CLI watchdog, TUI/Desktop poller, gateway idle watcher — the same drivers that
fire ``/heartbeat``). State is ``hermes_cli.wakeups``; this module is the tool surface only.

Unavailable where no live session owner could ever wake: cron jobs (the process exits after the
run), delegated subagents (the parent owns the next turn) and Kanban workers.
"""

from __future__ import annotations

import json
import os

from tools.registry import registry, tool_error


def _owner_session_id() -> str:
    """Raw id of the live conversation, or ``""`` when there is no wakeable owner."""
    from gateway.session_context import get_session_env
    from utils import is_truthy_value

    if is_truthy_value(get_session_env("HERMES_CRON_SESSION", "")) or os.environ.get("HERMES_KANBAN_TASK"):
        return ""
    try:
        from agent.delegation_context import is_delegated_child_context
        if is_delegated_child_context():
            return ""
    except Exception:
        pass
    return get_session_env("HERMES_SESSION_ID", "") or ""


def check_wakeup_requirements() -> bool:
    """Only sessions with a live owner process can be woken (same gate as the cronjob tool)."""
    from utils import env_var_enabled

    return bool(
        env_var_enabled("HERMES_INTERACTIVE")
        or env_var_enabled("HERMES_GATEWAY_SESSION")
        or env_var_enabled("HERMES_EXEC_ASK")
    )


def schedule_wakeup_tool(action: str = "schedule", prompt: str = "", delay=None, when=None,
                         reason: str = "", wakeup_id: str = "") -> str:
    from hermes_cli.wakeups import WakeupManager

    session_id = _owner_session_id()
    if not session_id:
        return tool_error(
            "No wakeable session: wakeups need a live interactive session (not a cron job, "
            "subagent or Kanban worker). For a deferred task from here, use cronjob_manage with a "
            "one-shot schedule like 'in 30m'.")
    mgr = WakeupManager(session_id)
    action = (action or "schedule").strip().lower()
    if action == "list":
        pending = mgr.load().pending
        return json.dumps({"pending": [w.summary() for w in pending], "count": len(pending)}, ensure_ascii=False)
    if action == "cancel":
        if not wakeup_id:
            return tool_error("wakeup_id is required for action='cancel'")
        cancelled = mgr.cancel(wakeup_id)
        return json.dumps({"cancelled": cancelled, "id": wakeup_id,
                           "note": None if cancelled else "no such pending wakeup (already fired or cancelled)"})
    if action != "schedule":
        return tool_error(f"unknown action {action!r}; use schedule, list or cancel")
    wakeup, err, note = mgr.schedule(prompt, delay=str(delay) if delay is not None else None,
                                     when=when, reason=reason)
    if err or wakeup is None:
        return tool_error(err or "could not schedule wakeup")
    # The schedule's own clock is the reference: a fresh time.time() drifts a few ms past it and
    # would render a 10s minimum as "9s".
    out = {"scheduled": True, **wakeup.summary(now=wakeup.created_at)}
    if note:
        out["note"] = note
    return json.dumps(out, ensure_ascii=False)


SCHEDULE_WAKEUP_SCHEMA = {
    "name": "schedule_wakeup",
    "description": (
        "Schedule a wakeup for yourself: at the given time this SAME conversation resumes with the "
        "prompt you provide, as if the user had sent it. Use it to defer your own continuation when "
        "you cannot keep the turn open: a build/deploy/CI window finishing later, a slow-changing "
        "thing to re-check, a long-running task to poll after a sensible interval. Do NOT use it for "
        "waits a blocking terminal command covers (raise its timeout instead), for anything within "
        "the next few seconds, or for periodic schedules (that is cronjob_manage / the user's /loop). "
        "Give exactly one of `delay` (30s, 5m, 2h, 1d; bare number = seconds; min 10s, max 7 days) or "
        "`when` (ISO-8601; an offset or Z makes it absolute, otherwise host local time; must be in the "
        "future). A session holds at most 10 pending wakeups. action='list' shows pending ids; "
        "action='cancel' with wakeup_id removes one (cancelling an unknown id is a harmless no-op). "
        "When it fires you receive the prompt marked [Scheduled wakeup] with a note that no user may "
        "be present."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["schedule", "list", "cancel"], "default": "schedule",
                       "description": "schedule (default) a new wakeup, list pending ones, or cancel by wakeup_id."},
            "prompt": {"type": "string",
                       "description": "Required for schedule: the text this session resumes with when the wakeup fires."},
            "delay": {"type": "string",
                      "description": "Relative span from now, e.g. 30s, 5m, 2h, 1d. A bare number is seconds."},
            "when": {"type": "string",
                     "description": "Absolute ISO-8601 date-time, e.g. 2026-09-13T14:30:00Z."},
            "reason": {"type": "string",
                       "description": "Optional short label shown in status lines and the list."},
            "wakeup_id": {"type": "string", "description": "Required for cancel: the id returned at schedule time."},
        },
        "required": [],
    },
}


registry.register(
    name="schedule_wakeup", toolset="wakeup", schema=SCHEDULE_WAKEUP_SCHEMA,
    handler=lambda args, **kw: schedule_wakeup_tool(
        action=args.get("action", "schedule"), prompt=args.get("prompt", ""), delay=args.get("delay"),
        when=args.get("when"), reason=args.get("reason", ""), wakeup_id=args.get("wakeup_id", "")),
    check_fn=check_wakeup_requirements, emoji="⏰",
)
