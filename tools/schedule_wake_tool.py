"""schedule_wake - arm a one-shot self-wake deadline for this session (#122444).

The orchestrator's own alarm: at ``fires_at`` the CLI idle hook injects ``prompt`` as a
plain user turn, so a long delegation/gating session re-enters its loop without human
polling. One-shot only (recurring wake-ups are ``/heartbeat``'s job), refused for
subagent sessions (their session has no idle loop to fire a wake), and only exposed
where a live ``_tui_process_loop`` will actually fire it - a wake armed anywhere else
would be armed-but-dead, the exact failure class #122444 reports.
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import Optional

from tools.registry import registry, tool_error

# Same floor as heartbeat: re-entering more often than once a minute is a busy-loop,
# not a wake. The issue's alarm ladders sleep ~15 min; nothing legitimate is faster.
MIN_WAKE_DELAY_SECONDS = 60


def _has_live_turn_loop() -> bool:
    """True when this process runs the classic-TUI ``_tui_process_loop`` that drains
    ``_pending_input`` (and therefore fires armed wakes). Gateway sessions, the desktop
    session-owner, ACP, cron and -q runs never start it: a wake armed there would sit
    unfired (armed-but-dead)."""
    for thread in threading.enumerate():
        target = getattr(thread, "_target", None)
        if getattr(target, "__name__", "") == "_tui_process_loop":
            return True
    return False


def check_schedule_wake_requirements() -> bool:
    return _has_live_turn_loop()


def _resolve_fires_at(args: dict) -> float:
    delay = args.get("delay_secs")
    at_iso = (args.get("at_iso") or "").strip()
    if delay is None and not at_iso:
        raise ValueError("either delay_secs or at_iso is required")
    if delay is not None and at_iso:
        raise ValueError("pass only one of delay_secs or at_iso")
    if delay is not None:
        return time.time() + float(delay)
    fires_at = datetime.fromisoformat(at_iso).timestamp()
    if fires_at < time.time():
        raise ValueError("at_iso is in the past")
    return fires_at


def schedule_wake_tool(args: dict, session_id: Optional[str] = None) -> str:
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return tool_error("schedule_wake needs a non-empty prompt: the message injected when the wake fires.")
    if str(args.get("recurring", False)).lower() in {"1", "true", "yes"}:
        return tool_error("schedule_wake is one-shot; for recurring wake-ups use /heartbeat.")
    try:
        from agent.delegation_context import is_delegated_child_context

        if is_delegated_child_context():
            return tool_error(
                "schedule_wake arms the CALLING session's idle loop; a subagent session has no loop "
                "to fire it. The orchestrating session must arm its own wake."
            )
    except Exception:
        pass
    try:
        fires_at = _resolve_fires_at(args)
    except Exception as exc:
        return tool_error(f"schedule_wake: {exc}")
    if fires_at - time.time() < MIN_WAKE_DELAY_SECONDS - 1:  # 1s slack for clock reads
        return tool_error(
            f"schedule_wake: delay must be at least {MIN_WAKE_DELAY_SECONDS}s "
            "(faster re-entry than once a minute is a busy-loop, not a wake)."
        )
    sid = session_id or os.environ.get("HERMES_SESSION_ID", "")
    if not sid:
        return tool_error("schedule_wake: no session id in this context; run from a session.")
    try:
        from hermes_cli.wake import schedule_wake

        schedule_wake(sid, prompt, fires_at)
    except Exception as exc:
        return tool_error(f"schedule_wake failed: {exc}")
    return json.dumps(
        {"success": True, "session_id": sid, "fires_at": fires_at, "prompt": prompt},
        ensure_ascii=False,
    )


SCHEDULE_WAKE_SCHEMA = {
    "name": "schedule_wake",
    "description": (
        "Arm a ONE-SHOT self-wake for this session: at the deadline the prompt below is "
        "injected as a user message and the session's loop re-enters with no human input - "
        "use it to keep a long orchestration (delegation batches, gated pipelines) alive "
        "instead of sleeping forever when nothing else will wake you. Requires delay_secs "
        "(seconds from now) or at_iso (ISO timestamp), both >= 60s out; exactly one wake per "
        "session (latest wins); fires once and is consumed. For recurring wake-ups use "
        "/heartbeat. One-shot only: recurring=false is implied."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Message injected as a user turn when the wake fires.",
            },
            "delay_secs": {
                "type": "number",
                "description": "Seconds from now until the wake fires (min 60).",
            },
            "at_iso": {
                "type": "string",
                "description": "ISO timestamp for the wake instead of delay_secs (future only).",
            },
            "recurring": {
                "type": "boolean",
                "description": "Must be false/omitted; schedule_wake is one-shot (use /heartbeat for recurring).",
            },
        },
        "required": ["prompt"],
    },
}


registry.register(
    name="schedule_wake",
    toolset="wake",
    schema=SCHEDULE_WAKE_SCHEMA,
    handler=lambda args, **kw: schedule_wake_tool(args, session_id=kw.get("session_id") or None),
    check_fn=check_schedule_wake_requirements,
)
