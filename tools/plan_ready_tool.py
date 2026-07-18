#!/usr/bin/env python3
"""Plan Ready Tool — request approval to leave plan mode and execute.

Exposed ONLY while plan mode is active (the ``plan`` toolset is injected into
``enabled_toolsets`` on entry). The agent calls this after it has written the
plan markdown under ``.hermes/plans/`` to ask the user "approve execution?".

Approval rides the existing clarify machinery: this tool triggers a clarify
with choices ["Approve", "Keep planning"] via the platform-provided callback
(the same ``agent.clarify_callback`` used by the ``clarify`` tool). On
**Approve** the session moves to ``approved`` and the mutating-toolset
restriction lifts on the next turn's agent rebuild. On **Keep planning** the
session stays in ``planning`` and the user's feedback is returned to the agent.

NOTE: ``/plan exit`` (a slash command, not this tool) DISCARDS a pending plan
— only this tool's Approve path ever approves.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

APPROVE_CHOICE = "Approve"
KEEP_PLANNING_CHOICE = "Keep planning"


def plan_ready_tool(
    session_id: str = "",
    plan_path: Optional[str] = None,
    summary: Optional[str] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Request approval to execute the plan.

    Args:
        session_id: The active session id (injected by the runner).
        plan_path:  Path of the plan markdown just written (optional).
        summary:    Short summary of the plan (surfaced in the approval prompt).
        callback:   Platform clarify callback ``callback(question, choices) -> str``
                    (injected by the runner — the same one the clarify tool uses).

    Returns:
        JSON string describing the outcome.
    """
    from hermes_cli.plan_mode import PlanManager

    if not session_id:
        return tool_error("plan_ready is unavailable without a session context.")

    mgr = PlanManager(session_id)
    if not mgr.is_active():
        return tool_error(
            "Plan mode is not active — there is nothing to approve. "
            "plan_ready only applies while planning."
        )

    if plan_path:
        mgr.set_plan_path(plan_path)

    if callback is None:
        return json.dumps(
            {"error": "Approval is not available in this execution context."},
            ensure_ascii=False,
        )

    # Move to pending_approval while the clarify is in flight, then ask.
    mgr.request_approval()
    question = "Plan ready — approve execution?"
    if summary:
        question = f"Plan ready: {summary}\nApprove execution?"

    try:
        answer = callback(question, [APPROVE_CHOICE, KEEP_PLANNING_CHOICE])
    except Exception as exc:
        # Could not collect the decision — stay in planning (fail safe: do NOT
        # approve on error).
        mgr.keep_planning()
        return json.dumps(
            {"error": f"Failed to get approval: {exc}", "status": "planning"},
            ensure_ascii=False,
        )

    answer_text = str(answer or "").strip()
    if answer_text.lower().startswith("approve"):
        mgr.approve()
        return json.dumps(
            {
                "status": "approved",
                "message": (
                    "Plan approved. The mutating-tool restriction lifts on the next "
                    "turn — begin executing the plan."
                ),
            },
            ensure_ascii=False,
        )

    # Anything else — including the "Other" free-text option — is treated as
    # "keep planning", with the user's text surfaced as feedback.
    mgr.keep_planning()
    feedback = "" if answer_text.lower().startswith("keep planning") else answer_text
    return json.dumps(
        {
            "status": "planning",
            "message": "Still planning — approval declined.",
            "feedback": feedback,
        },
        ensure_ascii=False,
    )


def check_plan_ready_requirements() -> bool:
    """No external requirements. Visibility is gated by the ``plan`` toolset
    being injected into ``enabled_toolsets`` only while plan mode is active."""
    return True


PLAN_READY_SCHEMA = {
    "name": "plan_ready",
    "description": (
        "Signal that your plan is complete and request approval to execute it. "
        "Call this ONLY after you have written the plan markdown to a file "
        "under `.hermes/plans/`. It asks the user to Approve or Keep planning. "
        "On approval, mutating tools unlock on the next turn and you may start "
        "implementing; otherwise you stay in plan mode and receive the user's "
        "feedback."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "plan_path": {
                "type": "string",
                "description": "Path of the plan markdown you wrote (under .hermes/plans/).",
            },
            "summary": {
                "type": "string",
                "description": "One-line summary of the plan, shown in the approval prompt.",
            },
        },
        "required": [],
    },
}


registry.register(
    name="plan_ready",
    toolset="plan",
    schema=PLAN_READY_SCHEMA,
    handler=lambda args, **kw: plan_ready_tool(
        session_id=kw.get("session_id", "") or "",
        plan_path=args.get("plan_path"),
        summary=args.get("summary"),
        callback=kw.get("callback"),
    ),
    check_fn=check_plan_ready_requirements,
    emoji="📝",
)
