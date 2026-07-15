"""Completion gate tools — declare_complete, mark_goal_met, cancel_goal.

``declare_complete`` is an agent-loop tool (handled in invoke_tool) because
it needs access to the agent's CompletionGate and conversation messages.

``mark_goal_met`` and ``cancel_goal`` are standard registry tools — they
modify goal state on the gate, which is accessible via the agent reference
passed through tool kwargs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tools.registry import registry

logger = logging.getLogger(__name__)


DECLARE_COMPLETE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "declare_complete",
        "description": (
            "Declare the current autonomous task complete. This triggers a "
            "two-gate verification: (1) all goals must be marked met or cancelled, "
            "(2) an independent judge reviews a compressed execution trace. "
            "If verification fails, you'll get specific feedback on what to fix. "
            "Use this instead of just writing a summary when running autonomously."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["complete", "partial", "blocked"],
                    "description": (
                        "Completion status. 'complete' = fully done and verified. "
                        "'partial' = some goals met, others cancelled. "
                        "'blocked' = cannot proceed (requires human intervention)."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": "Summary of what was accomplished, for the verification trace.",
                },
            },
            "required": ["status", "summary"],
        },
    },
}

MARK_GOAL_MET_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mark_goal_met",
        "description": (
            "Mark a goal as completed with CONCRETE evidence. Evidence MUST be "
            "specific: exit codes, pass counts, file:line references, HTTP status "
            "codes, specific measurable outputs. Do NOT use vague self-assessments "
            "like 'looks good' or 'should work' — the completion gate will reject them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal_id": {
                    "type": "string",
                    "description": "The goal ID to mark as met (e.g., 'goal-1')",
                },
                "evidence": {
                    "type": "string",
                    "description": (
                        "Concrete evidence of completion. Examples: 'npm test exit 0, "
                        "41/41 passed', 'HTTP 200 from health check', "
                        "'wrote 3 files: src/a.py, src/b.py, src/c.py'"
                    ),
                },
            },
            "required": ["goal_id", "evidence"],
        },
    },
}

CANCEL_GOAL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "cancel_goal",
        "description": (
            "Cancel a goal that cannot be completed, with a specific reason. "
            "Use this when a goal is genuinely impossible (blocked by external "
            "factor, requires unavailable data, etc.), not because it's difficult."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal_id": {
                    "type": "string",
                    "description": "The goal ID to cancel (e.g., 'goal-1')",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this goal cannot be completed",
                },
            },
            "required": ["goal_id", "reason"],
        },
    },
}


def _get_gate_from_kwargs(kwargs: dict) -> Any:
    """Resolve the CompletionGate from tool kwargs.

    Tools receive the agent via ``_agent`` kwarg (set by invoke_tool).
    The gate lives on ``agent._completion_gate``.
    """
    agent = kwargs.get("_agent")
    if agent is None:
        return None
    return getattr(agent, "_completion_gate", None)


def _tool_mark_goal_met(goal_id: str, evidence: str, **kwargs) -> str:
    """Mark a goal as completed with evidence."""
    gate = _get_gate_from_kwargs(kwargs)
    if gate is None:
        return json.dumps({"error": "Completion gate not enabled for this session. "
                           "mark_goal_met is only available in autonomous mode."})
    ok = gate.goals.mark_met(goal_id, evidence)
    if ok:
        return json.dumps({"success": True, "goal_id": goal_id,
                           "message": f"Goal {goal_id} marked as met."})
    else:
        return json.dumps({"success": False, "goal_id": goal_id,
                           "error": f"Goal {goal_id} not found or already resolved."})


def _tool_cancel_goal(goal_id: str, reason: str, **kwargs) -> str:
    """Cancel a goal that cannot be completed."""
    gate = _get_gate_from_kwargs(kwargs)
    if gate is None:
        return json.dumps({"error": "Completion gate not enabled for this session. "
                           "cancel_goal is only available in autonomous mode."})
    ok = gate.goals.cancel(goal_id, reason)
    if ok:
        return json.dumps({"success": True, "goal_id": goal_id,
                           "message": f"Goal {goal_id} cancelled: {reason}"})
    else:
        return json.dumps({"success": False, "goal_id": goal_id,
                           "error": f"Goal {goal_id} not found or already resolved."})


def invoke_declare_complete(agent: Any, status: str, summary: str) -> str:
    """Handle declare_complete from the agent loop.

    Called by invoke_tool in agent_runtime_helpers.py. Has access to the
    agent for the gate, messages, and model client (for the judge call).

    Returns JSON with either ``{"error": "..."}`` (gate rejected, loop
    continues) or ``{"output": "..."}`` (gate passed, loop closes).
    """
    gate = getattr(agent, "_completion_gate", None)
    if gate is None:
        return json.dumps({
            "error": "Completion gate not enabled for this session. "
                     "declare_complete is only available in autonomous mode."
        })

    # Access conversation messages for trace building
    messages = getattr(agent, "_session_messages", None)
    if messages is None:
        messages = getattr(agent, "_conversation_messages", [])
    if messages is None:
        messages = []

    result = gate.declare_complete(agent, messages, status=status, summary=summary)
    return json.dumps(result, ensure_ascii=False)


# ── Register mark_goal_met and cancel_goal as standard tools ────────────────

# declare_complete is handled by the agent loop (invoke_tool), but we
# register its schema here so it appears in the tool list and tool discovery.

def _declare_complete_stub(*args, **kwargs) -> str:
    """Stub — real dispatch is in invoke_tool (agent_runtime_helpers.py)."""
    return json.dumps({"error": "declare_complete must be handled by the agent loop"})

registry.register(
    name="declare_complete",
    toolset="completion_gate",
    schema=DECLARE_COMPLETE_SCHEMA,
    handler=_declare_complete_stub,
    description="Declare autonomous task complete (triggers two-gate verification)",
    emoji="✅",
)

registry.register(
    name="mark_goal_met",
    toolset="completion_gate",
    schema=MARK_GOAL_MET_SCHEMA,
    handler=_tool_mark_goal_met,
    description="Mark an autonomous task goal as completed with concrete evidence",
    emoji="🎯",
)

registry.register(
    name="cancel_goal",
    toolset="completion_gate",
    schema=CANCEL_GOAL_SCHEMA,
    handler=_tool_cancel_goal,
    description="Cancel an autonomous task goal that cannot be completed",
    emoji="🚫",
)
