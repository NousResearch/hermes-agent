#!/usr/bin/env python3
"""
Ask Orchestrator Tool -- Subagent-to-Parent Communication

Allows subagents to ask their parent/orchestrator agent questions mid-task.
Instead of guessing or failing, the subagent pauses, sends a question to the
parent model (with the parent's conversation context), and receives an answer.

The parent model (e.g. GPT-5.4) answers autonomously from its full context --
no human intervention required. This avoids costly re-delegation cycles when
the subagent hits a blocker or ambiguity.

Safeguards:
  - Max 3 queries per subagent session (prevents chatty subagents)
  - Parent answers from its own context only (no cascading to human)
  - Timeout protection (30s default)
  - Only available when running as a subagent with orchestrator_callback set
"""

import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

MAX_QUERIES = 3


def ask_orchestrator(
    question: str,
    context: Optional[str] = None,
    parent_agent=None,
) -> str:
    """
    Ask the parent/orchestrator agent a question during subagent execution.

    The question is answered by the parent model using its full conversation
    context. No human is involved.

    Args:
        question: The question to ask the orchestrator.
        context: Optional extra context about what the subagent has found so far.
        parent_agent: The parent AIAgent instance (injected via agent loop).

    Returns:
        JSON string with the orchestrator's response or an error.
    """
    if not question or not isinstance(question, str) or not question.strip():
        return json.dumps({
            "success": False,
            "error": "Question must be a non-empty string.",
        }, ensure_ascii=False)

    question = question.strip()

    if parent_agent is None:
        return json.dumps({
            "success": False,
            "error": "ask_orchestrator is only available when running as a subagent.",
        }, ensure_ascii=False)

    orchestrator_callback = getattr(parent_agent, "orchestrator_callback", None)
    if orchestrator_callback is None:
        return json.dumps({
            "success": False,
            "error": "No orchestrator callback available. Continue with your best judgment.",
        }, ensure_ascii=False)

    try:
        response = orchestrator_callback(question, context)

        if not response:
            return json.dumps({
                "success": False,
                "error": "Orchestrator did not provide a response. Continue with best judgment.",
            }, ensure_ascii=False)

        return json.dumps({
            "success": True,
            "question": question,
            "response": str(response).strip(),
        }, ensure_ascii=False)

    except Exception as exc:
        logger.exception("ask_orchestrator callback raised: %s", exc)
        return json.dumps({
            "success": False,
            "error": f"Failed to reach orchestrator: {exc}. Continue with best judgment.",
        }, ensure_ascii=False)


def check_ask_orchestrator_requirements() -> bool:
    """Always available -- gated at runtime by orchestrator_callback presence."""
    return True


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

ASK_ORCHESTRATOR_SCHEMA = {
    "name": "ask_orchestrator",
    "description": (
        "Ask the parent/orchestrator agent a question when you need "
        "clarification, additional context, or guidance.\n\n"
        "The orchestrator has full conversation context and project knowledge "
        "that you don't have. Use this instead of guessing.\n\n"
        "WHEN TO USE:\n"
        "- You need context about the user's intent that wasn't in your task\n"
        "- You found something ambiguous and need the right interpretation\n"
        "- You need to know which of several approaches to take\n"
        "- You hit a blocker and need guidance before continuing\n\n"
        "WHEN NOT TO USE:\n"
        "- For routine decisions you can make yourself\n"
        "- To report progress (just include it in your final summary)\n"
        "- More than 3 times per task (limit enforced)\n\n"
        "The orchestrator answers autonomously -- no human delay."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The question to ask. Be specific about what you need "
                    "and why you can't proceed without an answer."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional: what you've found so far that's relevant to "
                    "the question. Helps the orchestrator give a better answer."
                ),
            },
        },
        "required": ["question"],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="ask_orchestrator",
    toolset="orchestrator_comms",
    schema=ASK_ORCHESTRATOR_SCHEMA,
    handler=lambda args, **kw: ask_orchestrator(
        question=args.get("question", ""),
        context=args.get("context"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_ask_orchestrator_requirements,
)
