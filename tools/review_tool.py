"""review — the agent starts the same independent reviewer the user gets from /review.

Agent-level tool: it needs the live ``AIAgent`` and the turn's messages, so the real executor is
``agent/inline_tool_executors.py``; the registry handler below is the fallback when that intercept is
bypassed and reviews the last persisted history instead. All behaviour lives in
``agent.review_engine`` (reviewer briefing, ``auxiliary.review`` model routing, async completion).
"""

from tools.registry import registry

REVIEW_SCHEMA = {
    "name": "review",
    "description": (
        "Independent review of the work in this conversation (PR, code, docs) by the configured reviewer model — "
        "what the user's /review starts. It is briefed with your recent conversation and loaded skills and "
        "investigates the artifacts itself. Runs in the background; the review re-enters as a new message, so end "
        "your turn after starting it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "description": "What to review and what to look hardest at. Optional.",
            },
        },
        "required": [],
    },
}


def review_tool(focus: str = "", parent_agent=None, messages=None) -> str:
    from agent.review_engine import run_agent_review
    from tools.registry import tool_error

    if parent_agent is None:
        return tool_error("review requires a parent agent context.")
    return run_agent_review(parent_agent, messages, focus or "")


registry.register(
    name="review",
    toolset="review",
    schema=REVIEW_SCHEMA,
    handler=lambda args, **kw: review_tool(focus=args.get("focus") or "", parent_agent=kw.get("parent_agent")),
    emoji="⚖",
)
