#!/usr/bin/env python3
"""
Clarify Tool Module - Interactive Clarifying Questions

Allows the agent to present structured multiple-choice questions or open-ended
prompts to the user. In CLI mode, choices are navigable with arrow keys. On
messaging platforms, choices are rendered as a numbered list.

The actual user-interaction logic lives in the platform layer (cli.py for CLI,
gateway/run.py for messaging). This module defines the schema, validation, and
a thin dispatcher that delegates to a platform-provided callback.

Context enforcement (kanban t_f2e0e531)
----------------------------------------
Per Alex's standing rule and the `asking-the-user-well` skill, every clarify
call MUST carry verified-facts context: at minimum a theme → epic → task →
question chain, ideally with a verified-facts canvas (table, code block, quote)
that grounds each offered choice in evidence the user can see at-a-glance.

The canonical `context` parameter for an attached canvas is spec'd on kanban
card `t_bcacbcb9` but not yet shipped. Until that lands, this module enforces
context inline on the `question` field. Agent-invoked calls (those that come
through the tool registry, where the handler passes `require_context=True`)
that lack a detectable context block are rejected with a structured tool_error
that tells the agent exactly what to add and retry.

Library callers (tests, direct Python use) can call `clarify_tool()` with
`require_context=False` (the default) to bypass enforcement; the agent path
always enforces.
"""

import json
from typing import List, Optional, Callable


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4


# =============================================================================
# Context detection
# =============================================================================

# Markers that indicate the agent included context. Any ONE of these is enough
# — the heuristic is permissive on purpose; the goal is to catch the obvious
# context-free calls ("What do you want?", "Pick one", "Yes or no?"), not to
# police canvas quality.
_CONTEXT_MARKERS = (
    # Chain marker (preferred shape — matches the skill's mandatory format)
    "→",
    # Header-style markers the skill suggests
    "Theme:",
    "Epic:",
    "Task:",
    "Context:",
    "Verified",
    "Background:",
    # Markdown structures that imply embedded artifacts
    "```",      # code fence
    "| ---",    # markdown table separator
    "|---",
    "\n- ",     # bulleted list (multi-line)
    "\n* ",
    "\n> ",     # blockquote
)

# Minimum length below which a question is presumed context-free regardless of
# markers. A "Yes?" with a single `→` doesn't really carry context.
_MIN_CONTEXT_LEN = 120


def _has_context(question: str, choices: Optional[List[str]]) -> tuple[bool, str]:
    """Return (has_context, reason_if_missing).

    A question is considered to carry context if it is long enough AND contains
    at least one structural marker (chain arrow, header label, code/table/list
    structure). Multi-line questions get a bonus — if the question is ≥4 lines,
    we accept it on the assumption that the agent embedded a canvas.

    This is a heuristic. It will accept some weak-canvas calls and may reject
    some genuinely-context-bearing single-line questions. The cost asymmetry
    favours false-accepts: a rejected call costs one retry, an accepted bad
    call corrupts the user's session.
    """
    q = question or ""
    qstrip = q.strip()
    if not qstrip:
        return False, "empty"

    line_count = qstrip.count("\n") + 1

    # Multi-line questions (≥4 lines) — assume canvas embedded.
    if line_count >= 4 and len(qstrip) >= 80:
        return True, ""

    # Short questions — context-free by definition.
    if len(qstrip) < _MIN_CONTEXT_LEN:
        return False, (
            f"question is {len(qstrip)} chars (need ≥{_MIN_CONTEXT_LEN}) and "
            f"only {line_count} line(s); no canvas detected"
        )

    # Long enough — require a structural marker.
    has_marker = any(marker in q for marker in _CONTEXT_MARKERS)
    if has_marker:
        return True, ""

    return False, (
        "question is long enough but lacks any context marker "
        "(chain arrow '→', a 'Theme:/Epic:/Task:/Context:' header, "
        "a code fence, table, blockquote, or bulleted list)"
    )


def _missing_context_error(reason: str) -> str:
    """Build the tool_error payload for a context-free clarify call.

    The error message is the agent's repair instructions — it should be
    actionable enough that the agent can retry the SAME call with the correct
    context block, not start a new chain of thought.
    """
    msg = (
        "Clarify call rejected: missing verified-facts context. "
        f"({reason}.)\n\n"
        "Per Alex's standing rule and the `asking-the-user-well` skill, every "
        "clarify call must show the user the context the decision is being "
        "made in. Until the `context=` canvas field ships (kanban t_bcacbcb9), "
        "embed the context inline in `question`. Required shape:\n\n"
        "  Theme: <theme> → Epic: <epic> → Task: <task> → Question: <one-line ask>\n\n"
        "  <verified-facts slice — table / code block / blockquote / bulleted list>\n\n"
        "  <one-sentence sharper restatement of the decision>\n\n"
        "Then re-issue the clarify call. Do NOT split context and question "
        "across separate messages — the user must see them together. If the "
        "decision isn't actually grounded in verified facts, do not ask; load "
        "the docs first (see `read-before-claiming` and `asking-the-user-well`)."
    )
    return tool_error(msg)


# =============================================================================
# Tool entry point
# =============================================================================


def clarify_tool(
    question: str,
    choices: Optional[List[str]] = None,
    callback: Optional[Callable] = None,
    require_context: bool = False,
) -> str:
    """
    Ask the user a question, optionally with multiple-choice options.

    Args:
        question: The question text to present.
        choices:  Up to 4 predefined answer choices. When omitted the
                  question is purely open-ended.
        callback: Platform-provided function that handles the actual UI
                  interaction. Signature: callback(question, choices) -> str.
                  Injected by the agent runner (cli.py / gateway).
        require_context: When True, reject calls whose `question` does not
                  carry detectable context (theme chain, canvas, code/table).
                  The tool registry handler passes True so every agent call
                  is gated; direct library callers (tests) default to False
                  for backward compatibility.

    Returns:
        JSON string with the user's response, or a tool_error JSON when the
        call is rejected for missing context / bad arguments.
    """
    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()

    # Context enforcement (kanban t_f2e0e531).
    if require_context:
        ok, reason = _has_context(question, choices)
        if not ok:
            return _missing_context_error(reason)

    # Validate and trim choices
    if choices is not None:
        if not isinstance(choices, list):
            return tool_error("choices must be a list of strings.")
        choices = [str(c).strip() for c in choices if str(c).strip()]
        if len(choices) > MAX_CHOICES:
            choices = choices[:MAX_CHOICES]
        if not choices:
            choices = None  # empty list → open-ended

    if callback is None:
        return json.dumps(
            {"error": "Clarify tool is not available in this execution context."},
            ensure_ascii=False,
        )

    try:
        user_response = callback(question, choices)
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to get user input: {exc}"},
            ensure_ascii=False,
        )

    return json.dumps({
        "question": question,
        "choices_offered": choices,
        "user_response": str(user_response).strip(),
    }, ensure_ascii=False)


def check_clarify_requirements() -> bool:
    """Clarify tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

CLARIFY_SCHEMA = {
    "name": "clarify",
    "description": (
        "Ask the user a question when you need clarification, feedback, or a "
        "decision before proceeding. Supports two modes:\n\n"
        "1. **Multiple choice** — provide up to 4 choices. The user picks one "
        "or types their own answer via a 5th 'Other' option.\n"
        "2. **Open-ended** — omit choices entirely. The user types a free-form "
        "response.\n\n"
        "**MANDATORY context**: every `question` MUST embed a verified-facts "
        "context block. Required shape:\n\n"
        "  Theme: <theme> → Epic: <epic> → Task: <task> → Question: <ask>\n\n"
        "  <verified-facts canvas — table / code block / blockquote / "
        "bulleted list — the artifact the decision is about>\n\n"
        "Calls whose `question` is short and lacks a chain arrow, header "
        "label, or canvas structure are rejected with a retry hint. See the "
        "`asking-the-user-well` skill for the canvas pattern.\n\n"
        "Use this tool when:\n"
        "- The task is ambiguous and you need the user to choose an approach\n"
        "- You want post-task feedback ('How did that work out?')\n"
        "- You want to offer to save a skill or update memory\n"
        "- A decision has meaningful trade-offs the user should weigh in on\n\n"
        "Do NOT use this tool for simple yes/no confirmation of dangerous "
        "commands (the terminal tool handles that). Prefer making a reasonable "
        "default choice yourself when the decision is low-stakes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The question to present to the user. MUST embed a "
                    "verified-facts context block (theme → epic → task chain "
                    "plus a canvas slice). Single-line context-free questions "
                    "are rejected; re-issue with context inline."
                ),
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_CHOICES,
                "description": (
                    "Up to 4 answer choices. Omit this parameter entirely to "
                    "ask an open-ended question. When provided, the UI "
                    "automatically appends an 'Other (type your answer)' option."
                ),
            },
        },
        "required": ["question"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="clarify",
    toolset="clarify",
    schema=CLARIFY_SCHEMA,
    handler=lambda args, **kw: clarify_tool(
        question=args.get("question", ""),
        choices=args.get("choices"),
        callback=kw.get("callback"),
        # Agent path always enforces context.
        require_context=True),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
