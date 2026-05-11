#!/usr/bin/env python3
"""
Clarify Tool Module - Interactive Clarifying Questions

Allows the agent to present structured multiple-choice questions or open-ended
prompts to the user. In CLI mode, choices are navigable with arrow keys. On
messaging platforms, choices are rendered as a numbered list.

Supports both single-select (radio) and multi-select (checkbox) modes via the
``multi_select`` parameter.

The actual user-interaction logic lives in the platform layer (cli.py for CLI,
gateway/run.py for messaging). This module defines the schema, validation, and
a thin dispatcher that delegates to a platform-provided callback.
"""

import json
from typing import List, Optional, Callable


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4


def _invoke_callback(callback, question, choices, multi_select):
    """Invoke the platform callback, passing multi_select if supported."""
    try:
        return callback(question, choices, multi_select=multi_select)
    except TypeError:
        # Callback does not accept the multi_select keyword; fall back
        return callback(question, choices)


def _parse_multi_select_response(raw_response) -> List[str]:
    """Parse a multi-select response into a list of cleaned choice strings.

    Handles three forms:
      - Already a list  →  stringify + strip each element
      - JSON array      →  parse and strip
      - Comma-separated →  split, strip, drop empties
    """
    if isinstance(raw_response, list):
        return [str(r).strip() for r in raw_response if str(r).strip()]

    raw = str(raw_response).strip()

    # Try JSON array
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(p).strip() for p in parsed if str(p).strip()]
        except json.JSONDecodeError:
            pass

    # Fall back to comma-separated
    return [s.strip() for s in raw.split(",") if s.strip()]


def clarify_tool(
    question: str,
    choices: Optional[List[str]] = None,
    multi_select: bool = False,
    callback: Optional[Callable] = None,
) -> str:
    """
    Ask the user a question, optionally with multiple-choice options.

    Args:
        question:     The question text to present.
        choices:      Up to 4 predefined answer choices. When omitted the
                      question is purely open-ended.
        multi_select: When True, the user can select multiple choices
                      (checkboxes).  The ``user_response`` in the output JSON
                      will be a list of strings instead of a single string.
                      Has no effect when ``choices`` is omitted.
        callback:     Platform-provided function that handles the actual UI
                      interaction.  Signature:
                      ``callback(question, choices, multi_select=False) -> str``.
                      The optional ``multi_select`` keyword is passed so the
                      platform can render checkboxes instead of radio buttons.
                      Injected by the agent runner (cli.py / gateway).

    Returns:
        JSON string with the user's response.
    """
    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()

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
        raw_response = _invoke_callback(callback, question, choices, multi_select)
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to get user input: {exc}"},
            ensure_ascii=False,
        )

    if multi_select and choices is not None:
        user_response = _parse_multi_select_response(raw_response)
    else:
        user_response = str(raw_response).strip()

    return json.dumps({
        "question": question,
        "choices_offered": choices,
        "user_response": user_response,
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
        "decision before proceeding. Supports three modes:\n\n"
        "1. **Single-select multiple choice** — provide up to 4 choices. The user picks one "
        "or types their own answer via a 5th 'Other' option.\n"
        "2. **Multi-select multiple choice** — set multi_select=true. The user can select "
        "multiple options via checkboxes. user_response will be a list of selected choices.\n"
        "3. **Open-ended** — omit choices entirely. The user types a free-form "
        "response.\n\n"
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
                "description": "The question to present to the user.",
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
            "multi_select": {
                "type": "boolean",
                "description": (
                    "When true, the user can select MULTIPLE options (like checkboxes). "
                    "The user_response will be a list of selected choices. "
                    "When false (default), single selection (radio). "
                    "Has no effect when choices is omitted (open-ended question)."
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
        multi_select=args.get("multi_select", False),
        callback=kw.get("callback")),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
