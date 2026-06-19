#!/usr/bin/env python3
"""
Clarify Tool Module - Interactive Clarifying Questions

Allows the agent to present structured multiple-choice questions or open-ended
prompts to the user. In CLI mode, choices are navigable with arrow keys. On
messaging platforms, choices are rendered as a numbered list.

The actual user-interaction logic lives in the platform layer (cli.py for CLI,
gateway/run.py for messaging). This module defines the schema, validation, and
a thin dispatcher that delegates to a platform-provided callback.
"""

import json
from typing import List, Optional, Callable


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4

# Keys models commonly use when they pass a choice as a structured object
# instead of a plain string. Ordered by preference for the short "label" half
# and the longer "description" half.
_CHOICE_LABEL_KEYS = ("label", "title", "name", "text", "option", "choice", "id", "value")
_CHOICE_DESC_KEYS = ("description", "detail", "details", "summary", "explanation", "value")


def _coerce_choice(choice) -> str:
    """Render a single choice as a readable string.

    Models frequently ignore the "array of strings" schema and pass choices as
    structured objects (e.g. ``{"choice": "a", "description": "..."}``). Doing a
    naive ``str(choice)`` on those yields a Python dict repr
    (``{'choice': 'a', 'description': '...'}``) that is noise in the CLI and
    gets hard-truncated to an unreadable stub on platforms with short button
    labels (Discord caps button text at 80 chars). Normalize dict-shaped
    choices into a clean ``label: description`` (or whichever half is present)
    so every platform shows the actual option text.
    """
    if isinstance(choice, str):
        return choice.strip()
    if isinstance(choice, dict):
        def _first(keys):
            for key in keys:
                val = choice.get(key)
                if val not in (None, "") and not isinstance(val, (dict, list)):
                    text = str(val).strip()
                    if text:
                        return text
            return ""

        label = _first(_CHOICE_LABEL_KEYS)
        desc = _first(_CHOICE_DESC_KEYS)
        if label and desc and label.lower() != desc.lower():
            return f"{label}: {desc}"
        if label or desc:
            return label or desc
        # Fall back to a JSON object (still better than a Python repr) so the
        # text is at least valid and free of single-quote ``{'k': 'v'}`` noise.
        return json.dumps(choice, ensure_ascii=False)
    return str(choice).strip()


def clarify_tool(
    question: str,
    choices: Optional[List[str]] = None,
    callback: Optional[Callable] = None,
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
        choices = [text for c in choices if (text := _coerce_choice(c))]
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
        callback=kw.get("callback")),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
