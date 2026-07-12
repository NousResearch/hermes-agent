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
import re
from typing import Any, Callable, List, Optional


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4
_CHOICE_CONTAINER_KEYS = ("choices", "options", "items", "values")


def _flatten_choice(c) -> str:
    """Coerce a single choice into its user-facing display string.

    The schema declares choices as bare strings, but LLMs sometimes emit
    dict-shaped choices like ``[{"description": "..."}]``. A naive ``str(c)``
    turns the whole dict into its Python repr — ``{'description': '...'}`` —
    which then leaks onto every surface that renders the choice (CLI panel,
    Discord buttons, Telegram numbered list) AND is returned verbatim as the
    user's answer. Normalising here, at the one platform-agnostic entry point,
    fixes the whole class in one place instead of per-adapter.

    Dict unwrap order is the canonical LLM tool-call user-facing keys:
    ``label`` → ``description`` → ``text`` → ``title``. ``name`` and ``value``
    are deliberately excluded — they're component-shaped fields that could
    carry raw enum values or short identifiers, not human-readable labels. A
    dict with none of the canonical keys is dropped (returns ""), since a
    garbage label is worse than no choice at all.
    """
    if c is None:
        return ""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, dict):
        for key in ("label", "description", "text", "title"):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(c, (list, tuple)):
        return " ".join(_flatten_choice(x) for x in c).strip()
    return str(c).strip()


def _strip_choice_prefix(value: str) -> str:
    """Drop lightweight numbering/bullets from loose text choices."""
    return re.sub(r"^\s*(?:[-*]|\d+[\).\:-])\s*", "", value).strip()


def _coerce_choices_string(raw: str) -> List[str]:
    """Accept numbered lines / CSV-ish strings from weak tool-call emitters."""
    stripped = raw.strip()
    if not stripped:
        return []

    numbered_lines = [
        _strip_choice_prefix(line)
        for line in stripped.splitlines()
        if line.strip()
    ]
    numbered_lines = [line for line in numbered_lines if line]
    if len(numbered_lines) > 1:
        return numbered_lines

    if any(sep in stripped for sep in (",", ";", "|")):
        parts = re.split(r"\s*[,;|]\s*", stripped)
        return [_strip_choice_prefix(part) for part in parts if _strip_choice_prefix(part)]

    cleaned = _strip_choice_prefix(stripped)
    return [cleaned] if cleaned else []


def _normalize_choices(choices: Any) -> tuple[bool, Optional[List[str]]]:
    """Coerce common loose tool-call payloads into list[str]."""
    candidate: Any = choices

    if isinstance(candidate, str):
        stripped = candidate.strip()
        if not stripped:
            return True, None
        if stripped[:1] in "[{":
            try:
                candidate = json.loads(stripped)
            except Exception:
                candidate = stripped
        if isinstance(candidate, str):
            candidate = _coerce_choices_string(candidate)

    if isinstance(candidate, dict):
        for key in _CHOICE_CONTAINER_KEYS:
            value = candidate.get(key)
            if value is not None:
                candidate = value
                break
        else:
            candidate = list(candidate.values())

    if isinstance(candidate, (tuple, set)):
        candidate = list(candidate)

    if candidate is None:
        return True, None
    if not isinstance(candidate, list):
        return False, None

    normalized = [s for s in (_flatten_choice(c) for c in candidate) if s]
    if len(normalized) > MAX_CHOICES:
        normalized = normalized[:MAX_CHOICES]
    return True, normalized or None


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
        valid_choices, normalized_choices = _normalize_choices(choices)
        if not valid_choices:
            return tool_error("choices must be a list of strings.")
        # Weak structured-output providers sometimes serialize choices as
        # a JSON string, numbered text block, or {"options": [...]} object.
        # Normalize once here so the existing platform adapters still get the
        # clean list[str] shape they already know how to render.
        choices = normalized_choices

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
        "CRITICAL: when you are offering options, put each option ONLY in the "
        "`choices` array — NEVER enumerate the options inside the `question` "
        "text. The UI renders `choices` as selectable rows; options written "
        "into the question string render as dead prose the user can't pick. "
        "Right: question='Which deployment target?', choices=['staging', "
        "'prod']. Wrong: question='Which target? 1) staging 2) prod', choices=[].\n\n"
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
                    "The question itself, and ONLY the question (e.g. 'Which "
                    "deployment target?'). Do NOT embed the answer options here "
                    "— pass them as separate elements in `choices`."
                ),
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_CHOICES,
                "description": (
                    "REQUIRED whenever you are presenting selectable options: "
                    "each distinct option is its own array element (up to 4). "
                    "The UI renders these as pickable rows and auto-appends an "
                    "'Other (type your answer)' option. Omit this parameter "
                    "entirely ONLY for a genuinely open-ended free-text question."
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
