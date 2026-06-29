#!/usr/bin/env python3
"""
Clarify Tool Module - Interactive Clarifying Questions

Allows the agent to present structured multiple-choice questions, multi-select
questions, or open-ended prompts to the user. In CLI mode, choices are
navigable with arrow keys. On messaging platforms, choices are rendered as
buttons or a numbered fallback list.

The actual user-interaction logic lives in the platform layer (cli.py for CLI,
gateway/run.py for messaging). This module defines the schema, validation, and
a thin dispatcher that delegates to a platform-provided callback.
"""

import json
import re
from typing import Callable, List, Optional


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4


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


def _selected_choices_from_response(response: str, choices: Optional[List[str]]) -> List[str]:
    """Best-effort parse of selected choice labels from a response string.

    ``user_response`` remains the backcompat source of truth. This additive
    structured field helps UI/test consumers understand which offered choices
    were selected. Free-form custom text returns an empty list.
    """
    if not response or not choices:
        return []

    labels = list(choices)
    label_map = {label.lower(): label for label in labels}
    selected: List[str] = []
    tokens = [t.strip() for t in re.split(r"\s*(?:,|\+|;|\n)\s*", response) if t.strip()]
    if not tokens:
        tokens = [response.strip()]

    for token in tokens:
        # Exact labels win before numeric/letter shortcuts so a real choice
        # named "A" is not misread as shortcut A -> first option.
        label = label_map.get(token.lower())
        if label is None and token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(labels):
                label = labels[idx]
        elif label is None and len(token) == 1 and token.isalpha():
            idx = ord(token.upper()) - ord("A")
            if 0 <= idx < len(labels):
                label = labels[idx]
        if label and label not in selected:
            selected.append(label)
    return selected


def clarify_tool(
    question: str,
    choices: Optional[List[str]] = None,
    callback: Optional[Callable] = None,
    multi_select: bool = False,
    min_selections: int = 0,
    max_selections: Optional[int] = None,
    allow_other: bool = True,
) -> str:
    """
    Ask the user a question, optionally with selectable options.

    Args:
        question: The question text to present.
        choices:  Up to 4 predefined answer choices. When omitted the
                  question is purely open-ended.
        callback: Platform-provided function that handles the actual UI
                  interaction. Preferred signature:
                  callback(question, choices, **metadata) -> str.
                  Injected by the agent runner (cli.py / gateway).
        multi_select: Preserve multi-pick UX when several choices can be
                      selected together.
        min_selections: Minimum requested selections for multi-select prompts.
        max_selections: Maximum requested selections for multi-select prompts.
        allow_other: Whether the UI should allow free-form "Other" answers.

    Returns:
        JSON string with the user's response.
    """
    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()

    try:
        min_selections = int(min_selections or 0)
    except (TypeError, ValueError):
        return tool_error("min_selections must be an integer.")
    if max_selections is not None:
        try:
            max_selections = int(max_selections)
        except (TypeError, ValueError):
            return tool_error("max_selections must be an integer or null.")
    if min_selections < 0:
        return tool_error("min_selections must be >= 0.")
    if max_selections is not None and max_selections < 0:
        return tool_error("max_selections must be >= 0 when provided.")
    if max_selections is not None and min_selections > max_selections:
        return tool_error("min_selections cannot be greater than max_selections.")

    # Validate and trim choices
    if choices is not None:
        if not isinstance(choices, list):
            return tool_error("choices must be a list of strings.")
        # LLMs sometimes emit dict-shaped choices (e.g. [{"description": "..."}])
        # instead of bare strings. _flatten_choice unwraps them to their
        # user-facing text here — the single platform-agnostic entry point —
        # so the CLI panel, Discord buttons, and Telegram list all render clean
        # text and the resolved answer is never a raw Python dict repr.
        choices = [s for s in (_flatten_choice(c) for c in choices) if s]
        if len(choices) > MAX_CHOICES:
            choices = choices[:MAX_CHOICES]
        if not choices:
            choices = None  # empty list → open-ended
        elif multi_select:
            if min_selections > len(choices):
                return tool_error("min_selections cannot exceed the number of available choices.")
            if max_selections is not None and max_selections > len(choices):
                return tool_error("max_selections cannot exceed the number of available choices.")

    if callback is None:
        return json.dumps(
            {"error": "Clarify tool is not available in this execution context."},
            ensure_ascii=False,
        )

    try:
        try:
            user_response = callback(
                question,
                choices,
                multi_select=bool(multi_select),
                min_selections=min_selections,
                max_selections=max_selections,
                allow_other=bool(allow_other),
            )
        except TypeError:
            # Backcompat for platform callbacks/plugins that still implement
            # callback(question, choices). Hermes-owned callbacks accept the
            # metadata, but this avoids breaking older integrations.
            user_response = callback(question, choices)
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to get user input: {exc}"},
            ensure_ascii=False,
        )

    user_response_text = str(user_response).strip()
    selected_choices = _selected_choices_from_response(user_response_text, choices)
    if multi_select and choices and selected_choices:
        if len(selected_choices) < min_selections:
            return tool_error(f"Select at least {min_selections} choices.")
        if max_selections is not None and len(selected_choices) > max_selections:
            return tool_error(f"Select at most {max_selections} choices.")

    return json.dumps({
        "question": question,
        "choices_offered": choices,
        "selection_mode": "multi" if multi_select else "single",
        "user_response": user_response_text,
        "selected_choices": selected_choices,
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
        "1. **Single-choice multiple choice** — provide up to 4 choices. The "
        "user picks one or types their own answer via a 5th 'Other' option.\n"
        "2. **Multi-select** — set `multi_select=true` when several choices "
        "may be selected together; the UI should preserve checkbox/multi-pick "
        "semantics where supported.\n"
        "3. **Open-ended** — omit choices entirely. The user types a free-form "
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
        "- A decision has meaningful trade-offs the user should weigh in on\n"
        "- A final report or next-action section asks the user to choose, approve, "
        "modify, defer, forbid, or select multiple follow-up actions. Do not end "
        "such responses with only Markdown lists or fenced ```select blocks; call "
        "this tool instead. Use `multi_select=true` for additive next actions.\n\n"
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
            "multi_select": {
                "type": "boolean",
                "description": (
                    "Set true only when the user may select multiple choices "
                    "together. Leave false for exclusive single-choice decisions."
                ),
            },
            "min_selections": {
                "type": "integer",
                "minimum": 0,
                "description": "Minimum number of selections requested for multi-select prompts.",
            },
            "max_selections": {
                "type": ["integer", "null"],
                "minimum": 0,
                "description": "Maximum number of selections requested for multi-select prompts, or null for no explicit cap.",
            },
            "allow_other": {
                "type": "boolean",
                "description": "Whether the UI should offer a free-form Other answer. Defaults to true.",
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
        multi_select=bool(args.get("multi_select", False)),
        min_selections=args.get("min_selections", 0),
        max_selections=args.get("max_selections"),
        allow_other=bool(args.get("allow_other", True)),
    ),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
