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
from typing import Any, Callable, Dict, List, Optional


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4

# Desktop may send the selected option id alongside the legacy string answer.
# New gateways wrap that id in this transport-only token before resolving the
# blocking callback. Old gateways ignore the extra field and keep returning the
# legacy answer, so mixed Desktop/runtime versions remain compatible.
CLARIFY_OPTION_RESPONSE_PREFIX = "__hermes_clarify_option__:"
CLARIFY_CUSTOM_RESPONSE_PREFIX = "__hermes_clarify_custom__:"


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


def _clean_text(value: Any) -> str:
    """Return trimmed schema text without leaking container representations."""
    return value.strip() if isinstance(value, str) else ""


def _normalize_choices(choices: List[Any]) -> List[Dict[str, Any]]:
    """Normalize legacy strings and structured choices into canonical options."""
    options: List[Dict[str, Any]] = []
    used_ids = set()

    for raw in choices:
        label = ""
        description = ""
        option_id = ""
        value = ""

        if isinstance(raw, dict):
            label = next(
                (
                    text
                    for key in ("label", "text", "title")
                    if (text := _clean_text(raw.get(key)))
                ),
                "",
            )
            description = _clean_text(raw.get("description"))
            if not label:
                label, description = description, ""
            option_id = _clean_text(raw.get("id"))
            value = _clean_text(raw.get("value"))
        else:
            label = _flatten_choice(raw)

        if not label:
            continue

        index = len(options) + 1
        fallback_id = f"option-{index}"
        option_id = option_id or fallback_id
        if option_id in used_ids:
            option_id = fallback_id
            suffix = 2
            while option_id in used_ids:
                option_id = f"{fallback_id}-{suffix}"
                suffix += 1
        used_ids.add(option_id)

        options.append({
            "id": option_id,
            "index": index,
            "label": label,
            "description": description,
            "value": value or label,
        })

    return options


def _format_choice(option: Dict[str, Any]) -> str:
    """Render one canonical option for text-only and legacy callback surfaces."""
    label = str(option.get("label") or "").strip()
    description = str(option.get("description") or "").strip()
    if description and description != label:
        return f"{label} — {description}"
    return label


def _format_prompt(question: str, context: str, recommendation: str) -> str:
    """Render a context-rich prompt for surfaces without a dedicated card."""
    parts = [question]
    if context:
        parts.append(f"Context: {context}")
    if recommendation:
        parts.append(f"Recommendation: {recommendation}")
    return "\n\n".join(parts)


def _resolve_selected_option(
    response: str,
    options: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve a callback answer through id/value/label/display/index aliases."""
    if response.startswith(CLARIFY_OPTION_RESPONSE_PREFIX):
        selected_id = response[len(CLARIFY_OPTION_RESPONSE_PREFIX):].strip()
        return next(
            (
                option
                for option in options
                if str(option.get("id") or "").strip() == selected_id
            ),
            None,
        )

    folded = response.casefold()
    numeric_index = int(response) if response.isdigit() else None

    for option in options:
        aliases = {
            str(option.get("id") or "").strip().casefold(),
            str(option.get("value") or "").strip().casefold(),
            str(option.get("label") or "").strip().casefold(),
            _format_choice(option).casefold(),
        }
        if folded in aliases or numeric_index == option.get("index"):
            return option
    return None


def clarify_tool(
    question: str,
    choices: Optional[List[Any]] = None,
    context: str = "",
    recommendation: str = "",
    callback: Optional[Callable] = None,
) -> str:
    """
    Ask the user a question, optionally with multiple-choice options.

    Args:
        question: The question text to present.
        choices: Up to 4 legacy string or structured answer choices.
        context: Decision context shown with the question when provided.
        recommendation: Optional recommendation reason.
        callback: Platform-provided function that handles the actual UI
                  interaction. Signature: callback(question, choices) -> str.
                  Injected by the agent runner (cli.py / gateway).

    Returns:
        JSON string with the user's response.
    """
    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()
    context = _clean_text(context)
    recommendation = _clean_text(recommendation)

    # Validate and trim choices
    options: Optional[List[Dict[str, Any]]] = None
    display_choices: Optional[List[str]] = None
    if choices is not None:
        if not isinstance(choices, list):
            return tool_error("choices must be a list.")
        options = _normalize_choices(choices)[:MAX_CHOICES]
        if options:
            display_choices = [_format_choice(option) for option in options]
        else:
            options = None  # empty list → open-ended

    if callback is None:
        return json.dumps(
            {"error": "Clarify tool is not available in this execution context."},
            ensure_ascii=False,
        )

    try:
        user_response = callback(
            _format_prompt(question, context, recommendation),
            display_choices,
        )
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to get user input: {exc}"},
            ensure_ascii=False,
        )

    raw_response = str(user_response).strip()
    is_custom_response = raw_response.startswith(CLARIFY_CUSTOM_RESPONSE_PREFIX)
    custom_response = (
        raw_response[len(CLARIFY_CUSTOM_RESPONSE_PREFIX):].strip()
        if is_custom_response
        else ""
    )
    selected_option = (
        None
        if is_custom_response
        else _resolve_selected_option(raw_response, options or [])
    )
    if (
        raw_response.startswith(CLARIFY_OPTION_RESPONSE_PREFIX)
        and selected_option is None
    ):
        return json.dumps({
            "question": question,
            "context": context,
            "recommendation": recommendation,
            "choices_offered": display_choices,
            "options": options,
            "selected_option": None,
            "user_response": "",
            "error": "Selected clarify option is no longer available.",
        }, ensure_ascii=False)

    canonical_response = (
        custom_response
        if is_custom_response
        else (
            str(selected_option["value"]).strip()
            if selected_option is not None
            else raw_response
        )
    )

    return json.dumps({
        "question": question,
        "context": context,
        "recommendation": recommendation,
        "choices_offered": display_choices,
        "options": options,
        "selected_option": selected_option,
        "user_response": canonical_response,
        "response_kind": (
            "custom"
            if is_custom_response
            else ("option" if selected_option is not None else "free_text")
        ),
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
            "context": {
                "type": "string",
                "description": (
                    "Decision context the user needs to compare the options. "
                    "Keep it concise and factual."
                ),
            },
            "recommendation": {
                "type": "string",
                "description": (
                    "Optional recommendation and its concrete reason. Omit it "
                    "when there is no justified recommendation."
                ),
            },
            "choices": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Stable internal option id.",
                                },
                                "label": {
                                    "type": "string",
                                    "description": "Short clickable option label.",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Concrete explanation of this option.",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "Canonical answer value returned after selection.",
                                },
                            },
                            "required": ["label", "description"],
                        },
                    ],
                },
                "maxItems": MAX_CHOICES,
                "description": (
                    "REQUIRED whenever you are presenting selectable options: "
                    "each distinct option is its own array element (up to 4). "
                    "Prefer structured objects with a short `label` and a "
                    "specific `description`; legacy string choices remain valid. "
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
        context=args.get("context", ""),
        recommendation=args.get("recommendation", ""),
        callback=kw.get("callback")),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
