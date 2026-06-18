#!/usr/bin/env python3
"""
Clarify Tool Module - Interactive Clarifying Questions

Allows the agent to present structured multiple-choice questions or open-ended
prompts to the user. In CLI mode, choices are navigable with arrow keys. On
messaging platforms, choices are rendered as a numbered list.

Supports two modes:

1. **Simple** -- provide ``choices`` (up to 4 strings). Auto-appends
   'Other (type your answer)'.
2. **Rich** -- provide ``options`` (up to 25 objects with label, value,
   style, and optional modal forms). The caller controls the full options
   array -- no synthetic 'Other' is appended.

``choices`` and ``options`` are mutually exclusive.

The actual user-interaction logic lives in the platform layer (cli.py for CLI,
gateway/run.py for messaging). This module defines the schema, validation, and
a thin dispatcher that delegates to a platform-provided callback.
"""

import json
from typing import Any, Callable, Dict, List, Optional


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4

# -- Rich-option validation constants ----------------------------------------

# -- Rich-option validation constants ----------------------------------------
MAX_OPTIONS = 25
MAX_LABEL_LEN = 80
MAX_VALUE_LEN = 100
MAX_DESC_LEN = 100
MAX_MODAL_TITLE_LEN = 45
MIN_MODAL_FIELDS = 1
MAX_MODAL_FIELDS = 5
MAX_QUESTION_LEN = 2000

VALID_DISPLAY_TYPES = {"buttons"}
VALID_AUTH_POLICIES = {
    "session_owner_only",
    "any_allowed_user",
    "any_allowed_role",
    "any_allowed_user_or_role",
}
VALID_FIELD_TYPES = {"text", "select", "radio", "checkbox", "file_upload"}
VALID_STYLES = {"primary", "secondary", "success", "danger"}
VALID_ACTIONS = {"return", "modal"}


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
        "or types their own answer via a 5th 'Other' option.\\n"
        "2. **Open-ended** — omit choices entirely. The user types a free-form "
        "response.\\n"
        "2b. **Rich** -- provide `options` (up to 25 objects with label, value, "
        "style, and optional modal forms). The caller controls the full "
        "options array -- no synthetic 'Other' is appended.\\n"
        "`choices` and `options` are mutually exclusive.\\n\\n"
        "CRITICAL: when you are offering options, put each option ONLY in the "
        "`choices` array — NEVER enumerate the options inside the `question` "
        "text. The UI renders `choices` as selectable rows; options written "
        "into the question string render as dead prose the user can't pick. "
        "Right: question='Which deployment target?', choices=['staging', "
        "'prod']. Wrong: question='Which target? 1) staging 2) prod', choices=[].\\n\\n"
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
            "options": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_OPTIONS,
                "description": (
                    "Rich option objects (1-25). Each has at least `label` "
                    "and `value`; may include `description`, `style` "
                    "(primary/secondary/success/danger), `action` "
                    "(return/modal), and `modal` (form spec). Mutually "
                    "exclusive with `choices`."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "label":       {"type": "string", "maxLength": MAX_LABEL_LEN},
                        "value":       {"type": "string", "maxLength": MAX_VALUE_LEN},
                        "description": {"type": "string", "maxLength": MAX_DESC_LEN},
                        "style":       {"type": "string", "enum": sorted(VALID_STYLES)},
                        "action":      {"type": "string", "enum": sorted(VALID_ACTIONS)},
                        "modal": {
                            "type": "object",
                            "properties": {
                                "title":  {"type": "string", "maxLength": MAX_MODAL_TITLE_LEN},
                                "fields": {
                                    "type": "array", "minItems": MIN_MODAL_FIELDS, "maxItems": MAX_MODAL_FIELDS,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "key":         {"type": "string"},
                                            "label":       {"type": "string"},
                                            "description": {"type": "string"},
                                            "type":        {"type": "string", "enum": sorted(VALID_FIELD_TYPES)},
                                            "required":    {"type": "boolean"},
                                            "placeholder": {"type": "string"},
                                            "options":     {"type": "array", "items": {"type": "string"}},
                                            "min_length":  {"type": "integer"},
                                            "max_length":  {"type": "integer"},
                                            "multiline":   {"type": "boolean"},
                                            "file_policy": {
                                                "type": "object",
                                                "properties": {
                                                    "max_files":         {"type": "integer", "minimum": 1, "maximum": 10},
                                                    "min_files":         {"type": "integer", "minimum": 0, "maximum": 10},
                                                },
                                            },
                                        },
                                        "required": ["key", "label", "type"],
                                    },
                                },
                            },
                            "required": ["title", "fields"],
                        },
                    },
                    "required": ["label", "value", "action"],
                },
            },
            "display_type":    {"type": "string", "enum": sorted(VALID_DISPLAY_TYPES), "default": "buttons"},
            "auth_policy":     {"type": "string", "enum": sorted(VALID_AUTH_POLICIES), "default": "session_owner_only"},
            "timeout_seconds": {"type": "integer", "minimum": 60, "maximum": 3600},
        },
        "required": ["question"],
    },
}


# =============================================================================
# Validation helpers (rich path)
# =============================================================================

def _validate_options(options: list) -> Optional[str]:
    """Validate the rich ``options`` array.

    Returns an error message string on failure, or ``None`` if valid.
    """
    if not options or not isinstance(options, list):
        return "options must be a non-empty list of option objects."

    if len(options) > MAX_OPTIONS:
        return f"Too many options ({len(options)}). Maximum is {MAX_OPTIONS}."

    for idx, opt in enumerate(options):
        if not isinstance(opt, dict):
            return f"Option {idx} must be a dict."

        label = opt.get("label")
        value = opt.get("value")

        if not label or not isinstance(label, str) or not label.strip():
            return f"Option {idx} is missing a non-empty 'label'."
        if not value or not isinstance(value, str) or not value.strip():
            return f"Option {idx} is missing a non-empty 'value'."

        if len(label) > MAX_LABEL_LEN:
            return f"Option {idx} label exceeds {MAX_LABEL_LEN} characters ({len(label)})."
        if len(value) > MAX_VALUE_LEN:
            return f"Option {idx} value exceeds {MAX_VALUE_LEN} characters ({len(value)})."

        desc = opt.get("description")
        if desc is not None and len(str(desc)) > MAX_DESC_LEN:
            return f"Option {idx} description exceeds {MAX_DESC_LEN} characters ({len(str(desc))})."

        style = opt.get("style", "secondary")
        if style not in VALID_STYLES:
            return f"Option {idx} has invalid style '{style}'. Must be one of {sorted(VALID_STYLES)}."

        action = opt.get("action", "return")
        if action not in VALID_ACTIONS:
            return f"Option {idx} has invalid action '{action}'. Must be one of {sorted(VALID_ACTIONS)}."

        if action == "modal":
            modal = opt.get("modal")
            if not isinstance(modal, dict):
                return f"Option {idx} has action='modal' but no valid 'modal' object."
            title = modal.get("title")
            if not title or not isinstance(title, str) or not title.strip():
                return f"Option {idx} modal is missing a non-empty 'title'."
            if len(title) > MAX_MODAL_TITLE_LEN:
                return f"Option {idx} modal title exceeds {MAX_MODAL_TITLE_LEN} characters ({len(title)})."
            fields = modal.get("fields")
            if not isinstance(fields, list):
                return f"Option {idx} modal 'fields' must be a list."
            if len(fields) < MIN_MODAL_FIELDS or len(fields) > MAX_MODAL_FIELDS:
                return (
                    f"Option {idx} modal must have {MIN_MODAL_FIELDS}-"
                    f"{MAX_MODAL_FIELDS} fields (got {len(fields)})."
                )

            seen_keys: set = set()
            for fi, fld in enumerate(fields):
                if not isinstance(fld, dict):
                    return f"Option {idx} modal field {fi} must be a dict."
                key = fld.get("key", "")
                if not key or not isinstance(key, str) or not key.strip():
                    return f"Option {idx} modal field {fi} is missing a non-empty 'key'."
                if key in seen_keys:
                    return f"Option {idx} modal field {fi} has duplicate key '{key}'."
                seen_keys.add(key)
                lbl = fld.get("label")
                if not lbl or not isinstance(lbl, str) or not lbl.strip():
                    return f"Option {idx} modal field {fi} is missing a non-empty 'label'."
                field_type = fld.get("type", "text")
                if field_type not in VALID_FIELD_TYPES:
                    return (
                        f"Option {idx} modal field {fi} has invalid type "
                        f"'{field_type}'. Must be one of {sorted(VALID_FIELD_TYPES)}."
                    )

    return None


# =============================================================================
# Tool handler
# =============================================================================

def clarify_tool(
    question: str,
    choices: Optional[List[str]] = None,
    options: Optional[List[Dict[str, Any]]] = None,
    display_type: str = "buttons",
    auth_policy: str = "session_owner_only",
    timeout_seconds: Optional[float] = None,
    callback: Optional[Callable] = None,
) -> str:
    """
    Ask the user a question, optionally with multiple-choice or rich options.

    Args:
        question:        The question text to present.
        choices:         Up to 4 predefined answer choices (simple path).
        options:         Up to 25 rich option objects (rich path).
                         Mutually exclusive with ``choices``.
        display_type:    How to render options (currently only "buttons").
        auth_policy:     Who may interact with the prompt.
        timeout_seconds: Seconds to wait (60-3600). ``None`` uses config default.
        callback:        Platform-provided function that handles the actual UI
                         interaction.

    Returns:
        JSON string with the user's response.
    """
    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()
    if len(question) > MAX_QUESTION_LEN:
        return tool_error(
            f"Question text too long ({len(question)} chars, max {MAX_QUESTION_LEN})."
        )

    # -- Mutual exclusivity --
    if options is not None and choices is not None:
        return tool_error("Use either 'choices' (simple) or 'options' (rich), not both.")

    # -- Validate and trim choices (simple path) --
    if choices is not None:
        if not isinstance(choices, list):
            return tool_error("choices must be a list of strings.")
        choices = [str(c).strip() for c in choices if str(c).strip()]
        if len(choices) > MAX_CHOICES:
            choices = choices[:MAX_CHOICES]
        if not choices:
            choices = None  # empty list -> open-ended

    # -- Validate rich options --
    if options is not None:
        err = _validate_options(options)
        if err:
            return tool_error(err)

    # -- Validate display_type --
    if display_type not in VALID_DISPLAY_TYPES:
        return tool_error(
            f"Unsupported display_type '{display_type}'. "
            f"Must be one of {sorted(VALID_DISPLAY_TYPES)}."
        )

    # -- Validate auth_policy --
    if auth_policy not in VALID_AUTH_POLICIES:
        return tool_error(
            f"Unsupported auth_policy '{auth_policy}'. "
            f"Must be one of {sorted(VALID_AUTH_POLICIES)}."
        )

    # -- Validate / clamp timeout --
    if timeout_seconds is not None:
        if not isinstance(timeout_seconds, (int, float)):
            return tool_error("timeout_seconds must be a number.")
        timeout_seconds = max(60, min(3600, int(timeout_seconds)))

    if callback is None:
        return json.dumps(
            {"error": "Clarify tool is not available in this execution context."},
            ensure_ascii=False,
        )

    try:
        if options is not None:
            # Rich path -- delegate to callback with structured params
            user_response = callback(
                question, choices=None, options=options,
                display_type=display_type, auth_policy=auth_policy,
                timeout_seconds=timeout_seconds,
            )
        else:
            # Simple path -- existing behavior, zero change
            user_response = callback(question, choices)
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to get user input: {exc}"},
            ensure_ascii=False,
        )

    # Rich path: callback returns a JSON string (ClarifyResult.to_dict()).
    # Detect and pass through to avoid double-encoding.
    if options is not None and isinstance(user_response, str):
        try:
            parsed = json.loads(user_response)
            if isinstance(parsed, dict) and "status" in parsed:
                return user_response  # already valid JSON -- pass through
        except (json.JSONDecodeError, ValueError):
            pass

    # Simple path: return in the existing format
    return json.dumps({
        "question": question,
        "choices_offered": choices,
        "user_response": str(user_response).strip(),
    }, ensure_ascii=False)


def check_clarify_requirements() -> bool:
    """Clarify tool has no external requirements -- always available."""
    return True


# =============================================================================
# Registry
# =============================================================================

from tools.registry import registry, tool_error

registry.register(
    name="clarify",
    toolset="clarify",
    schema=CLARIFY_SCHEMA,
    handler=lambda args, **kw: clarify_tool(
        question=args.get("question", ""),
        choices=args.get("choices"),
        options=args.get("options"),
        display_type=args.get("display_type", "buttons"),
        auth_policy=args.get("auth_policy", "session_owner_only"),
        timeout_seconds=args.get("timeout_seconds"),
        callback=kw.get("callback")),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
