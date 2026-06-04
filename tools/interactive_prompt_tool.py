#!/usr/bin/env python3
"""
Interactive Prompt Tool Module — Rich Clarify (v2)

Presents structured forms, file-request dialogs, and multi-field decision
prompts to the user.  Each option can either return its value immediately
(action="return") or pop a modal form (action="modal") that collects
additional structured input before returning.

Use the plain ``clarify`` tool for ordinary single-choice or open-ended
questions.  Use ``interactive_prompt`` when you need:

* File uploads from the user
* Multi-field form collection
* Per-option styling or conditional modal popups

The actual user-interaction logic lives in the platform layer.  This module
defines the OpenAI function-calling schema, input validation, and a thin
dispatcher that delegates to a platform-provided callback (injected by the
gateway runner).

The gateway runner wires the callback like so:

1. ``generate_prompt_id()``  → opaque prompt ID
2. ``human_input_gateway.register()``  → pending entry with threading.Event
3. ``get_notify(session_key)``  → adapter bridge callback
4. ``notify(entry)``  → triggers adapter's ``send_human_input()``
5. ``human_input_gateway.wait_for_response()``  → blocks agent thread
6. Return the ``HumanInputResult`` as JSON
"""

import json
from typing import Any, Callable, Dict, List, Optional

from tools.human_input_gateway import HumanInputResult, get_interactive_prompt_timeout
from tools.registry import registry, tool_error


# =============================================================================
# Validation constants
# =============================================================================

MAX_OPTIONS = 25
MAX_LABEL_LEN = 80
MAX_VALUE_LEN = 100
MAX_DESC_LEN = 100
MAX_MODAL_TITLE_LEN = 45
MIN_MODAL_FIELDS = 1
MAX_MODAL_FIELDS = 5
MAX_QUESTION_LEN = 2000  # Discord embed description is 4096; leave room for framing


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

INTERACTIVE_PROMPT_SCHEMA = {
    "name": "interactive_prompt",
    "description": (
        "Present a structured interactive prompt to the user — with buttons "
        "where each option can either return a value "
        "immediately or open a modal form to collect additional fields.\n\n"
        "Use this tool when you need:\n"
        "• Structured forms (text inputs, selects, checkboxes)\n"
        "• Multi-field decision collection\n"
        "• Per-option styling or conditional modal popups\n\n"
        "For ordinary single-choice or open-ended questions, prefer the "
        "``clarify`` tool instead.\n\n"
        "**Per-option action model:**\n"
        "- ``action=return`` — selecting the option immediately returns its "
        "``value`` to the agent.\n"
        "- ``action=modal``  — selecting the option opens a modal dialog "
        "with the fields defined in ``modal.fields``.  The user fills in the "
        "form and submits; both the option ``value`` and the field values are "
        "returned."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question or prompt text to present to the user.",
            },
            "options": {
                "type": "array",
                "minItems": 1,
                "maxItems": 25,
                "description": (
                    "List of option objects the user can choose from (1–25)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "maxLength": 80,
                            "description": "Display label for the option (max 80 chars).",
                        },
                        "value": {
                            "type": "string",
                            "maxLength": 100,
                            "description": "Machine-readable value returned when selected (max 100 chars).",
                        },
                        "description": {
                            "type": "string",
                            "maxLength": 100,
                            "description": "Optional short description shown below the label (max 100 chars).",
                        },
                        "style": {
                            "type": "string",
                            "enum": ["primary", "secondary", "success", "danger"],
                            "description": "Visual style hint for the option button.",
                        },
                        "action": {
                            "type": "string",
                            "enum": ["return", "modal"],
                            "description": (
                                "'return' = immediately return the value; "
                                "'modal' = open a form dialog first."
                            ),
                        },
                        "modal": {
                            "type": "object",
                            "description": "Modal specification — required when action is 'modal'.",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "maxLength": 45,
                                    "description": "Modal title (max 45 chars).",
                                },
                                "fields": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 5,
                                    "description": "Fields to render inside the modal (1–5).",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "key": {
                                                "type": "string",
                                                "description": "Unique field identifier.",
                                            },
                                            "label": {
                                                "type": "string",
                                                "description": "Human-readable field label.",
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "Optional help text below the field.",
                                            },
                                            "type": {
                                                "type": "string",
                                                "enum": [
                                                    "text",
                                                    "select",
                                                    "radio",
                                                    "checkbox",
                                                    "file_upload",
                                                ],
                                                "description": "Input widget type.",
                                            },
                                            "required": {
                                                "type": "boolean",
                                                "description": "Whether the field must be filled (default false).",
                                            },
                                            "placeholder": {
                                                "type": "string",
                                                "description": "Placeholder text for text/select inputs.",
                                            },
                                            "options": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "Choices for select/radio/checkbox fields.",
                                            },
                                            "min_length": {
                                                "type": "integer",
                                                "description": "Minimum character count for text fields.",
                                            },
                                            "max_length": {
                                                "type": "integer",
                                                "description": "Maximum character count for text fields.",
                                            },
                                            "multiline": {
                                                "type": "boolean",
                                                "description": "Allow multi-line text input.",
                                            },
                                            "file_policy": {
                                                "type": "object",
                                                "description": "Constraints for file_upload fields.",
                                                "properties": {
                                                    "allowed_extensions": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                        "description": "Allowed file extensions (e.g. ['.png', '.pdf']).",
                                                    },
                                                    "allowed_mime_types": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                        "description": "Allowed MIME types.",
                                                    },
                                                    "max_files": {
                                                        "type": "integer",
                                                        "description": "Maximum number of files accepted.",
                                                    },
                                                    "max_bytes": {
                                                        "type": "integer",
                                                        "description": "Maximum total upload size in bytes.",
                                                    },
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
                    "required": ["label", "value"],
                },
            },
            "display_type": {
                "type": "string",
                "enum": ["buttons"],
                "description": "How to render the options. Currently only 'buttons' is supported.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 60,
                "maximum": 3600,
                "default": 900,
                "description": (
                    "Seconds to wait for the user to respond (60–3600, default 900)."
                ),
            },
            "auth_policy": {
                "type": "string",
                "enum": [
                    "session_owner_only",
                    "any_allowed_user",
                    "any_allowed_role",
                    "any_allowed_user_or_role",
                ],
                "default": "session_owner_only",
                "description": (
                    "Who is allowed to interact with this prompt. Defaults to "
                    "'session_owner_only'."
                ),
            },
        },
        "required": ["question", "options"],
    },
}


# =============================================================================
# Tool implementation
# =============================================================================

def interactive_prompt_tool(
    question: str,
    options: Optional[List[Dict[str, Any]]] = None,
    display_type: str = "buttons",
    timeout_seconds: int = 900,
    auth_policy: str = "session_owner_only",
    callback: Optional[Callable] = None,
) -> str:
    """
    Present a structured interactive prompt to the user.

    Args:
        question:        The question text to present.
        options:         List of option dicts (1–25), each with at least
                         ``label`` and ``value``.  May also include
                         ``description``, ``style``, ``action``, and ``modal``.
        display_type:    Must be \"buttons\".
        timeout_seconds: Seconds to wait for a response (60–3600).
        auth_policy:     Who may interact with the prompt.
        callback:        Platform-provided function that handles the actual UI
                         interaction.  Signature:
                         ``callback(question, options, display_type,
                                    timeout_seconds, auth_policy) -> HumanInputResult | dict``.
                         Injected by the gateway runner.

    Returns:
        JSON string with the user's response.
    """
    # --- question ---
    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()
    if len(question) > MAX_QUESTION_LEN:
        return tool_error(
            f"Question text too long ({len(question)} chars, max {MAX_QUESTION_LEN})."
        )

    # --- options ---
    if not options or not isinstance(options, list):
        return tool_error("options must be a non-empty list of option objects.")

    if len(options) > MAX_OPTIONS:
        return tool_error(f"Too many options ({len(options)}). Maximum is {MAX_OPTIONS}.")

    for idx, opt in enumerate(options):
        if not isinstance(opt, dict):
            return tool_error(f"Option {idx} must be a dict.")

        label = opt.get("label")
        value = opt.get("value")

        if not label or not isinstance(label, str) or not label.strip():
            return tool_error(f"Option {idx} is missing a non-empty 'label'.")
        if not value or not isinstance(value, str) or not value.strip():
            return tool_error(f"Option {idx} is missing a non-empty 'value'.")

        if len(label) > MAX_LABEL_LEN:
            return tool_error(
                f"Option {idx} label exceeds {MAX_LABEL_LEN} characters "
                f"({len(label)})."
            )
        if len(value) > MAX_VALUE_LEN:
            return tool_error(
                f"Option {idx} value exceeds {MAX_VALUE_LEN} characters "
                f"({len(value)})."
            )

        desc = opt.get("description")
        if desc is not None and len(str(desc)) > MAX_DESC_LEN:
            return tool_error(
                f"Option {idx} description exceeds {MAX_DESC_LEN} characters "
                f"({len(str(desc))})."
            )

        # --- modal validation ---
        action = opt.get("action", "return")
        if action == "modal":
            modal = opt.get("modal")
            if not isinstance(modal, dict):
                return tool_error(
                    f"Option {idx} has action='modal' but no valid 'modal' object."
                )
            title = modal.get("title")
            if not title or not isinstance(title, str) or not title.strip():
                return tool_error(
                    f"Option {idx} modal is missing a non-empty 'title'."
                )
            if len(title) > MAX_MODAL_TITLE_LEN:
                return tool_error(
                    f"Option {idx} modal title exceeds {MAX_MODAL_TITLE_LEN} "
                    f"characters ({len(title)})."
                )
            fields = modal.get("fields")
            if not isinstance(fields, list):
                return tool_error(
                    f"Option {idx} modal 'fields' must be a list."
                )
            if len(fields) < MIN_MODAL_FIELDS or len(fields) > MAX_MODAL_FIELDS:
                return tool_error(
                    f"Option {idx} modal must have {MIN_MODAL_FIELDS}–"
                    f"{MAX_MODAL_FIELDS} fields (got {len(fields)})."
                )

            # Per-field validation
            VALID_FIELD_TYPES = {"text", "select", "radio", "checkbox", "file_upload"}
            seen_keys: set = set()
            for fi, fld in enumerate(fields):
                if not isinstance(fld, dict):
                    return tool_error(
                        f"Option {idx} modal field {fi} must be a dict."
                    )
                key = fld.get("key", "")
                if not key or not isinstance(key, str) or not key.strip():
                    return tool_error(
                        f"Option {idx} modal field {fi} is missing a non-empty 'key'."
                    )
                if key in seen_keys:
                    return tool_error(
                        f"Option {idx} modal field {fi} has duplicate key '{key}'."
                    )
                seen_keys.add(key)
                label = fld.get("label")
                if not label or not isinstance(label, str) or not label.strip():
                    return tool_error(
                        f"Option {idx} modal field {fi} is missing a non-empty 'label'."
                    )
                field_type = fld.get("type", "text")
                if field_type not in VALID_FIELD_TYPES:
                    return tool_error(
                        f"Option {idx} modal field {fi} has invalid type "
                        f"'{field_type}'. Must be one of {sorted(VALID_FIELD_TYPES)}."
                    )

                # file_policy validation (only relevant for file_upload fields)
                if field_type == "file_upload":
                    file_policy = fld.get("file_policy")
                    if isinstance(file_policy, dict):
                        max_files = file_policy.get("max_files")
                        if max_files is not None:
                            if not isinstance(max_files, int) or max_files < 1 or max_files > 10:
                                return tool_error(
                                    f"Option {idx} modal field {fi} "
                                    f"file_policy.max_files must be 1–10 "
                                    f"(got {max_files})."
                                )
                        max_bytes = file_policy.get("max_bytes")
                        if max_bytes is not None:
                            if not isinstance(max_bytes, int) or max_bytes <= 0:
                                return tool_error(
                                    f"Option {idx} modal field {fi} "
                                    f"file_policy.max_bytes must be > 0 "
                                    f"(got {max_bytes})."
                                )
                        allowed_extensions = file_policy.get("allowed_extensions")
                        if allowed_extensions is not None:
                            if not isinstance(allowed_extensions, list) or not all(
                                isinstance(ext, str) and ext.startswith(".")
                                for ext in allowed_extensions
                            ):
                                return tool_error(
                                    f"Option {idx} modal field {fi} "
                                    f"file_policy.allowed_extensions must be a "
                                    f"list of strings each starting with '.'."
                                )

    # --- callback guard ---
    if callback is None:
        return json.dumps(
            {"error": "Interactive prompt tool is not available in this execution context."},
            ensure_ascii=False,
        )

    # --- delegate to platform callback ---
    try:
        result = callback(question, options, display_type, timeout_seconds, auth_policy)
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to get user input: {exc}"},
            ensure_ascii=False,
        )

    # Normalise result to a plain dict for JSON serialization
    if isinstance(result, HumanInputResult):
        return json.dumps(result.to_dict(), ensure_ascii=False)

    # Fallback: result is already a dict or simple type
    return json.dumps(result, ensure_ascii=False)


# =============================================================================
# Requirements check
# =============================================================================

def check_interactive_prompt_requirements() -> bool:
    """Check whether interactive_prompt is enabled.

    Controlled by ``agent.interactive_prompt_enabled`` in config.yaml.
    Defaults to ``False`` (opt-in) so the feature ships dormant and can be
    activated per-deployment without code changes.

    When disabled the tool is silently excluded from the schema list — the
    agent never sees it and cannot call it.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        agent_cfg = cfg.get("agent", {}) or {}
        # Explicit opt-in required; absent key = disabled
        return bool(agent_cfg.get("interactive_prompt_enabled", False))
    except Exception:
        return False


# =============================================================================
# Registry
# =============================================================================

registry.register(
    name="interactive_prompt",
    toolset="interactive_prompt",
    schema=INTERACTIVE_PROMPT_SCHEMA,
    handler=lambda args, **kw: interactive_prompt_tool(
        question=args.get("question", ""),
        options=args.get("options"),
        display_type=args.get("display_type", "buttons"),
        timeout_seconds=args.get("timeout_seconds") or get_interactive_prompt_timeout(),
        auth_policy=args.get("auth_policy", "session_owner_only"),
        callback=kw.get("callback"),
    ),
    check_fn=check_interactive_prompt_requirements,
    emoji="🔘",
)
