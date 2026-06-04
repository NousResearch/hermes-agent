#!/usr/bin/env python3
"""
Reaction-Menu Tool — present_menu

Lets the agent present an interactive menu the user resolves by TAPPING a
reaction, rather than typing.  On Matrix the menu is a message with one seeded
emoji per option; tapping an emoji collapses the menu and injects the chosen
option's ``payload`` as the next user turn.  On platforms without reaction UIs
the base adapter renders a numbered text list and a bare-number reply resolves it.

Unlike ``clarify``, ``present_menu`` is **non-blocking and terminal for the turn**:
it presents the menu and returns immediately, ending the agent's turn.  The user
may tap now or much later — their tap starts a fresh turn through the normal
inbound path (queued behind any busy agent automatically).  Call it LAST.

The actual send + choreography lives in the platform layer (the gateway wires a
``present_menu`` callback in ``gateway/run.py`` that bridges to the adapter's
``send_reaction_menu``).  This module defines the schema, validation, and a thin
dispatcher that delegates to that callback.
"""

import json
from typing import Any, Callable, List, Optional

from tools.reaction_menu_gateway import (
    MAX_OPTIONS,
    MIN_OPTIONS,
    MenuValidationError,
    validate_options,
)


def present_menu_tool(
    prompt: str,
    options: Optional[List[dict]] = None,
    context_id: Optional[str] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Present an interactive reaction menu to the user.

    Args:
        prompt:     Short text shown above the options (what is being chosen).
        options:    1–5 option objects, each ``{emoji, label, payload,
                    terminal?}``.  ``payload`` is injected as the next user turn
                    when that option is tapped; ``terminal`` (default False)
                    suppresses the ♻️ reload reaction for ending choices.
        context_id: Optional caller-defined id echoed back, so the agent can tell
                    which menu a later choice belongs to.
        callback:   Platform-provided sender, injected by the gateway runner.
                    Signature: ``callback(prompt, options, context_id) -> str``.

    Returns:
        JSON string describing the outcome.
    """
    if not prompt or not str(prompt).strip():
        return tool_error("prompt text is required.")
    prompt = str(prompt).strip()

    try:
        normalized = validate_options(options)
    except MenuValidationError as exc:
        return tool_error(str(exc))

    if callback is None:
        return tool_error(
            "present_menu is not available in this execution context "
            "(no reaction-capable platform attached)."
        )

    try:
        outcome = callback(prompt, normalized, context_id)
    except Exception as exc:
        return tool_error(f"failed to present menu: {exc}")

    if not outcome:
        return tool_error("menu could not be delivered.")

    return json.dumps(
        {
            "status": "menu_presented",
            "context_id": context_id,
            "options_offered": [
                {"emoji": o["emoji"], "label": o["label"]} for o in normalized
            ],
            "note": (
                "Menu presented. This turn is complete — the user's tap will "
                "arrive as a new turn. Do not wait or poll for it."
            ),
        },
        ensure_ascii=False,
    )


def check_present_menu_requirements() -> bool:
    """No external requirements — availability is gated by the platform callback."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

PRESENT_MENU_SCHEMA = {
    "name": "present_menu",
    "description": (
        "Present an interactive menu the user resolves by TAPPING a reaction "
        "(no typing). Use when you want to offer a small set of next steps and "
        "let the user pick one with a single tap — e.g. 'read the next passage / "
        "re-read / stop', or picking among a few drafted options.\n\n"
        "This tool is NON-BLOCKING and ENDS YOUR TURN: it presents the menu and "
        "returns immediately. The user may tap now or later; their tap arrives as "
        "a brand-new turn carrying the chosen option's payload. Do NOT wait or "
        "poll for the choice, and call this tool LAST in your turn.\n\n"
        "The menu is posted as its own message DIRECTLY AFTER your reply text, at "
        "the bottom of the chat. Do NOT tell the user where to find it (no 'the "
        "menu is above/below') — just present the choices in your reply.\n\n"
        "Each option has an `emoji` (the tappable reaction), a short `label`, and "
        "a `payload` (the text delivered to you as the next turn when tapped). "
        "Mark an option `terminal: true` when choosing it should end the menu "
        "(no reload offered). Provide 1–5 options with distinct emoji.\n\n"
        "FORMATTING — the `prompt` and option `label`/`emoji` are shown to the user "
        "EXACTLY as written. Write all emoji, dashes, and accented characters as "
        "literal characters (📖, 🌌, —, é), NEVER as escape sequences such as "
        "\\uXXXX — escapes are rendered verbatim and look broken.\n\n"
        "Prefer `clarify` instead when you need a typed/free-form answer or must "
        "block until the user responds."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Text shown above the options, rendered verbatim. Use literal "
                    "emoji/dashes/accents (📖, 🌌, —), never \\uXXXX escapes."
                ),
            },
            "options": {
                "type": "array",
                "minItems": MIN_OPTIONS,
                "maxItems": MAX_OPTIONS,
                "description": "1–5 tappable options with distinct emoji.",
                "items": {
                    "type": "object",
                    "properties": {
                        "emoji": {
                            "type": "string",
                            "description": "The reaction the user taps to pick this option.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Short human-readable choice text.",
                        },
                        "payload": {
                            "type": "string",
                            "description": (
                                "Text delivered to you as the next user turn when "
                                "this option is tapped."
                            ),
                        },
                        "terminal": {
                            "type": "boolean",
                            "description": (
                                "When true, picking this option ends the menu (no "
                                "reload reaction is offered). Default false."
                            ),
                        },
                    },
                    "required": ["emoji", "label", "payload"],
                },
            },
            "context_id": {
                "type": "string",
                "description": (
                    "Optional id echoed back so you can tell which menu a later "
                    "choice belongs to."
                ),
            },
        },
        "required": ["prompt", "options"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="present_menu",
    toolset="reaction_menu",
    schema=PRESENT_MENU_SCHEMA,
    handler=lambda args, **kw: present_menu_tool(
        prompt=args.get("prompt", ""),
        options=args.get("options"),
        context_id=args.get("context_id"),
        callback=kw.get("present_menu_callback") or kw.get("callback"),
    ),
    check_fn=check_present_menu_requirements,
    emoji="🎛️",
)
