#!/usr/bin/env python3
"""
Suggest-Actions Tool — non-blocking tappable follow-up actions.

Where ``clarify`` *blocks* the agent until the user answers an A/B/C question,
``suggest_actions`` is fire-and-forget: the agent attaches a set of suggested
next actions to a message, then finishes its turn. Each action renders as a
platform-native affordance (Telegram inline button, Discord button, or a
numbered markdown list on plain-text platforms). Tapping one starts a *new*
agent turn seeded with that action's payload — the agent does not wait.

This generalizes the interaction pattern behind ``clarify``:

  * clarify           = blocking, single question, ≤4 choices, agent waits.
  * suggest_actions   = non-blocking, N actions, agent moves on; a tap later
                        injects a fresh user turn.

Any turn that would naturally end in "want me to do X or Y?" can instead emit
tappable actions, so meal-rec cards, cron confirmations, PR approvals, and
reminder snoozes all get the same affordance for free. On platforms without
button support the actions degrade to a readable numbered list the user can
answer by number or free text.

The platform-interaction logic lives in the gateway (``gateway/run.py`` wires
the callback; ``gateway/suggested_actions.py`` holds the registry; each
adapter renders/handles the buttons). This module defines the schema,
validation, and a thin dispatcher that delegates to the injected callback.
"""

import json
from typing import Callable, List, Optional

# Maximum number of suggested actions an agent can attach to one message.
# Telegram tolerates more, but a tall stack of buttons is bad UX on mobile;
# keep the ceiling low so the agent curates rather than dumps.
MAX_ACTIONS = 6

# Caps to keep callback_data within Telegram's 64-byte budget and labels
# readable on mobile. Payloads longer than the cap are stored server-side and
# referenced by id, so this only bounds what round-trips inside button data.
MAX_LABEL_LEN = 64
MAX_PAYLOAD_LEN = 512


def _normalize_actions(actions) -> Optional[List[dict]]:
    """Validate and normalize the ``actions`` argument.

    Returns a list of ``{"label", "payload"}`` dicts, or ``None`` when the
    input is unusable. Accepts either bare strings (label doubles as payload)
    or dicts with explicit ``label`` and optional ``payload``.
    """
    if not isinstance(actions, list) or not actions:
        return None

    normalized: List[dict] = []
    for item in actions:
        if isinstance(item, str):
            label = item.strip()
            if not label:
                continue
            normalized.append({
                "label": label[:MAX_LABEL_LEN],
                "payload": label[:MAX_PAYLOAD_LEN],
            })
        elif isinstance(item, dict):
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            payload = str(item.get("payload", label)).strip() or label
            normalized.append({
                "label": label[:MAX_LABEL_LEN],
                "payload": payload[:MAX_PAYLOAD_LEN],
            })
        # anything else is silently skipped
        if len(normalized) >= MAX_ACTIONS:
            break

    return normalized or None


def suggest_actions_tool(
    message: str,
    actions,
    callback: Optional[Callable] = None,
) -> str:
    """Attach tappable follow-up actions to a message and return immediately.

    Args:
        message:  The text shown above the action buttons.
        actions:  Up to :data:`MAX_ACTIONS` actions. Each is either a string
                  (used as both label and injected payload) or a dict with
                  ``label`` and optional ``payload`` (the text injected as a
                  new user turn when tapped).
        callback: Platform-provided sender. Signature
                  ``callback(message, actions) -> dict``. Injected by the
                  agent runner (cli.py / gateway). Returns a small status dict.

    Returns:
        JSON string describing what was attached. This is fire-and-forget:
        the agent's turn ends normally; a later tap starts a new turn.
    """
    if not message or not message.strip():
        return tool_error("message text is required.")
    message = message.strip()

    normalized = _normalize_actions(actions)
    if normalized is None:
        return tool_error(
            "actions must be a non-empty list of strings or "
            "{label, payload} objects."
        )

    if callback is None:
        # No platform interaction context (e.g. batch run). Degrade gracefully:
        # report the actions as text so the caller can still surface them.
        return json.dumps({
            "delivered": False,
            "reason": "no interactive context; actions returned as data",
            "message": message,
            "actions": normalized,
        }, ensure_ascii=False)

    try:
        status = callback(message, normalized) or {}
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to send suggested actions: {exc}"},
            ensure_ascii=False,
        )

    return json.dumps({
        "delivered": bool(status.get("delivered", True)),
        "message": message,
        "actions": normalized,
        "note": (
            "Actions were attached to the message. This turn is complete; "
            "a tap will start a new turn with the action's payload."
        ),
    }, ensure_ascii=False)


def check_suggest_actions_requirements() -> bool:
    """No external requirements — always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

SUGGEST_ACTIONS_SCHEMA = {
    "name": "suggest_actions",
    "description": (
        "Attach tappable follow-up actions to a message, then finish your "
        "turn. Use this instead of ending a reply with a free-text 'want me "
        "to do X or Y?' — the actions render as native buttons (Telegram, "
        "Discord) or a numbered list (SMS, plain platforms), and tapping one "
        "starts a NEW turn seeded with that action's payload.\n\n"
        "This is NON-BLOCKING: unlike `clarify`, you do NOT wait for a "
        "response. Your turn ends normally after calling this. The user may "
        "tap an action now, later, or never.\n\n"
        "Good uses:\n"
        "- Offer concrete next steps ('Book reservation', 'See more options')\n"
        "- Confirm-or-adjust after presenting a result\n"
        "- Snooze/act shortcuts on reminders\n\n"
        "Prefer `clarify` when you genuinely cannot proceed without the "
        "answer (blocking, ≤4 choices). Prefer `suggest_actions` when you've "
        "delivered something complete and are offering optional next moves."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The text shown above the action buttons.",
            },
            "actions": {
                "type": "array",
                "maxItems": MAX_ACTIONS,
                "description": (
                    "Up to 6 suggested actions. Each item is either a short "
                    "string (used as both the button label and the text "
                    "injected when tapped) or an object with 'label' (button "
                    "text) and 'payload' (the instruction injected as a new "
                    "user turn when tapped)."
                ),
                "items": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "payload": {"type": "string"},
                            },
                            "required": ["label"],
                        },
                    ]
                },
            },
        },
        "required": ["message", "actions"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="suggest_actions",
    toolset="clarify",
    schema=SUGGEST_ACTIONS_SCHEMA,
    handler=lambda args, **kw: suggest_actions_tool(
        message=args.get("message", ""),
        actions=args.get("actions"),
        callback=kw.get("suggest_actions_callback")),
    check_fn=check_suggest_actions_requirements,
    emoji="🔘",
)
