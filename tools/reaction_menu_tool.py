"""Present a Matrix menu without waiting for a choice on the agent thread."""

import json
from collections.abc import Callable

from tools.reaction_menu_model import MAX_OPTIONS, MIN_OPTIONS, MenuValidationError, ReactionMenu
from tools.registry import registry, tool_error


def present_menu_tool(
    prompt: object, options: object = None, context_id: object = None,
    callback: Callable[[ReactionMenu], bool] | None = None,
) -> str:
    if callback is None:
        return tool_error("Reaction menus require an active Matrix session with Reaction Menus enabled.")
    try:
        menu = ReactionMenu.from_arguments(prompt, options, context_id)
    except MenuValidationError as exc:
        return tool_error(str(exc))
    try:
        if not callback(menu):
            return tool_error("Menu could not be delivered. Ask the user in your reply.")
    except Exception:
        return tool_error("Menu delivery failed. Ask the user in your reply.")
    return json.dumps({
        "status": "menu_presented", "context_id": menu.context_id,
        "options_offered": [{"emoji": option.emoji, "label": option.label} for option in menu.options],
        "note": "The choice will arrive in a new turn. Finish your reply without waiting or polling.",
    }, ensure_ascii=False)


PRESENT_MENU_SCHEMA = {
    "name": "present_menu",
    "description": (
        "Offer 1 to 5 choices in the current Matrix conversation. The requester can choose by reacting "
        "within five minutes. Returns after delivery without waiting for a choice. "
        "Call this tool last, then finish your reply. A choice starts a new turn with the original "
        "prompt, context_id, label and payload. A new menu replaces the previous menu in this session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "minLength": 1, "maxLength": 500, "description": "Question above the choices."},
            "options": {
                "type": "array", "minItems": MIN_OPTIONS, "maxItems": MAX_OPTIONS,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "emoji": {"type": "string", "minLength": 1, "maxLength": 32, "description": "Distinct reaction for this choice."},
                        "label": {"type": "string", "minLength": 1, "maxLength": 120, "description": "Visible choice text."},
                        "payload": {"type": "string", "minLength": 1, "maxLength": 2000, "description": "Chosen request for the next turn."},
                    },
                    "required": ["emoji", "label", "payload"],
                },
            },
            "context_id": {"type": "string", "minLength": 1, "maxLength": 128, "description": "Optional context echoed with the choice."},
        },
        "required": ["prompt", "options"],
        "additionalProperties": False,
    },
}

registry.register(
    name="present_menu", toolset="reaction_menu", schema=PRESENT_MENU_SCHEMA,
    handler=lambda args, **kw: present_menu_tool(
        prompt=args.get("prompt", ""), options=args.get("options"), context_id=args.get("context_id"),
    ),
    emoji="🎛️",
)
