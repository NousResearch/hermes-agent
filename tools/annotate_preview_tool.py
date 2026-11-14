"""Legacy cleanup for element annotations in the Hermes desktop GUI's Browser.

Agent-owned visible annotations are disabled: page marks are user-controlled
through ``/design-mode``. ``remove`` and ``clear`` remain for cleaning marks left
by older Desktop builds. Same ``preview.act`` bridge as ``drive_preview``.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

ACTIONS = ("add", "hold", "remove", "clear")
BLOCKED_ACTIONS = ("add", "hold")

# Renderer verbs keyed by ours; `clear` is `unpin` with no target = "all of them".
WIRE = {"remove": "unpin", "clear": "unpin"}


def annotate_preview_tool(
    action: str = "clear", ref: Optional[str] = None, selector: Optional[str] = None,
    label: Optional[str] = None, callback: Optional[Callable] = None,
) -> str:
    """Remove one legacy annotation or clear all of them."""
    if callback is None:
        return tool_error("annotate_preview is only available in the Hermes desktop app.")
    verb = (action or "clear").strip().lower()
    if verb not in ACTIONS:
        return tool_error(f"action must be one of: {', '.join(ACTIONS)}.")
    if verb in BLOCKED_ACTIONS:
        return tool_error(
            "Agent-owned visible page marks are disabled. Use /design-mode and select the element yourself."
        )
    if verb == "remove" and not (ref or selector):
        return tool_error(f"{verb} needs a ref from drive_preview action='elements' (e.g. 'btn-sign-in') or a CSS selector.")

    targeted = verb != "clear"
    fields = (("action", WIRE[verb]), ("ref", ref if targeted else None), ("selector", selector if targeted else None), ("text", label))
    payload = {name: val for name, val in fields if val is not None}
    try:
        raw = callback(payload)
    except Exception as exc:
        return tool_error(f"Failed to annotate the in-app browser: {exc}")
    if not raw:
        return tool_error(
            "No GUI window answered with a page: no preview tab is open. "
            "Open a page with open_preview first. If the pane IS open, the desktop app "
            "may be older than this backend — its bridge-unavailable error names that case.")
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


ANNOTATE_PREVIEW_SCHEMA = {
    "name": "annotate_preview",
    "description": (
        "Remove legacy preview-page annotations. Visible agent-owned add/hold marks are disabled; "
        "the user selects page elements through /design-mode. remove/clear take one/all old marks down."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "description": "Defaults to 'clear'. add/hold are rejected; use /design-mode for user-owned selection.",
            },
            "ref": {
                "type": "string",
                "description": "Ref from drive_preview elements.",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector fallback. Prefer ref.",
            },
            "label": {
                "type": "string",
                "description": "Optional caption, e.g. 'cheapest'.",
            },
        },
        "required": [],
    },
}


registry.register(
    name="annotate_preview",
    toolset="desktop_ui",
    schema=ANNOTATE_PREVIEW_SCHEMA,
    handler=lambda args, **kw: annotate_preview_tool(
        action=args.get("action", "clear"),
        ref=args.get("ref"),
        selector=args.get("selector"),
        label=args.get("label"),
        callback=kw.get("callback"),
    ),
    emoji="🔖",
)
