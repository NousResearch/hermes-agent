"""Persistent element annotations in the Hermes desktop GUI's in-app browser.

Unlike ``drive_preview``'s self-retiring marks, an annotation outlines an element
(or, with ``hold``, the whole visible field) until removed. Annotations bind to
elements, not coordinates: they ride scrolls and vanish with their element, so
navigation clears them. Same ``preview.act`` bridge as ``drive_preview`` (the
renderer resolves refs and owns the overlay). ``desktop_ui`` toolset only.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

ACTIONS = ("add", "hold", "remove", "clear")

# Renderer verbs keyed by ours; `clear` is `unpin` with no target = "all of them".
WIRE = {"add": "pin", "hold": "hold", "remove": "unpin", "clear": "unpin"}


def annotate_preview_tool(
    action: str = "add", ref: Optional[str] = None, selector: Optional[str] = None,
    label: Optional[str] = None, callback: Optional[Callable] = None,
) -> str:
    """Put one annotation up, take one down, or clear them all."""
    if callback is None:
        return tool_error("annotate_preview is only available in the Hermes desktop app.")
    verb = (action or "add").strip().lower()
    if verb not in ACTIONS:
        return tool_error(f"action must be one of: {', '.join(ACTIONS)}.")
    if verb in ("add", "remove") and not (ref or selector):
        return tool_error(f"{verb} needs a ref from drive_preview action='elements' (e.g. 'btn-sign-in') or a CSS selector.")

    targeted = verb not in ("clear", "hold")
    fields = (("action", WIRE[verb]), ("ref", ref if targeted else None), ("selector", selector if targeted else None), ("text", label))
    payload = {name: val for name, val in fields if val is not None}
    try:
        raw = callback(payload)
    except Exception as exc:
        return tool_error(f"Failed to annotate the in-app browser: {exc}")
    if not raw:
        return tool_error("The annotation timed out, or no GUI window answered. Open a page with open_preview first.")
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


ANNOTATE_PREVIEW_SCHEMA = {
    "name": "annotate_preview",
    "description": (
        "Highlight elements on the preview-pane page, lastingly (drive_preview's own "
        "marks fade; annotations stay until removed) — point at findings, "
        "flag what you're about to change, keep your place. Use the refs "
        "from drive_preview action='elements'. add: outline one element "
        "(optional short label — a word or two, drawn on the page). hold: "
        "freeze the whole visible field, every element outlined and named. "
        "remove/clear: take one/all down. Marks follow their element on "
        "scroll; navigation clears them."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "description": "Defaults to 'add'.",
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
        action=args.get("action", "add"),
        ref=args.get("ref"),
        selector=args.get("selector"),
        label=args.get("label"),
        callback=kw.get("callback"),
    ),
    emoji="🔖",
)
