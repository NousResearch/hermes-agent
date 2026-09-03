#!/usr/bin/env python3
"""Leave a mark on the page in the Hermes desktop GUI's in-app browser.

``drive_preview`` already draws every move it makes — the field it can reach,
a box round its target, the cursor going there — but those are transients:
each one stands for a single action and retires itself. That is right for
narrating a click and no use at all for holding a finding on screen.

This is the deliberate one. An annotation outlines an element — or, with
``hold``, the entire visible field at once — and stays until the agent takes it
down, so it can show the user what it found, flag the fields
it is about to fill, or keep its place while it works elsewhere on the page.
Named for TouchDesigner's Annotate — the labelled box you drop around part of a
network to call it out.

Annotations are bound to elements, not coordinates: they ride scrolls and
reflows, and they go when their element does, so a navigation clears them
without the agent having to.

Rides the same ``preview.act`` bridge as ``drive_preview`` rather than opening
a second channel — the renderer already resolves ``@e`` refs and owns the
overlay, so this is one more verb on a wire that exists.

Lives in the ``desktop_ui`` toolset, which the GUI gateway enables only for
desktop-sourced sessions.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

ACTIONS = ("add", "hold", "remove", "clear")

# Verbs the renderer knows, keyed by ours. `clear` is `unpin` with nothing to
# aim at, which the overlay reads as "all of them".
WIRE = {"add": "pin", "hold": "hold", "remove": "unpin", "clear": "unpin"}


def annotate_preview_tool(
    action: str = "add",
    ref: Optional[str] = None,
    selector: Optional[str] = None,
    label: Optional[str] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Put one annotation up, take one down, or clear them all."""
    if callback is None:
        return tool_error("annotate_preview is only available in the Hermes desktop app.")

    verb = (action or "add").strip().lower()
    if verb not in ACTIONS:
        return tool_error(f"action must be one of: {', '.join(ACTIONS)}.")

    if verb in ("add", "remove") and not (ref or selector):
        return tool_error(
            f"{verb} needs a ref from drive_preview action='elements' "
            "(e.g. 'btn-sign-in') or a CSS selector."
        )

    payload = {
        name: val
        for name, val in (
            ("action", WIRE[verb]),
            ("ref", None if verb in ("clear", "hold") else ref),
            ("selector", None if verb in ("clear", "hold") else selector),
            ("text", label),
        )
        if val is not None
    }

    try:
        raw = callback(payload)
    except Exception as exc:
        return tool_error(f"Failed to annotate the in-app browser: {exc}")

    if not raw:
        return tool_error(
            "The annotation timed out, or no GUI window answered. "
            "Open a page with open_preview first."
        )

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
