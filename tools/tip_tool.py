#!/usr/bin/env python3
"""Point at something in the Hermes desktop GUI and say one line about it.

The quiet sibling of ``tour``. Same durable ``data-tour`` handles, same
discovery call (``tour(action="targets")``) — but no scrim, no spotlight, and no
Next/Prev. Just an accent-lit bubble with an arrow into whatever the tip is
about, which is the right weight for "that button, there" in the middle of a
sentence.

Fire-and-forget, unlike ``tour``: a tip is not a question, so blocking the turn
on a round-trip would stall the reply it belongs to.

Ungated, also like ``tour``. The desktop's Settings → Appearance switch governs
the app's own idle rotation — the half that talks unprompted — not this, which
Hermes raises mid-conversation in answer to something the user said.

Lives in the ``desktop_ui`` toolset, which the GUI gateway enables only for
desktop-sourced sessions.
"""

import json

from tools import desktop_ui
from tools.registry import registry, tool_error

SIDES = ("top", "right", "bottom", "left")


def tip_tool(text: str, selector: str, title: str = "", side: str = "") -> str:
    """Show one tip bubble anchored to ``selector``."""
    text = (text or "").strip()
    selector = (selector or "").strip()

    if not text:
        return tool_error("tip needs text — the one line the bubble says.")

    if not selector:
        return tool_error(
            "tip needs a selector to point at. Call tour(action='targets') to see "
            "what's on screen and prefer a target reporting stable: true."
        )

    if side and side not in SIDES:
        return tool_error(f"side must be one of: {', '.join(SIDES)}.")

    payload = {"selector": selector, "text": text}
    if title:
        payload["title"] = title
    if side:
        payload["side"] = side

    try:
        ok = desktop_ui.emit("tip.show", payload)
    except Exception as exc:
        return tool_error(f"Failed to show the tip: {exc}")
    if not ok:
        return tool_error("tip is only available in the Hermes desktop app.")

    return json.dumps({"success": True, "selector": selector}, ensure_ascii=False)


TIP_SCHEMA = {
    "name": "show_tip",
    "description": (
        "Point at one thing in the desktop UI with a small arrow bubble (no "
        "dimming, no tour chrome) — for when a sentence is clearer with a "
        "finger on its subject. Get selectors from tour(action='targets'), "
        "prefer stable:true, never guess. One tip at a time (new replaces "
        "last); say the same thing in chat too — the bubble is a pointer, "
        "not the message. Sparingly: a bubble every turn stops being read."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The one-sentence bubble text.",
            },
            "selector": {
                "type": "string",
                "description": "Selector from tour targets.",
            },
            "title": {
                "type": "string",
                "description": "Optional heading.",
            },
            "side": {
                "type": "string",
                "enum": list(SIDES),
                "description": "Omit for 'top'; flips at screen edges.",
            },
        },
        "required": ["text", "selector"],
    },
}


registry.register(
    name="show_tip",
    toolset="desktop_ui",
    schema=TIP_SCHEMA,
    handler=lambda args, **kw: tip_tool(
        text=args.get("text", ""),
        selector=args.get("selector", ""),
        title=args.get("title", ""),
        side=args.get("side", ""),
    ),
    emoji="💡",
)
