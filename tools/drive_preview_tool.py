#!/usr/bin/env python3
"""Interact with the in-app browser / preview pane in the Hermes desktop GUI.

``open_preview`` shows a page and ``read_preview`` reads it; this tool is the
third leg — clicking, typing, scrolling, and history — so the agent can drive
the same page the user is looking at instead of narrating from the outside.

Elements are addressed by refs from ``action="elements"`` that say what they
are: ``btn-sign-in``, ``inp-email``. A ref lasts as long as the page is open,
including across a re-render that destroys and rebuilds the element, and only a
navigation retires it — the renderer says so rather than acting on whatever now
occupies the spot.

Because the refs hold, the renderer answers with a *delta* — what appeared,
what went, what changed, and what was rebound — instead of re-sending the whole
inventory after every click. That is the cheap half of the arrangement, and it
only works because the refs are legible enough to read on their own three turns
later.

Round-trips through the gateway's blocking-prompt bridge like ``read_preview``:
tui_gateway emits ``preview.act.request``, the renderer injects the interaction
engine into the pane's webview and answers ``preview.act.respond`` with the
outcome plus whatever moved. This module is just schema + a thin dispatcher
over the platform-injected callback.

Lives in the ``desktop_ui`` toolset, which the GUI gateway enables only for
desktop-sourced sessions.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

ACTIONS = (
    "elements",
    "click",
    "hover",
    "type",
    "scroll",
    "press",
    "strobe",
    "back",
    "forward",
    "reload",
)
SCROLL_TO = ("top", "bottom")

# Verbs that need something to act on — a ref from the last inventory, or a
# raw CSS selector. `scroll` is deliberately absent: bare, it scrolls the page.
NEEDS_TARGET = ("click", "hover", "type", "press")


def drive_preview_tool(
    action: str = "",
    ref: Optional[str] = None,
    selector: Optional[str] = None,
    text: Optional[str] = None,
    key: Optional[str] = None,
    submit: Optional[bool] = None,
    amount: Optional[int] = None,
    to: Optional[str] = None,
    limit: Optional[int] = None,
    full: Optional[bool] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Dispatch one interaction to the desktop renderer and return its outcome."""
    if callback is None:
        return tool_error("drive_preview is only available in the Hermes desktop app.")

    verb = (action or "").strip().lower()
    if verb not in ACTIONS:
        return tool_error(f"action must be one of: {', '.join(ACTIONS)}.")

    if verb in NEEDS_TARGET and not (ref or selector):
        return tool_error(
            f"{verb} needs a ref from action='elements' (e.g. 'btn-sign-in') or a CSS selector."
        )

    if verb == "type" and text is None:
        return tool_error("type needs the text to enter.")

    if verb == "press" and not key:
        return tool_error("press needs a key, e.g. 'Enter' or 'Escape'.")

    if to is not None and to not in SCROLL_TO:
        return tool_error(f"to must be one of: {', '.join(SCROLL_TO)}.")

    try:
        payload = {
            name: val
            for name, val in (
                ("action", verb),
                ("ref", ref),
                ("selector", selector),
                ("text", text),
                ("key", key),
                ("submit", submit),
                ("full", full),
                ("to", to),
                ("amount", None if amount is None else int(amount)),
                ("max", None if limit is None else int(limit)),
            )
            if val is not None
        }
    except (TypeError, ValueError):
        return tool_error("amount and max must be integers.")

    try:
        raw = callback(payload)
    except Exception as exc:
        return tool_error(f"Failed to act on the in-app browser: {exc}")

    if not raw:
        return tool_error(
            "The action timed out, or no GUI window answered. "
            "Open a page with open_preview first."
        )

    # The renderer answers with a JSON object; pass it through, else wrap it.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


ACT_PREVIEW_SCHEMA = {
    "name": "drive_preview",
    "description": (
        "Interact with the page open in the in-app browser / preview pane of "
        "the Hermes desktop GUI — the pane open_preview opens beside this "
        "chat. This is how you USE a web app the user is looking at: log in, "
        "fill a form, click through a flow, page a long document. ALWAYS call "
        "action='elements' first to get the current inventory of clickable and "
        "typable things — each carries a ref like 'btn-sign-in' or 'inp-email' "
        "plus its role, label, and value — then act with that ref instead of "
        "guessing a selector. A ref keeps working for as long as the page is "
        "open, INCLUDING across a re-render that rebuilds the element, so hold "
        "onto the ones you were given. "
        "Every action answers with the live url/title plus what moved: the "
        "first look at a page returns the full 'elements' inventory, and after "
        "that you get a 'delta' instead — 'added' entries in full, 'changed' "
        "entries carrying only the ref and whichever of label/value/disabled "
        "actually moved, 'removed' and 'rebound' as bare ref lists, and 'same' "
        "counting the refs that held. A 'rebound' ref needs NO action from you; it "
        "means the page rebuilt that element and your ref already follows it. "
        "Anything not mentioned in a delta is unchanged, so do not re-read the "
        "page to check. Only a navigation invalidates refs; when told they are "
        "stale, call elements again. The mouse "
        "and keyboard are real: the pointer travels to its target and the page "
        "sees genuine input, so hover menus open and hover-only controls work. "
        "Actions: 'elements' (inventory), 'click', 'hover' (move the pointer "
        "onto something and leave it there — use it to open a dropdown or "
        "reveal a tooltip before clicking inside it), 'type' (set a field's "
        "text; submit=true also presses Enter and submits the form), 'scroll' "
        "(the page, or a ref'd scrollable), 'press' (a named key), 'strobe' "
        "(touch nothing — just rattle the highlight through the page again; "
        "'elements' already does this once, so reach for it only when asked to "
        "flick, flash, or bounce around the page some more, and note one call "
        "runs a whole multi-second burst, so never loop it per element), and "
        "'back'/'forward'/'reload' for history. The pane draws every move as "
        "it happens so the user can follow along; those marks fade on their "
        "own, and annotate_preview is how you leave one up on purpose. Use "
        "read_preview when you only "
        "need the page's text, and the browser_* tools when the work belongs "
        "in a separate automated browser rather than the user's own pane."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "description": "What to do. Start with 'elements'.",
            },
            "ref": {
                "type": "string",
                "description": "Element reference from any earlier elements call (e.g. 'btn-sign-in'). Good until the page navigates.",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector, as a fallback when no ref fits. Prefer ref.",
            },
            "text": {"type": "string", "description": "For 'type': the text to enter."},
            "submit": {
                "type": "boolean",
                "description": "For 'type': press Enter and submit the owning form afterwards.",
            },
            "key": {
                "type": "string",
                "description": "For 'press': the key name, e.g. 'Enter', 'Escape', 'ArrowDown'.",
            },
            "amount": {
                "type": "integer",
                "description": "For 'scroll': pixels to scroll (negative scrolls up). Defaults to about one screen.",
            },
            "to": {
                "type": "string",
                "enum": list(SCROLL_TO),
                "description": "For 'scroll': jump to the top or bottom instead of a distance.",
            },
            "max": {
                "type": "integer",
                "description": "For 'elements': cap the inventory. Defaults to the per-call maximum.",
            },
            "full": {
                "type": "boolean",
                "description": "For 'elements': re-read the whole page instead of a delta. Rarely needed.",
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="drive_preview",
    toolset="desktop_ui",
    schema=ACT_PREVIEW_SCHEMA,
    handler=lambda args, **kw: drive_preview_tool(
        action=args.get("action", ""),
        ref=args.get("ref"),
        selector=args.get("selector"),
        text=args.get("text"),
        key=args.get("key"),
        submit=args.get("submit"),
        amount=args.get("amount"),
        to=args.get("to"),
        limit=args.get("max"),
        full=args.get("full"),
        callback=kw.get("callback"),
    ),
    emoji="🖱️",
)
