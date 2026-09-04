#!/usr/bin/env python3
"""Interact with the in-app browser / preview pane in the Hermes desktop GUI (click, type, scroll).

Elements are addressed by legible refs from ``action="elements"`` (``btn-sign-in``); a ref survives
re-renders and only a navigation retires it, so the renderer answers with a *delta* instead of
re-sending the inventory. Round-trips through the gateway's blocking-prompt bridge
(``preview.act.request`` / ``preview.act.respond``); this module is schema + a thin dispatcher over
the platform-injected callback. ``desktop_ui`` toolset: desktop-sourced sessions only.
"""

from typing import Callable, Optional

from tools.desktop_ui import passthrough_json
from tools.registry import registry, tool_error

ACTIONS = ("elements", "click", "hover", "type", "scroll", "press", "strobe", "back", "forward", "reload")
SCROLL_TO = ("top", "bottom")

# Verbs that need something to act on — a ref from the last inventory, or a
# raw CSS selector. `scroll` is deliberately absent: bare, it scrolls the page.
NEEDS_TARGET = ("click", "hover", "type", "press")


def drive_preview_tool(
    action: str = "", ref: Optional[str] = None, selector: Optional[str] = None, text: Optional[str] = None,
    key: Optional[str] = None, submit: Optional[bool] = None, amount: Optional[int] = None,
    to: Optional[str] = None, limit: Optional[int] = None, full: Optional[bool] = None,
    callback: Optional[Callable] = None) -> str:
    """Dispatch one interaction to the desktop renderer and return its outcome."""
    if callback is None:
        return tool_error("drive_preview is only available in the Hermes desktop app.")
    verb = (action or "").strip().lower()
    if verb not in ACTIONS:
        return tool_error(f"action must be one of: {', '.join(ACTIONS)}.")
    if verb in NEEDS_TARGET and not (ref or selector):
        return tool_error(f"{verb} needs a ref from action='elements' (e.g. 'btn-sign-in') or a CSS selector.")
    if verb == "type" and text is None:
        return tool_error("type needs the text to enter.")
    if verb == "press" and not key:
        return tool_error("press needs a key, e.g. 'Enter' or 'Escape'.")
    if to is not None and to not in SCROLL_TO:
        return tool_error(f"to must be one of: {', '.join(SCROLL_TO)}.")
    try:
        fields = (
            ("action", verb), ("ref", ref), ("selector", selector), ("text", text), ("key", key),
            ("submit", submit), ("full", full), ("to", to),
            ("amount", None if amount is None else int(amount)), ("max", None if limit is None else int(limit)),
        )
    except (TypeError, ValueError):
        return tool_error("amount and max must be integers.")
    try:
        raw = callback({name: val for name, val in fields if val is not None})
    except Exception as exc:
        return tool_error(f"Failed to act on the in-app browser: {exc}")
    if not raw:
        return tool_error("The action timed out, or no GUI window answered. Open a page with open_preview first.")
    return passthrough_json(raw)


ACT_PREVIEW_SCHEMA = {
    "name": "drive_preview",
    # Response-shape teaching kept only where skipping it wastes calls (delta
    # semantics, rebound refs, strobe's burst): a model that doesn't know them
    # re-reads pages or loops strobe.
    # See #95681.
    "description": (
        "Use the web page open in the desktop preview pane (the one "
        "`desktop_preview` opens): log in, fill forms, click through flows. ALWAYS "
        "start with action='elements' — it inventories clickable/typable "
        "things as refs ('btn-sign-in') with role/label/value; act by ref, "
        "not guessed selectors. Refs survive re-renders and only die on "
        "navigation (you'll be told they're stale — call elements again). "
        "After the first full inventory, actions answer with a DELTA: "
        "'added' in full, 'changed' as ref + moved fields, 'removed'/"
        "'rebound' as ref lists ('rebound' needs nothing from you — the ref "
        "already follows the rebuilt element). Anything unmentioned is "
        "unchanged; do not re-read to check. Input is real (pointer travels, "
        "hover menus open). Actions: elements, click, hover (park the "
        "pointer — opens dropdowns before clicking in), type (submit=true "
        "also presses Enter), scroll, press, strobe (visual flourish only — "
        "one call runs a multi-second burst; never loop it), back/forward/"
        "reload. Moves draw live and fade; annotate_preview leaves a lasting "
        "mark. Page text only: desktop_preview action=read. Separate automated "
        "browser: browser_* tools."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "description": "Start with 'elements'.",
            },
            "ref": {
                "type": "string",
                "description": "Element ref from an earlier elements call.",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector fallback. Prefer ref.",
            },
            "text": {"type": "string", "description": "type: the text."},
            "submit": {
                "type": "boolean",
                "description": "type: press Enter + submit the form after.",
            },
            "key": {
                "type": "string",
                "description": "press: key name ('Enter', 'Escape', 'ArrowDown').",
            },
            "amount": {
                "type": "integer",
                "description": "scroll: pixels (negative = up; default ~one screen).",
            },
            "to": {
                "type": "string",
                "enum": list(SCROLL_TO),
                "description": "scroll: jump to top/bottom instead.",
            },
            "max": {
                "type": "integer",
                "description": "elements: cap the inventory.",
            },
            "full": {
                "type": "boolean",
                "description": "elements: full re-read instead of a delta. Rarely needed.",
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
        action=args.get("action", ""), limit=args.get("max"), callback=kw.get("callback"),
        **{k: args.get(k) for k in ("ref", "selector", "text", "key", "submit", "amount", "to", "full")},
    ),
    emoji="🖱️")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
