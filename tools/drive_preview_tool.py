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
from tools.preview_validation import preview_drag_error
from tools.registry import registry, tool_error

ACTIONS = ("elements", "click", "drag", "hover", "type", "scroll", "press", "strobe", "back", "forward", "reload")
SCROLL_TO = ("top", "bottom")

# Verbs that need something to act on — a ref from the last inventory, or a
# raw CSS selector. `scroll` is deliberately absent: bare, it scrolls the page.
NEEDS_TARGET = ("click", "hover", "type", "press")


def _drive_preview_tool(
    action: str = "", ref: Optional[str] = None, selector: Optional[str] = None, text: Optional[str] = None,
    key: Optional[str] = None, submit: Optional[bool] = None, amount: Optional[int] = None,
    to: Optional[str] = None, limit: Optional[int] = None, full: Optional[bool] = None,
    allow_shortcut: Optional[bool] = None,
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
            ("submit", submit), ("full", full), ("to", to), ("allow_shortcut", allow_shortcut),
            ("amount", None if amount is None else int(amount)), ("max", None if limit is None else int(limit)),
        )
    except (TypeError, ValueError):
        return tool_error("amount and max must be integers.")
    try:
        raw = callback({name: val for name, val in fields if val is not None})
    except Exception as exc:
        return tool_error(f"Failed to act on the in-app browser: {exc}")
    if not raw:
        return tool_error(
            "No GUI window answered with a page: no preview tab is open. "
            "Open a page with open_preview first. If the pane IS open, the desktop app "
            "may be older than this backend — its bridge-unavailable error names that case.")
    return passthrough_json(raw)


def drive_preview_tool(action: str = "", callback: Optional[Callable] = None, **fields) -> str:
    """Keyword entry point; keep raw presence/types for the shared dispatcher."""
    if "limit" in fields:
        fields["max"] = fields.pop("limit")
    return dispatch_drive_preview({"action": action, **fields}, callback)


def dispatch_drive_preview(args: dict, callback: Optional[Callable] = None) -> str:
    """Both registry and inline dispatch retain and validate the entire raw payload."""
    error = preview_drag_error(args)
    if error:
        return tool_error(error)
    if isinstance(args.get("action"), str) and args["action"].strip().lower() == "drag":
        if callback is None:
            return tool_error("drive_preview is only available in the Hermes desktop app.")
        try:
            raw = callback({**args, "action": "drag"})
        except Exception as exc:
            return tool_error(f"Failed to act on the in-app browser: {exc}")
        return passthrough_json(raw) if raw else tool_error("No GUI window answered with a page.")
    fields = {k: args[k] for k in (
        "action", "ref", "selector", "text", "key", "submit", "amount", "to", "full", "allow_shortcut"
    ) if k in args}
    return _drive_preview_tool(**fields, limit=args.get("max"), callback=callback)


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
        "hover menus open). drag: exactly one ref or selector, and finite dx/dy "
        "in guest CSS pixels (each within ±2000, not both zero). Native input delivery "
        "does not prove application state changed; file/DataTransfer drops are not supported. "
        "Actions: elements, click, drag, hover (park the "
        "pointer — opens dropdowns before clicking in), type (submit=true "
        "also presses Enter), scroll, press, strobe (visual flourish only — "
        "one call runs a multi-second burst; never loop it), back/forward/"
        "reload. Moves draw live and fade; annotate_preview leaves a lasting "
        "mark. A printable press on body/html is refused unless allow_shortcut "
        "is true. Page text only: desktop_preview action=read. Separate automated "
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
            "dx": {"type": "number", "minimum": -2000, "maximum": 2000, "description": "drag: horizontal CSS-pixel delta."},
            "dy": {"type": "number", "minimum": -2000, "maximum": 2000, "description": "drag: vertical CSS-pixel delta."},
            "text": {"type": "string", "description": "type: the text."},
            "submit": {
                "type": "boolean",
                "description": "type: press Enter + submit the form after.",
            },
            "key": {
                "type": "string",
                "description": "press: key name ('Enter', 'Escape', 'ArrowDown').",
            },
            "allow_shortcut": {
                "type": "boolean",
                "description": "press: allow a printable key on body/html. Off by default.",
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
    handler=lambda args, **kw: dispatch_drive_preview(args, callback=kw.get("callback")),
    emoji="🖱️")
