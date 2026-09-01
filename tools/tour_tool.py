#!/usr/bin/env python3
"""Run a guided tour (highlight + narrate UI elements) in the Hermes desktop GUI.

One generic tool, no baked-in tour definitions: the agent discovers what is on
screen (``action="targets"``), then highlights any element by CSS selector with
its own title/text — either one step at a time (``show``, agent-paced) or as a
full step list the user pages through with Next/Prev (``start``).

Two surfaces share the same engine (driver.js in the renderer):

- ``surface="app"`` — the Hermes desktop app's own DOM (tours of Hermes itself).
- ``surface="preview"`` — the page loaded in the in-app browser/preview pane
  (tours of ANY web app, e.g. a project open via open_preview).

Round-trips through the gateway's blocking-prompt bridge like ``read_preview``:
tui_gateway emits ``tour.request``, the renderer drives driver.js (injecting it
into the preview's webview when needed) and answers ``tour.respond`` with the
outcome, so the agent knows whether the selector matched. This module is just
schema + a thin dispatcher over the platform-injected callback.

Lives in the ``desktop_ui`` toolset, which the GUI gateway enables only for
desktop-sourced sessions, and withdraws itself when the user has switched tours
off (Settings → Appearance). A tour takes the whole screen, so "no thanks" has
to mean the model is never told the tool exists — a switch that only made the
call fail would leave Hermes offering walkthroughs it cannot give.
"""

import json
from typing import Callable, Optional

from tools import desktop_ui
from tools.registry import registry, tool_error

ACTIONS = ("targets", "show", "start", "next", "prev", "stop")
SURFACES = ("app", "preview")
SIDES = ("top", "right", "bottom", "left")


def tour_tool(
    action: str = "",
    surface: Optional[str] = None,
    selector: Optional[str] = None,
    title: Optional[str] = None,
    text: Optional[str] = None,
    side: Optional[str] = None,
    steps: Optional[list] = None,
    step_index: Optional[int] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Dispatch one tour action to the desktop renderer and return its outcome."""
    if callback is None:
        return tool_error("tour is only available in the Hermes desktop app.")

    verb = (action or "").strip().lower()
    if verb not in ACTIONS:
        return tool_error(f"action must be one of: {', '.join(ACTIONS)}.")

    where = (surface or "app").strip().lower()
    if where not in SURFACES:
        return tool_error(f"surface must be one of: {', '.join(SURFACES)}.")

    if side is not None and side not in SIDES:
        return tool_error(f"side must be one of: {', '.join(SIDES)}.")

    # Every highlighted moment needs something to point at or something to say.
    def _empty(step: dict) -> bool:
        return not (step.get("selector") or step.get("title") or step.get("text"))

    if verb == "show" and _empty({"selector": selector, "title": title, "text": text}):
        return tool_error("show needs a selector (and/or title/text for the popover).")

    if verb == "start":
        if not isinstance(steps, list) or not steps:
            return tool_error("start needs a non-empty steps array.")
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return tool_error(f"steps[{i}] must be an object.")
            if _empty(step):
                return tool_error(f"steps[{i}] needs a selector and/or title/text.")

    payload = {
        key: val
        for key, val in (
            ("action", verb),
            ("surface", where),
            ("selector", selector),
            ("title", title),
            ("text", text),
            ("side", side),
            ("steps", steps),
            ("step_index", step_index),
        )
        if val is not None
    }

    try:
        raw = callback(payload)
    except Exception as exc:
        return tool_error(f"Tour action failed: {exc}")

    if not raw:
        return tool_error(
            "The tour request timed out, or no GUI window answered. "
            "For surface='preview' open a page in the preview pane first."
        )

    # The renderer answers with a JSON object; pass it through, else wrap it.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "selector": {
            "type": "string",
            "description": "Element to highlight; omit = centered narration.",
        },
        "title": {"type": "string", "description": "Popover title."},
        "text": {"type": "string", "description": "Popover body."},
        "side": {
            "type": "string",
            "enum": list(SIDES),
            "description": "Popover side; omit to auto-place.",
        },
    },
}

TOUR_SCHEMA = {
    "name": "tour",
    # Dieted (#95681): targets-first flow + stable-selector preference kept
    # (pre-effect: skipping them means guessed selectors on re-rendering UI).
    "description": (
        "Guided tour in the desktop GUI: dim the screen, highlight an "
        "element, attach a titled popover. Surfaces: 'app' (Hermes itself) "
        "or 'preview' (the page in the preview pane). ALWAYS call "
        "action='targets' first — prefer targets marked stable:true (their "
        "selectors survive re-renders); re-scan if one stops matching. Then "
        "narrate with action='show' (one highlight per call, replaces the "
        "last — pair each with a chat message) or hand over with "
        "action='start' + steps (user gets Next/Prev; 'next'/'prev' also "
        "page it). 'stop' clears. Use for how-does-X-work / where-is-Y "
        "walkthroughs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "description": "targets first; show narrates; start hands over.",
            },
            "surface": {
                "type": "string",
                "enum": list(SURFACES),
                "description": "'app' (default) or 'preview'.",
            },
            "selector": {
                "type": "string",
                "description": "show: selector from targets (prefer stable). Omit = centered narration.",
            },
            "title": {"type": "string", "description": "show: popover title."},
            "text": {"type": "string", "description": "show: popover body."},
            "side": {
                "type": "string",
                "enum": list(SIDES),
                "description": "show: popover side; omit to auto-place.",
            },
            "steps": {
                "type": "array",
                "items": _STEP_SCHEMA,
                "description": "start: ordered steps.",
            },
            "step_index": {
                "type": "integer",
                "description": "start: 0-indexed first step.",
            },
        },
        "required": ["action"],
    },
}


def check_tours_enabled() -> bool:
    """The user's Settings → Appearance switch. On unless they turned it off."""
    return desktop_ui.user_enabled("in_app_tours", default=True)


registry.register(
    name="tour",
    toolset="desktop_ui",
    schema=TOUR_SCHEMA,
    handler=lambda args, **kw: tour_tool(
        action=args.get("action", ""),
        surface=args.get("surface"),
        selector=args.get("selector"),
        title=args.get("title"),
        text=args.get("text"),
        side=args.get("side"),
        steps=args.get("steps"),
        step_index=args.get("step_index"),
        callback=kw.get("callback"),
    ),
    check_fn=check_tours_enabled,
    emoji="🧭",
)
