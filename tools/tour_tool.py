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
desktop-sourced sessions.
"""

import json
from typing import Callable, Optional

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
            "description": "CSS selector of the element this step highlights. Omit for a centered narration-only step.",
        },
        "title": {"type": "string", "description": "Popover title."},
        "text": {"type": "string", "description": "Popover body text."},
        "side": {
            "type": "string",
            "enum": list(SIDES),
            "description": "Preferred popover side. Omit to auto-place.",
        },
    },
}

TOUR_SCHEMA = {
    "name": "tour",
    "description": (
        "Give a live guided tour in the Hermes desktop GUI: dim the screen, "
        "highlight an element, and attach a popover with your own title/text. "
        "Works on two surfaces — 'app' (the Hermes app itself) and 'preview' "
        "(whatever page is open in the in-app browser, so any web app can be "
        "toured). ALWAYS call action='targets' first to discover what is on "
        "screen instead of guessing selectors; each target reports "
        "`stable: true` when its selector keys off identity (data-tour, id, "
        "data-testid, aria-label) and survives a re-render — prefer those, and "
        "re-scan if a selector stops matching. Then either narrate at your own "
        "pace with action='show' (one highlight per call — replaces the "
        "previous one; pair each with a chat message describing it), or hand "
        "control to the user with action='start' + a steps array (driver.js "
        "renders Next/Prev buttons; 'next'/'prev' also page it "
        "programmatically). action='stop' clears the tour. Use when the user "
        "asks how something works, where something is, or for a walkthrough of "
        "an app or workflow."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "description": "targets: list tourable elements. show: highlight one element. start: begin a multi-step user-paced tour. next/prev: page a started tour. stop: end the tour.",
            },
            "surface": {
                "type": "string",
                "enum": list(SURFACES),
                "description": "Where the tour runs: 'app' (Hermes desktop UI, default) or 'preview' (the page in the in-app browser pane).",
            },
            "selector": {
                "type": "string",
                "description": "For show: CSS selector of the element to highlight (from action='targets', preferring a stable one). Omit for a centered narration popover.",
            },
            "title": {"type": "string", "description": "For show: popover title."},
            "text": {"type": "string", "description": "For show: popover body text."},
            "side": {
                "type": "string",
                "enum": list(SIDES),
                "description": "For show: preferred popover side. Omit to auto-place.",
            },
            "steps": {
                "type": "array",
                "items": _STEP_SCHEMA,
                "description": "For start: the ordered tour steps.",
            },
            "step_index": {
                "type": "integer",
                "description": "For start: 0-indexed step to begin at (default 0).",
            },
        },
        "required": ["action"],
    },
}


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
    emoji="🧭",
)
