"""Guided tour (highlight + narrate UI elements) in the Hermes desktop GUI: the agent discovers
targets (``action="targets"``), then highlights one step at a time (``show``) or hands over a
step list the user pages (``start``). Round-trips through the gateway blocking-prompt bridge
(``tour.request``/``tour.respond``) so the agent learns whether the selector matched. Lives in
``desktop_ui`` and withdraws itself when tours are off: a tour takes the whole screen, so "off"
must mean the model is never told the tool exists rather than offered a call that fails."""

import json
from typing import Callable, Optional

from tools import desktop_ui
from tools.registry import registry, tool_error

ACTIONS = ("targets", "show", "start", "next", "prev", "stop")
SURFACES = ("app", "preview")
SIDES = ("top", "right", "bottom", "left")


def tour_tool(action: str = "", surface: Optional[str] = None, selector: Optional[str] = None,
              title: Optional[str] = None, text: Optional[str] = None, side: Optional[str] = None,
              steps: Optional[list] = None, step_index: Optional[int] = None,
              callback: Optional[Callable] = None) -> str:
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
    if verb == "show" and not (selector or title or text):
        return tool_error("show needs a selector (and/or title/text for the popover).")
    if verb == "start":
        if not isinstance(steps, list) or not steps:
            return tool_error("start needs a non-empty steps array.")
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return tool_error(f"steps[{i}] must be an object.")
            if not (step.get("selector") or step.get("title") or step.get("text")):
                return tool_error(f"steps[{i}] needs a selector and/or title/text.")
    fields = {"action": verb, "surface": where, "selector": selector, "title": title,
              "text": text, "side": side, "steps": steps, "step_index": step_index}
    try:
        raw = callback({key: val for key, val in fields.items() if val is not None})
    except Exception as exc:
        return tool_error(f"Tour action failed: {exc}")
    if not raw:
        return tool_error("The tour request timed out, or no GUI window answered. "
                          "For surface='preview' open a page in the preview pane first.")
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
    "name": "gui_tour",
    # Description keeps the targets-first flow + stable-selector preference:
    # without them the model guesses selectors on re-rendering UI.
    # See #95681.
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
    name="gui_tour", toolset="desktop_ui", schema=TOUR_SCHEMA, check_fn=check_tours_enabled,
    handler=lambda args, **kw: tour_tool(
        action=args.get("action", ""), callback=kw.get("callback"),
        **{k: args.get(k) for k in ("surface", "selector", "title", "text", "side", "steps",
                                    "step_index")}),
    emoji="🧭")
