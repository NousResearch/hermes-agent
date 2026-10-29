"""Schema for the generic `computer_use` tool (model-facing; value is byte-frozen —
the schema goes to the model every turn, so prompt-cache parity depends on it).

Model-agnostic: any tool-calling model can drive this. Vision-capable models
should prefer `capture(mode='som')` then `click(element=N)` — much more reliable
than pixel coordinates, which remain supported for models trained on them.
"""

from __future__ import annotations

from typing import Any, Dict

# One consolidated tool with an `action` discriminator keeps the schema compact
# and the per-turn token cost low. Property groups: capture (mode, app, pid,
# window_id) / targeting (element, coordinate, button, modifiers) / drag / scroll /
# set_value / type-key-wait / focus_app / delivery ladder / return shape.
_PROPERTIES: dict[str, Any] = {
    "action": {
        "type": "string",
        "enum": [
            "capture",
            "click",
            "double_click",
            "right_click",
            "middle_click",
            "drag",
            "scroll",
            "type",
            "key",
            "set_value",
            "wait",
            "list_apps",
            "list_windows",
            "focus_app",
            "run_steps",
        ],
        "description": (
            "Which action to perform. `capture` is free (no side effects). All other actions "
            "require approval unless auto-approved. Use `set_value` for select/popup elements and "
            "sliders — it selects the matching option directly without opening the native menu (no "
            "focus steal). Use `run_steps` to send an ordered BATCH of actions in ONE call; "
            "separate calls do not provide batch-level validation or one end-of-batch observation. "
            "Parallel tool calls have no guaranteed input order."
        ),
    },
    "mode": {
        "type": "string",
        "enum": ["som", "vision", "ax"],
        "description": (
            "Capture mode. `som` (default) is a screenshot with numbered overlays on every "
            "interactable element plus the AX tree — best for vision models, lets you click by "
            "element index. `vision` is a plain screenshot. `ax` is the accessibility tree only "
            "(no image; useful for text-only models)."
        ),
    },
    "app": {
        "type": "string",
        "description": (
            "Optional. Limit capture/action to one app (name e.g. 'Safari', or bundle ID). Omitted "
            "= frontmost window. app='screen' = composited full-screen grab (image only, no "
            "clickable elements); app='desktop' = the OS desktop/shell surface (wallpaper, icons, "
            "taskbar) with its elements."
        ),
    },
    "pid": {
        "type": "integer",
        "description": (
            "Optional exact process target for action='capture'. Pair with window_id when "
            "discovery cannot resolve an X11 app."
        ),
    },
    "window_id": {
        "type": "integer",
        "description": (
            "Optional exact native window target for action='capture'. Pair with pid when an "
            "external cua-driver list_windows lookup has already identified the window."
        ),
    },
    "element": {
        "type": "integer",
        "description": (
            "The 1-based SOM index returned by the last `capture(mode='som')` call. Strongly "
            "preferred over raw coordinates."
        ),
    },
    "coordinate": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
        "description": (
            "Pixel coordinates [x, y] relative to the captured window screenshot (top-left "
            "origin). Only use this if no element index is available."
        ),
    },
    "button": {
        "type": "string",
        "enum": ["left", "right", "middle"],
        "description": "Mouse button. Defaults to left.",
    },
    "modifiers": {
        "type": "array",
        "items": {
            "type": "string",
            "enum": [
                "cmd",
                "shift",
                "option",
                "alt",
                "ctrl",
                "fn",
                "win",
                "windows",
                "super",
                "meta",
            ],
        },
        "description": "Modifier keys held during click/drag/scroll actions; in run_steps click/drag steps this plural Hermes spelling is translated to the driver's singular `modifier` field.",
    },
    "from_element": {"type": "integer", "description": "Source element index (drag)."},
    "to_element": {"type": "integer", "description": "Target element index (drag)."},
    "from_coordinate": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
        "description": "Source [x,y] (drag; use when no element available).",
    },
    "to_coordinate": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
        "description": "Target [x,y] (drag; use when no element available).",
    },
    "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "Scroll direction."},
    "amount": {"type": "integer", "description": "Scroll wheel ticks. Default 3."},
    "value": {
        "type": "string",
        "description": (
            "For action='set_value': the value to set on the element. For AXPopUpButton / select "
            "dropdowns, pass the option's display label (e.g. 'Blue'). For sliders and other "
            "AXValue-settable elements, pass the numeric or string value."
        ),
    },
    "text": {"type": "string", "description": "Text to type (respects the current layout)."},
    "keys": {
        "type": "string",
        "description": (
            "Key combo, e.g. 'cmd+s', 'ctrl+alt+t', 'return', 'escape', 'tab'. Use '+' to combine."
        ),
    },
    "seconds": {"type": "number", "description": "wait: seconds to pause (max 30)."},
    "raise_window": {
        "type": "boolean",
        "description": (
            "Only for action='focus_app'. If true, brings the window to front (DISRUPTS the user). "
            "Default false — input is routed to the app without raising, matching the background "
            "co-work model."
        ),
    },
    "delivery_mode": {
        "type": "string",
        "enum": ["background", "foreground"],
        "description": (
            "For input actions (click, type, key, drag, scroll, run_steps). `background` (DEFAULT) delivers "
            "without raising the window or stealing focus. `foreground` briefly fronts the window "
            "then restores focus — a visible change needing its own approval; use it only when a "
            "result's verdict tells you to escalate there. Each result's `verdict` carries the "
            "next step; follow it rather than guessing."
        ),
    },
    "bring_to_front": {
        "type": "boolean",
        "description": (
            "Optional and only valid with delivery_mode='foreground' for single-action inputs. "
            "Explicitly invokes cua-driver's standalone bring_to_front tool before the input; it is "
            "never passed as an input property and has a separate approval scope. Not supported "
            "for run_steps; use focus_app or an explicit foreground action instead. Default false."
        ),
    },
    "capture_after": {
        "type": "boolean",
        "description": (
            "If true, take a follow-up capture after a single action and include it in the response. "
            "Not supported for run_steps; use its observe argument for an end-of-batch read. "
            "Saves a round-trip when you need to verify an action's effect."
        ),
    },
    "steps": {
        "type": "array",
        "items": {"type": "object"},
        "maxItems": 64,
        "description": (
            "For action='run_steps': ordered driver actions, each as {'tool': <driver tool>, 'args': {...}} "
            "or short form {'<tool>': {...}}. Hermes aliases: key uses keys='cmd+s' or key+modifiers; "
            "type accepts text= or value=; right_click/middle_click select the click button. Click/drag "
            "modifiers are translated to the driver's singular 'modifier' property; drag accepts "
            "from_coordinate/to_coordinate pixel pairs (not element indices). A step may add wait_for "
            "(must hold before) and/or expect (checks after); execution is ordered and stops at the first "
            "failure, with all steps validated before the first runs. Capture a target window (or use "
            "focus_app) first; role/name steps need a window. Maximum 64 steps. The driver press_key takes "
            "a single key name; key combinations use the key alias or hotkey. Operator names like 'plus' "
            "may not be valid key names; click an app's operator button by role/name instead."
        ),
    },
    "observe": {
        "type": "object",
        "description": (
            "For action='run_steps': optional end-of-batch observation — arguments for ONE "
            "get_window_state read after the last step. Pass include_screenshot=true to see the "
            "window. Defaults to a since='latest' diff of what changed."
        ),
    },
    "delay_ms": {
        "type": "integer",
        "maximum": 2000,
        "minimum": 0,
        "description": (
            "For action='run_steps': pause between steps in milliseconds (not after the last). "
            "Default 0. Use a small delay (e.g. 50) when a target needs a moment to settle."
        ),
    },
}

COMPUTER_USE_SCHEMA: dict[str, Any] = {
    "name": "computer_use",
    "description": (
        "Drive the desktop via cua-driver: screenshots, mouse, keyboard, scroll, drag. macOS, "
        "Windows, and Linux. Input is background-FIRST, not background-only: the default delivery "
        "routes to the target window without stealing the user's cursor or focus (works even on "
        "hidden/minimized windows), and when a result's `verdict` says to escalate you climb — "
        "pixel coordinates, or delivery_mode='foreground' (briefly fronts the window; separate "
        "approval). Each result carries a `verdict` with the next step; follow it — never repeat "
        "confirmed input, and re-capture to verify an unverifiable one before retrying. Workflow: "
        "action='capture' (mode='som' gives numbered element overlays), then click by `element` "
        "index; re-capture after state-changing actions (or pass capture_after=true). Image "
        "captures include a shareable `screenshot_path`; deliver it via the platform's MEDIA "
        "syntax when the user asks to see it — not for captures used only for control."
    ),
    "parameters": {"type": "object", "properties": _PROPERTIES, "required": ["action"]},
}

def get_computer_use_schema() -> dict[str, Any]:
    """Return the generic OpenAI function-calling schema."""
    return COMPUTER_USE_SCHEMA
