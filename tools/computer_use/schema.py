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
        ],
        "description": (
            "Which action to perform. `capture` is free (no side effects). All other actions "
            "require approval unless auto-approved. Use `set_value` for select/popup elements and "
            "sliders — it selects the matching option directly without opening the native menu (no "
            "focus steal)."
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
    "max_elements": {
        "type": "integer",
        "description": (
            "Only for action='capture'. Bound on how many accessibility nodes the driver walks "
            "(depth-first; the tree is a prefix of the unbounded one). Use it to get a usable "
            "capture when a window's UIA provider is slow and the default walk times out."
        ),
    },
    "max_depth": {
        "type": "integer",
        "description": (
            "Only for action='capture'. Bound on accessibility-tree walk depth; nodes deeper than "
            "this are omitted. Same timeout escape hatch as max_elements, by tree level instead of "
            "count."
        ),
    },
    "element": {
        "type": "integer",
        "description": (
            "The element index returned by the last capture (driver indices may start at 0). Prefer this "
            "over raw coordinates: the backend automatically attaches the snapshot token when available. "
            "A stale index can be refused (`snapshot_id_required`) — re-capture rather than guessing pixels."
        ),
    },
    "element_token": {
        "type": "string",
        "description": (
            "Opaque per-snapshot element handle from the last capture's `elements[].element_token` "
            "(e.g. 's00000001:6'). Carries the snapshot with the index, so a superseded capture fails "
            "closed as stale instead of silently re-resolving to a different element. Use the token "
            "EXACTLY as returned — never construct one from indices, screenshot or cache file names."
        ),
    },
    "snapshot_id": {
        "type": "string",
        "description": (
            "The snapshot handle from the last capture (`elements[].element_token` prefixes it). "
            "Fallback provenance for `element=` actions on drivers that accept snapshot_id but not "
            "element_token. Stale snapshots are refused; re-capture to refresh."
        ),
    },
    "coordinate": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
        "description": (
            "Pixel coordinates [x, y] in WINDOW-LOCAL SCREENSHOT pixels — the same space as the "
            "image the last capture returned for this window (top-left of the screenshot is 0,0). "
            "Element `bounds` are NOT this space: they are screen-absolute native pixels; convert "
            "with the frame/scale note the capture prints, or prefer `element`/`element_token`. "
            "Only use coordinates if no element index is available."
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
        "description": "Modifier keys held during the action.",
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
            "For input actions (click, type, key, drag, scroll). `background` (DEFAULT) delivers "
            "without raising the window or stealing focus. `foreground` briefly fronts the window "
            "then restores focus — a visible change needing its own approval; use it only when a "
            "result's verdict tells you to escalate there. Each result's `verdict` carries the "
            "next step; follow it rather than guessing."
        ),
    },
    "bring_to_front": {
        "type": "boolean",
        "description": (
            "Optional and only valid with delivery_mode='foreground'. Explicitly invokes "
            "cua-driver's standalone bring_to_front tool before the input; it is never passed as "
            "an input property. This persistent focus change has a separate approval scope. "
            "Default false."
        ),
    },
    "capture_after": {
        "type": "boolean",
        "description": (
            "If true, take a follow-up capture after the action and include it in the response. "
            "Saves a round-trip when you need to verify an action's effect."
        ),
    },
}

COMPUTER_USE_SCHEMA: dict[str, Any] = {
    "name": "computer_use",
    "description": (
        "Drive the desktop via cua-driver — screenshots, mouse, keyboard, scroll, drag — on macOS, "
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
