"""Schema for the generic `computer_use` tool.

Model-agnostic. Any tool-calling model can drive this. Vision-capable models
should prefer `capture(mode='som')` then `click(element=N)` — much more
reliable than pixel coordinates. Pixel coordinates remain supported for
models that were trained on them (e.g. Claude's computer-use RL).
"""

from __future__ import annotations

from typing import Any, Dict


# One consolidated tool with an `action` discriminator. Keeps the schema
# compact and the per-turn token cost low.
COMPUTER_USE_SCHEMA: Dict[str, Any] = {
    "name": "computer_use",
    "description": (
        "Drive the desktop in the background via cua-driver — screenshots, "
        "mouse, keyboard, scroll, drag — without stealing the user's cursor "
        "or keyboard focus. Supported on macOS, Windows, and Linux. "
        "Preferred workflow: call with "
        "action='capture' (mode='som' gives numbered element overlays), "
        "then click by `element` index for reliability. Pixel coordinates "
        "are supported for models trained on them. Works on any window — "
        "hidden, minimized, or behind another app. Requires cua-driver to "
        "be installed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
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
                    "Which action to perform. `capture` is free (no side "
                    "effects). All other actions require approval unless "
                    "auto-approved. Use `set_value` for select/popup elements "
                    "and sliders — it selects the matching option directly "
                    "without opening the native menu (no focus steal)."
                ),
            },
            # ── capture ────────────────────────────────────────────
            "mode": {
                "type": "string",
                "enum": ["som", "vision", "ax"],
                "description": (
                    "Capture mode. `som` (default) is a screenshot with "
                    "numbered overlays on every interactable element plus "
                    "the AX tree — best for vision models, lets you click "
                    "by element index. `vision` is a plain screenshot. "
                    "`ax` is the accessibility tree only (no image; useful "
                    "for text-only models)."
                ),
            },
            "app": {
                "type": "string",
                "description": (
                    "Optional. Limit capture/action to a specific app "
                    "(by name, e.g. 'Safari', or bundle ID, "
                    "'com.apple.Safari'). If omitted, operates on the "
                    "frontmost app's window. Pass app='screen' (or "
                    "'desktop') to capture the OS desktop/shell surface — "
                    "e.g. to see the wallpaper or click the taskbar. Note: "
                    "capture is per-window; a single image cannot span "
                    "multiple monitors, so on a multi-screen setup capture "
                    "one window or display at a time."
                ),
            },
            "pid": {
                "type": "integer",
                "description": (
                    "Optional exact process target for action='capture'. Pair "
                    "with window_id when discovery cannot resolve an X11 app."
                ),
            },
            "window_id": {
                "type": "integer",
                "description": (
                    "Optional exact native window target for action='capture'. "
                    "Pair with pid when an external cua-driver list_windows "
                    "lookup has already identified the window."
                ),
            },
            "max_elements": {
                "type": "integer",
                "description": (
                    "Optional cap on the AX `elements` array returned by "
                    "`action='capture'`. Default 100, hard maximum 1000. "
                    "Dense UIs (Electron apps such as Obsidian or VS Code, "
                    "JetBrains IDEs) can publish 500+ AX nodes — capping "
                    "prevents a single capture from blowing session "
                    "context. When the cap trims the response, "
                    "`total_elements` and `truncated_elements` are "
                    "surfaced in the result so you can re-call with "
                    "`app=` to narrow scope or raise `max_elements` when "
                    "the full tree is required. Has no effect on "
                    "`mode='som'` / `mode='vision'` when a screenshot is "
                    "included in the response; only the rare image-"
                    "missing fallback returns an `elements` array and is "
                    "subject to the cap."
                ),
                "default": 100,
                "minimum": 1,
                "maximum": 1000,
            },
            # ── click / drag / scroll targeting ────────────────────
            "element": {
                "type": "integer",
                "description": (
                    "The 1-based SOM index returned by the last "
                    "`capture(mode='som')` call. Strongly preferred over "
                    "raw coordinates."
                ),
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Pixel coordinates [x, y] relative to the captured window "
                    "screenshot (top-left origin). Only use this if no element "
                    "index is available."
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
                        "cmd", "shift", "option", "alt", "ctrl", "fn",
                        "win", "windows", "super", "meta",
                    ],
                },
                "description": "Modifier keys held during the action.",
            },
            # ── drag ───────────────────────────────────────────────
            "from_element": {"type": "integer",
                              "description": "Source element index (drag)."},
            "to_element": {"type": "integer",
                            "description": "Target element index (drag)."},
            "from_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2, "maxItems": 2,
                "description": "Source [x,y] (drag; use when no element available).",
            },
            "to_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2, "maxItems": 2,
                "description": "Target [x,y] (drag; use when no element available).",
            },
            # ── scroll ─────────────────────────────────────────────
            "direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "Scroll direction.",
            },
            "amount": {
                "type": "integer",
                "description": "Scroll wheel ticks. Default 3.",
            },
            # ── set_value ──────────────────────────────────────────
            "value": {
                "type": "string",
                "description": (
                    "For action='set_value': the value to set on the element. "
                    "For AXPopUpButton / select dropdowns, pass the option's "
                    "display label (e.g. 'Blue'). For sliders and other "
                    "AXValue-settable elements, pass the numeric or string value."
                ),
            },
            # ── type / key / wait ──────────────────────────────────
            "text": {
                "type": "string",
                "description": "Text to type (respects the current layout).",
            },
            "keys": {
                "type": "string",
                "description": (
                    "Key combo, e.g. 'cmd+s', 'ctrl+alt+t', 'return', "
                    "'escape', 'tab'. Use '+' to combine."
                ),
            },
            "seconds": {
                "type": "number",
                "description": "Seconds to wait. Max 30.",
            },
            # ── focus_app ──────────────────────────────────────────
            "raise_window": {
                "type": "boolean",
                "description": (
                    "Only for action='focus_app'. If true, brings the "
                    "window to front (DISRUPTS the user). Default false "
                    "— input is routed to the app without raising, "
                    "matching the background co-work model."
                ),
            },
            # ── delivery (verify → escalate ladder) ────────────────
            "delivery_mode": {
                "type": "string",
                "enum": ["background", "foreground"],
                "description": (
                    "How input is delivered, for the input actions (click, "
                    "double_click, right_click, drag, scroll, type, key). "
                    "`background` (DEFAULT) routes input to the target without "
                    "raising it or stealing focus — the co-work model. "
                    "`foreground` briefly fronts the window, acts, then "
                    "restores the prior frontmost app. Only escalate to "
                    "`foreground` when a background attempt did NOT land — i.e. "
                    "a prior result had `effect: 'suspected_noop'`, "
                    "`code: 'background_unavailable'`, or "
                    "`escalation.recommended: 'foreground'`. Do not predict it "
                    "from the app being Electron/Chromium; react to the "
                    "returned signal. Foreground is a visible focus change and "
                    "needs its own approval."
                ),
            },
            "bring_to_front": {
                "type": "boolean",
                "description": (
                    "Optional, pairs with delivery_mode='foreground'. Keep the "
                    "target fronted after the action instead of restoring the "
                    "previous app, to avoid a per-call flash across a short "
                    "sequence of foreground actions. Default false."
                ),
            },
            # ── return shape ───────────────────────────────────────
            "capture_after": {
                "type": "boolean",
                "description": (
                    "If true, take a follow-up capture after the action "
                    "and include it in the response. Saves a round-trip "
                    "when you need to verify an action's effect."
                ),
            },
        },
        "required": ["action"],
    },
}


def get_computer_use_schema() -> Dict[str, Any]:
    """Return the generic OpenAI function-calling schema."""
    return COMPUTER_USE_SCHEMA
