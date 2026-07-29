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
                    "cua_browser_state",
                    "cua_browser_prepare",
                    "cua_browser_navigate",
                    "cua_browser_click",
                    "cua_browser_type",
                    "cua_browser_pointer",
                    "cua_browser_dialog",
                    "cua_browser_set_input_files",
                    "cua_browser_download",
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
                    "restores the prior frontmost app. A `confirmed` effect is "
                    "done. For `unverifiable`, inspect fresh state before any "
                    "retry even if escalation is recommended. Escalate only "
                    "after `suspected_noop` or a structured refusal. Do not "
                    "predict the rung from the app being Electron/Chromium. "
                    "Foreground is a visible focus change and needs its own "
                    "approval."
                ),
            },
            "bring_to_front": {
                "type": "boolean",
                "description": (
                    "Optional and only valid with delivery_mode='foreground'. "
                    "Explicitly invokes cua-driver's standalone bring_to_front "
                    "tool before the input; it is never passed as an input "
                    "property. This persistent focus change has a separate "
                    "approval scope. Default false."
                ),
            },
            # ── cua-driver typed browser route ─────────────────────
            "tab_id": {
                "type": "string",
                "description": "Opaque tab capability returned by cua_browser_state.",
            },
            "ref": {
                "type": "string",
                "description": "Current semantic ref from the latest cua_browser_state snapshot.",
            },
            "destination_ref": {
                "type": "string",
                "description": "Current destination ref for a typed pointer action.",
            },
            "url": {"type": "string", "description": "URL for cua_browser_navigate."},
            "input_route": {
                "type": "string",
                "enum": ["trusted", "dom_event"],
                "description": (
                    "Typed-browser trust class. Defaults to trusted. dom_event "
                    "is an explicit downgrade and is never selected silently."
                ),
            },
            "snapshot_format": {
                "type": "string",
                "enum": ["semantic_v2", "dom_refs_v1"],
                "description": "Typed-browser snapshot format; semantic_v2 is the default.",
            },
            "query": {"type": "string", "description": "Optional browser-state query."},
            "scope_ref": {"type": "string", "description": "Optional current ref to scope a snapshot."},
            "continuation": {"type": "string", "description": "Continuation minted by the current snapshot."},
            "profile_mode": {
                "type": "string",
                "enum": ["isolated_new", "isolated_named", "existing_profile"],
                "description": (
                    "Browser preparation mode. existing_profile always requires "
                    "the driver's separate interactive grant."
                ),
            },
            "profile_name": {"type": "string", "description": "Name for isolated_named setup."},
            "allow_launch": {
                "type": "boolean",
                "description": "Explicitly allow launch of a driver-owned isolated browser.",
            },
            "browser_pointer_action": {
                "type": "string",
                "enum": ["hover", "right_click", "double_click", "scroll", "drag"],
                "description": "Operation for cua_browser_pointer.",
            },
            "browser_dialog_action": {
                "type": "string",
                "enum": ["inspect", "accept", "dismiss"],
                "description": "Page JavaScript dialog action; native prompts stay on the native ladder.",
            },
            "browser_type_mode": {
                "type": "string",
                "enum": ["insert_text", "keystrokes"],
                "description": "Delivery form for cua_browser_type; defaults to insert_text.",
            },
            "dialog_id": {"type": "string", "description": "Opaque page-dialog capability."},
            "prompt_text": {"type": "string", "description": "Optional text for a page prompt dialog."},
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit paths for cua_browser_set_input_files.",
            },
            "destination_root": {
                "type": "string",
                "description": "Approved destination root for cua_browser_download.",
            },
            "delta_x": {"type": "number", "description": "Typed pointer horizontal delta."},
            "delta_y": {"type": "number", "description": "Typed pointer vertical delta."},
            "x": {"type": "number", "description": "Typed browser viewport x coordinate."},
            "y": {"type": "number", "description": "Typed browser viewport y coordinate."},
            "to_x": {"type": "number", "description": "Typed browser drag destination x."},
            "to_y": {"type": "number", "description": "Typed browser drag destination y."},
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
