"""Native Hermes Computer Use tool registrations.

Greenfield Computer Use exposes explicit ``computer_use_*`` tools. There is no
model-facing catch-all action dispatcher; each tool name carries intent so
policy, approvals, logging, and the future Swift app can reason about calls
without reverse-parsing an ``action`` field.

Keep the ``registry.register(...)`` calls as top-level expressions. Hermes'
built-in discovery intentionally imports only modules with literal top-level
registrations so tool discovery stays cheap and deterministic in long-lived
CLI/gateway runtimes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from tools.computer_use.tool import (
    check_computer_use_requirements,
    handle_computer_use,
    set_approval_callback,
)
from tools.registry import registry


def _schema(name: str, description: str, properties: Dict[str, Any] | None = None, required: list[str] | None = None) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
        },
    }


def _handle(action: str, args: Dict[str, Any], **kwargs):
    payload = dict(args or {})
    payload["action"] = action
    return handle_computer_use(payload, **kwargs)


_COMMON_TARGET = {
    "app": {"type": "string", "description": "App name or bundle id, e.g. Helium or net.imput.helium."},
    "element": {"type": "integer", "description": "Element index from the latest app state."},
    "coordinate": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2, "description": "Fallback [x, y] coordinate when no accessibility element works."},
    "capture_after": {"type": "boolean", "description": "Take and return app state after the action."},
}

_LIST_APPS_DESCRIPTION = "List macOS apps/windows available to Computer Use."
_LAUNCH_APP_DESCRIPTION = "Launch a macOS app by app name or bundle id, then optionally capture its state."
_GET_APP_STATE_DESCRIPTION = "Get app/window state. Call this before and after every action."
_CLICK_DESCRIPTION = "Click an element or coordinate in the current app state."
_PERFORM_SECONDARY_ACTION_DESCRIPTION = "Perform an accessibility secondary action on an element, e.g. show menu."
_SCROLL_DESCRIPTION = "Scroll in an app/window or element."
_DRAG_DESCRIPTION = "Drag from one element/coordinate to another."
_TYPE_TEXT_DESCRIPTION = "Type text into the targeted app/window."
_SET_VALUE_DESCRIPTION = "Set a value on an element, including text fields, selects, popups, and sliders."
_PRESS_KEY_DESCRIPTION = "Press a key or key combo in the targeted app/window."
_SELECT_TEXT_DESCRIPTION = "Select text in an element or text field."
_DAEMON_DESCRIPTION = "Manage the Computer Use driver/daemon (status, start, stop) without shelling out."

registry.register(
    name="computer_use_list_apps",
    toolset="computer_use",
    schema=_schema("computer_use_list_apps", _LIST_APPS_DESCRIPTION, {}, []),
    handler=lambda args, **kw: _handle("list_apps", args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=_LIST_APPS_DESCRIPTION,
    override=True,
)

registry.register(
    name="computer_use_launch_app",
    toolset="computer_use",
    schema=_schema("computer_use_launch_app", _LAUNCH_APP_DESCRIPTION, {
        "app": {"type": "string", "description": "App name, e.g. Messages, Spotify, Safari."},
        "bundle_id": {"type": "string", "description": "Bundle id, e.g. com.apple.MobileSMS or com.spotify.client. Preferred when known."},
        "background": {"type": "boolean", "description": "Launch without stealing foreground focus when supported. Defaults to true."},
        "capture_after": _COMMON_TARGET["capture_after"],
    }, []),
    handler=lambda args, **kw: _handle("launch_app", args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=_LAUNCH_APP_DESCRIPTION,
    override=True,
)

registry.register(
    name="computer_use_get_app_state",
    toolset="computer_use",
    schema=_schema("computer_use_get_app_state", _GET_APP_STATE_DESCRIPTION, {
        "app": _COMMON_TARGET["app"],
        "mode": {"type": "string", "enum": ["som", "vision", "ax"], "description": "som: screenshot + accessibility elements; ax: tree only; vision: screenshot only."},
    }, []),
    handler=lambda args, **kw: _handle("get_app_state", args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=_GET_APP_STATE_DESCRIPTION,
    override=True,
)

registry.register(
    name="computer_use_click",
    toolset="computer_use",
    schema=_schema("computer_use_click", _CLICK_DESCRIPTION, {
        **deepcopy(_COMMON_TARGET),
        "button": {"type": "string", "enum": ["left", "right"], "description": "Mouse button; alias for mouse_button."},
        "mouse_button": {"type": "string", "enum": ["left", "right"], "description": "Codex-compatible alias for button."},
        "click_count": {"type": "integer", "minimum": 1, "maximum": 2},
    }, ["app"]),
    handler=lambda args, **kw: _handle("click", args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=_CLICK_DESCRIPTION,
    override=True,
)

registry.register(
    name="computer_use_perform_secondary_action",
    toolset="computer_use",
    schema=_schema("computer_use_perform_secondary_action", _PERFORM_SECONDARY_ACTION_DESCRIPTION, {
        **deepcopy(_COMMON_TARGET),
        "secondary_action": {"type": "string", "description": "AX action name, e.g. AXShowMenu or AXPress."},
    }, ["app", "element"]),
    handler=lambda args, **kw: _handle("perform_secondary_action", args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=_PERFORM_SECONDARY_ACTION_DESCRIPTION,
    override=True,
)

registry.register(
    name="computer_use_scroll",
    toolset="computer_use",
    schema=_schema("computer_use_scroll", _SCROLL_DESCRIPTION, {
        **deepcopy(_COMMON_TARGET),
        "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
        "amount": {"type": "integer", "description": "Wheel ticks; pages is preferred for Codex-style page scrolling."},
        "pages": {"type": "number", "description": "Codex-compatible scroll distance in pages; converted to wheel ticks."},
    }, ["app", "direction"]),
    handler=lambda args, **kw: _handle("scroll", args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=_SCROLL_DESCRIPTION,
    override=True,
)

registry.register(
    name="computer_use_drag",
    toolset="computer_use",
    schema=_schema("computer_use_drag", _DRAG_DESCRIPTION, {
        "app": _COMMON_TARGET["app"],
        "from_element": {"type": "integer"},
        "to_element": {"type": "integer"},
        "from_coordinate": _COMMON_TARGET["coordinate"],
        "to_coordinate": _COMMON_TARGET["coordinate"],
        "capture_after": _COMMON_TARGET["capture_after"],
    }, ["app"]),
    handler=lambda args, **kw: _handle("drag", args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=_DRAG_DESCRIPTION,
    override=True,
)

registry.register(
    name="computer_use_type_text",
    toolset="computer_use",
    schema=_schema("computer_use_type_text", _TYPE_TEXT_DESCRIPTION, {
        "app": _COMMON_TARGET["app"],
        "text": {"type": "string"},
        "capture_after": _COMMON_TARGET["capture_after"],
    }, ["app", "text"]),
    handler=lambda args, **kw: _handle("type_text", args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=_TYPE_TEXT_DESCRIPTION,
    override=True,
)

registry.register(
    name="computer_use_set_value",
    toolset="computer_use",
    schema=_schema("computer_use_set_value", _SET_VALUE_DESCRIPTION, {
        "app": _COMMON_TARGET["app"],
        "element": _COMMON_TARGET["element"],
        "value": {"type": "string"},
        "capture_after": _COMMON_TARGET["capture_after"],
    }, ["app", "element", "value"]),
    handler=lambda args, **kw: _handle("set_value", args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=_SET_VALUE_DESCRIPTION,
    override=True,
)

registry.register(
    name="computer_use_press_key",
    toolset="computer_use",
    schema=_schema("computer_use_press_key", _PRESS_KEY_DESCRIPTION, {
        "app": _COMMON_TARGET["app"],
        "key": {"type": "string", "description": "Key or combo, e.g. Return, Escape, cmd+s."},
        "capture_after": _COMMON_TARGET["capture_after"],
    }, ["app", "key"]),
    handler=lambda args, **kw: _handle("press_key", args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=_PRESS_KEY_DESCRIPTION,
    override=True,
)

registry.register(
    name="computer_use_select_text",
    toolset="computer_use",
    schema=_schema("computer_use_select_text", _SELECT_TEXT_DESCRIPTION, {
        "app": _COMMON_TARGET["app"],
        "element": _COMMON_TARGET["element"],
        "text": {"type": "string", "description": "Exact text to select, if supported."},
        "selection": {"type": "string", "description": "Selection mode, usually all or text."},
        "prefix": {"type": "string", "description": "Optional text expected immediately before target text."},
        "suffix": {"type": "string", "description": "Optional text expected immediately after target text."},
        "cursor": {"type": "string", "enum": ["before", "after"], "description": "Place cursor before/after selection when backend supports it."},
        "capture_after": _COMMON_TARGET["capture_after"],
    }, ["app", "element"]),
    handler=lambda args, **kw: _handle("select_text", args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=_SELECT_TEXT_DESCRIPTION,
    override=True,
)

def _handle_daemon(args: Dict[str, Any], **kwargs):
    payload = dict(args or {})
    if "subaction" not in payload:
        payload["subaction"] = payload.get("action") or "status"
    payload["action"] = "daemon"
    return handle_computer_use(payload, **kwargs)


registry.register(
    name="computer_use_daemon",
    toolset="computer_use",
    schema=_schema("computer_use_daemon", _DAEMON_DESCRIPTION, {
        "action": {"type": "string", "enum": ["status", "start", "stop"], "description": "Lifecycle action — status reports installation/version/permissions/running, start launches the driver, stop terminates it. Defaults to status."},
    }, []),
    handler=_handle_daemon,
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=_DAEMON_DESCRIPTION,
    override=True,
)

__all__ = [
    "handle_computer_use",
    "set_approval_callback",
    "check_computer_use_requirements",
]
