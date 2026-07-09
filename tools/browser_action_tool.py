"""Agent-facing remote browser action request tool.

This is the model-facing half of the Hermes browser-extension action bridge.
The tool does not execute page actions directly; it asks the owning TUI/desktop
session to emit ``browser.action.requested`` to the browser extension, where the
extension UI/policy layer can approve, deny, or run the action and return a
sanitized acknowledgement.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from tools.registry import registry, tool_error
from utils import env_var_enabled


_ALLOWED_ACTION_TYPES = ("getSnapshot", "screenshot", "scroll", "click", "typeText", "select", "openUrl")


BROWSER_ACTION_REQUEST_SCHEMA: dict[str, Any] = {
    "name": "browser_action_request",
    "description": (
        "Request a browser action through the user's connected Hermes browser extension. "
        "This is a remote request/approval bridge: Hermes validates and emits the request, "
        "the extension decides whether to execute it, and mutating actions require explicit "
        "user approval. Prefer read-only actions (`getSnapshot`, `screenshot`) first; use "
        "`scroll` for low-risk page motion; `click`, `typeText`, `select`, and `openUrl` "
        "are approval-gated. The tool returns only sanitized status/metadata and never raw "
        "screenshot data or private URL paths/query strings."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action_type": {
                "type": "string",
                "enum": list(_ALLOWED_ACTION_TYPES),
                "description": "Browser action type to request.",
            },
            "target": {
                "type": "object",
                "description": (
                    "Optional target descriptor for click/type/select actions. Use safe fields "
                    "such as selector, text, role, name, or ref from a prior browser snapshot."
                ),
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                    "role": {"type": "string"},
                    "name": {"type": "string"},
                    "ref": {"type": "string"},
                },
            },
            "url": {
                "type": "string",
                "description": "URL for openUrl or context. The gateway strips path/query/hash to origin and blocks restricted categories.",
            },
            "value": {
                "type": "string",
                "description": "Text/value for typeText or select. Sensitive values must not be typed through this tool.",
            },
            "direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "Scroll direction for scroll actions. Defaults to down.",
            },
            "request_id": {
                "type": "string",
                "description": "Optional caller-supplied request id for tracing. Omit to let Hermes generate one.",
            },
            "wait_for_result": {
                "type": "boolean",
                "description": "Wait for the extension result acknowledgement before returning. Defaults to true.",
                "default": True,
            },
            "timeout": {
                "type": "number",
                "description": "Seconds to wait for a result when wait_for_result is true. Default 300, max 300.",
                "default": 300,
            },
        },
        "required": ["action_type"],
    },
}


def browser_action_request(
    action_type: str,
    target: Optional[dict[str, Any]] = None,
    url: Optional[str] = None,
    value: Optional[str] = None,
    direction: Optional[str] = None,
    request_id: Optional[str] = None,
    wait_for_result: bool = True,
    timeout: float = 300,
    callback: Optional[Callable[..., Any]] = None,
) -> str:
    """Request a browser-extension action through the platform callback."""
    if callback is None:
        return tool_error(
            "browser_action_request is only available in a TUI/desktop session with the browser action bridge enabled."
        )
    action: dict[str, Any] = {"type": action_type}
    if isinstance(target, dict) and target:
        action["target"] = target
    if url:
        action["url"] = url
    if value:
        action["value"] = value
    if direction:
        action["direction"] = direction
    if request_id:
        action["requestId"] = request_id

    try:
        bounded_timeout = min(max(float(timeout or 300), 1.0), 300.0)
    except (TypeError, ValueError):
        bounded_timeout = 300.0

    try:
        raw = callback(action=action, wait_for_result=bool(wait_for_result), timeout=bounded_timeout)
    except Exception as exc:
        return tool_error(f"Browser action request failed: {exc}")

    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"ok": True, "status": "requested", "raw": str(raw)}, ensure_ascii=False)


def check_browser_action_requirements() -> bool:
    """Gateway/TUI-only; the callback is injected by tui_gateway.server."""
    return env_var_enabled("HERMES_GATEWAY_SESSION") or bool(os.environ.get("HERMES_TUI_TOOLSETS"))


registry.register(
    name="browser_action_request",
    toolset="browser-action",
    schema=BROWSER_ACTION_REQUEST_SCHEMA,
    handler=lambda args, **kw: browser_action_request(
        action_type=args.get("action_type") or args.get("type", ""),
        target=args.get("target"),
        url=args.get("url"),
        value=args.get("value") or args.get("text"),
        direction=args.get("direction"),
        request_id=args.get("request_id") or args.get("requestId"),
        wait_for_result=args.get("wait_for_result", True),
        timeout=args.get("timeout", 300),
        callback=kw.get("callback"),
    ),
    check_fn=check_browser_action_requirements,
    emoji="🌐",
)
