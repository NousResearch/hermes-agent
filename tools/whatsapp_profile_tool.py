"""WhatsApp profile management tools.

Currently supports updating/removing the profile picture for the authenticated
WhatsApp account connected through Hermes' local Baileys bridge.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict

from tools.registry import registry, tool_error


WHATSAPP_PROFILE_PICTURE_SCHEMA = {
    "name": "whatsapp_profile_picture",
    "description": (
        "Update or remove the profile picture for the WhatsApp account connected "
        "to Hermes. This changes account identity/appearance and should only be "
        "used after the user explicitly confirms the exact action. The WhatsApp "
        "bridge must be running and connected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set", "remove"],
                "description": "Use 'set' to upload image_path as the account profile picture, or 'remove' to clear it.",
            },
            "image_path": {
                "type": "string",
                "description": "Absolute path to a local image file. Required when action='set'.",
            },
            "width": {
                "type": "integer",
                "description": "Optional square resize width passed to Baileys. Defaults to Baileys' 640px; values above 640 are not recommended.",
            },
            "height": {
                "type": "integer",
                "description": "Optional square resize height passed to Baileys. Defaults to Baileys' 640px; values above 640 are not recommended.",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true only after explicit user confirmation for this account-level change.",
            },
        },
        "required": ["action", "confirmed"],
    },
}


def _error(message: str) -> str:
    return json.dumps({"success": False, "error": message})


def _success(data: Dict[str, Any]) -> str:
    return json.dumps({"success": True, **data})


def _bridge_port() -> int:
    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.WHATSAPP)
        if pconfig and pconfig.extra:
            return int(pconfig.extra.get("bridge_port", 3000))
    except Exception:
        pass
    return int(os.getenv("WHATSAPP_BRIDGE_PORT", "3000"))


def _post_to_bridge(payload: Dict[str, Any]) -> Dict[str, Any]:
    port = _bridge_port()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/profile-picture",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Host": f"127.0.0.1:{port}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body or "{}")
            except json.JSONDecodeError:
                data = {"raw": body}
            return {"status": resp.status, "data": data}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body or "{}")
        except json.JSONDecodeError:
            data = {"error": body}
        return {"status": exc.code, "data": data}


def whatsapp_profile_picture_tool(args: Dict[str, Any], **kw) -> str:
    action = str(args.get("action") or "").strip().lower()
    confirmed = bool(args.get("confirmed"))
    if action not in {"set", "remove"}:
        return tool_error("action must be either 'set' or 'remove'")
    if not confirmed:
        return tool_error(
            "WhatsApp profile picture changes require explicit user confirmation. "
            "Ask the user to confirm the exact image/action, then retry with confirmed=true."
        )

    payload: Dict[str, Any]
    if action == "remove":
        payload = {"remove": True}
    else:
        image_path = str(args.get("image_path") or "").strip()
        if not image_path:
            return tool_error("image_path is required when action='set'")
        if not os.path.isabs(image_path):
            return tool_error("image_path must be an absolute path")
        if not os.path.exists(image_path):
            return tool_error(f"File not found: {image_path}")
        payload = {"filePath": image_path}
        for key in ("width", "height"):
            value = args.get(key)
            if value is not None:
                try:
                    value_int = int(value)
                    if value_int > 0:
                        payload[key] = min(value_int, 640)
                except (TypeError, ValueError):
                    return tool_error(f"{key} must be a positive integer")

    try:
        result = _post_to_bridge(payload)
    except Exception as exc:
        return _error(f"Failed to reach WhatsApp bridge: {exc}")

    data = result.get("data") or {}
    if result.get("status") == 200 and data.get("success"):
        return _success({"action": data.get("action", action)})
    return _error(str(data.get("error") or data or "WhatsApp bridge request failed"))


def _check_whatsapp_profile_picture() -> bool:
    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.WHATSAPP)
        return bool(pconfig and pconfig.enabled)
    except Exception:
        return False


registry.register(
    name="whatsapp_profile_picture",
    toolset="messaging",
    schema=WHATSAPP_PROFILE_PICTURE_SCHEMA,
    handler=whatsapp_profile_picture_tool,
    check_fn=_check_whatsapp_profile_picture,
    emoji="🖼️",
)
