"""Agent-callable masked secret capture.

Only secret metadata crosses the model/tool boundary. The value is collected by the
interactive surface callback and is never returned to the handler.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Optional

from hermes_constants import display_hermes_home
from tools.registry import registry

SecretCaptureCallback = Callable[[str, str, Optional[dict]], dict]
_callback_context: ContextVar[Optional[SecretCaptureCallback]] = ContextVar(
    "secret_capture_callback",
    default=None,
)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DESTINATIONS = {"profile_env", "bitwarden_sm"}


def _set_context_secret_capture_callback(callback: Optional[SecretCaptureCallback]) -> None:
    _callback_context.set(callback)


def get_secret_capture_callback() -> Optional[SecretCaptureCallback]:
    """Return the callback bound to the current turn context."""
    return _callback_context.get()


def secret_capture(
    var_name: str,
    prompt: str,
    destination: str = "profile_env",
    task_id: str | None = None,
) -> str:
    """Open the surface's masked prompt and persist the answer without returning it."""
    del task_id  # handler parity; capture is scoped by the current surface callback
    name = str(var_name or "").strip()
    label = str(prompt or "").strip()
    target = str(destination or "profile_env").strip()
    if not _ENV_NAME_RE.fullmatch(name):
        return json.dumps({"success": False, "error": "var_name must be a valid environment variable name."})
    if not label:
        return json.dumps({"success": False, "error": "prompt must not be empty."})
    if target not in _DESTINATIONS:
        return json.dumps({"success": False, "error": "destination must be profile_env or bitwarden_sm."})
    callback = get_secret_capture_callback()
    if callback is None:
        return json.dumps({
            "success": False,
            "error": (
                "Secure secret entry is unavailable on this surface. Use a local CLI, TUI, "
                f"or desktop session; do not paste the secret into chat. Profile secrets can "
                f"also be added directly to {display_hermes_home()}/.env."
            ),
        })
    try:
        result = callback(name, label, {"destination": target, "source": "secret_capture"})
    except Exception:
        # Callback exceptions can carry provider output. Keep them out of tool results and logs;
        # the surface is responsible for rendering a safe, local diagnostic.
        return json.dumps({"success": False, "error": "Secure secret capture failed."})
    if not isinstance(result, dict):
        return json.dumps({"success": False, "error": "Secure secret capture failed."})
    safe = {
        "success": bool(result.get("success")),
        "stored_as": str(result.get("stored_as") or name),
        "destination": target,
    }
    if result.get("skipped"):
        safe.update(skipped=True, reason=str(result.get("reason") or "cancelled"))
    elif not safe["success"]:
        safe["error"] = str(result.get("error") or "Secret storage failed.")
    return json.dumps(safe)


SECRET_CAPTURE_SCHEMA: dict[str, Any] = {
    "name": "secret_capture",
    "description": (
        "Open the current interactive surface's masked secret prompt and store an API key, token, "
        "password, or other secret without exposing its value to the model, chat, logs, argv, or "
        "tool result. Call this whenever a user needs to add, rotate, or replace a secret. Never "
        "ask the user to paste a secret into chat. The call is allowed even when the variable "
        "already exists. Messaging/headless surfaces fail safely instead of accepting plaintext."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "var_name": {
                "type": "string",
                "description": "Environment-variable name used to identify the secret, such as OPENROUTER_API_KEY.",
            },
            "prompt": {
                "type": "string",
                "description": "Short user-facing label shown beside the masked input; never include a secret value.",
            },
            "destination": {
                "type": "string",
                "enum": ["profile_env", "bitwarden_sm"],
                "default": "profile_env",
                "description": (
                    "profile_env stores in the active profile's .env. bitwarden_sm creates or "
                    "replaces the named secret in the configured Bitwarden Secrets Manager project."
                ),
            },
        },
        "required": ["var_name", "prompt"],
    },
}


registry.register(
    name="secret_capture",
    toolset="skills",
    schema=SECRET_CAPTURE_SCHEMA,
    handler=lambda args, **kw: secret_capture(
        var_name=args.get("var_name", ""),
        prompt=args.get("prompt", ""),
        destination=args.get("destination", "profile_env"),
        task_id=kw.get("task_id"),
    ),
    emoji="🔐",
)
