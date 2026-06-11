#!/usr/bin/env python3
"""Protected sensitive-input tool.

The agent can request one secret value from the current user/session without the
raw value entering LLM-visible tool results.  In gateway mode the next typed
message from the same user/session is intercepted by
``tools.secret_capture_gateway``; in contexts without a gateway callback the
tool fails closed and tells the agent not to ask for the secret in normal chat.
"""
from __future__ import annotations

import json
import re
import uuid

from tools.registry import registry, tool_error

_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_error(code: str, env_var: str, message: str | None = None) -> str:
    payload = {
        "success": False,
        "stored_as": env_var,
        "secret_capture_status": code,
        "error": code,
    }
    if message:
        payload["message"] = message
    return json.dumps(payload, ensure_ascii=False)


def request_sensitive_input(env_var: str, prompt: str, timeout_seconds: int | None = None) -> str:
    env_var = (env_var or "").strip()
    prompt = (prompt or "").strip()
    if not env_var or not _ENV_VAR_RE.match(env_var):
        return tool_error("env_var must be a valid environment variable name.")
    if not prompt:
        return tool_error("prompt is required.")

    try:
        from tools.approval import get_current_session_key
        from tools import secret_capture_gateway as scg
    except Exception:
        return _safe_error(
            "unavailable",
            env_var,
            "Protected sensitive input is not available in this session. Do not ask the user to paste the secret in normal chat.",
        )

    session_key = get_current_session_key("")
    notify = scg.get_notify(session_key)
    if notify is None:
        # Local CLI already has a masked secret-entry bridge for skill setup.
        # Reuse it when present so the tool is not advertised in CLI only to
        # fail at runtime.  The callback returns a safe receipt and never
        # exposes the raw value to the model.
        try:
            from tools import skills_tool as _skills_tool
            callback = getattr(_skills_tool, "_secret_capture_callback", None)
            if callback is not None:
                receipt = callback(env_var, prompt) or {}
                if receipt.get("success") and not receipt.get("skipped"):
                    receipt = dict(receipt)
                    receipt["secret_capture_status"] = "stored"
                    receipt.pop("value", None)
                    receipt.pop("secret", None)
                    return json.dumps(receipt, ensure_ascii=False)
        except Exception:
            pass
        return _safe_error(
            "unavailable",
            env_var,
            "Protected sensitive input is not available in this session. Do not ask the user to paste the secret in normal chat.",
        )

    secret_id = uuid.uuid4().hex[:12]
    entry = scg.register(secret_id, session_key, env_var, prompt)
    try:
        notify(entry)
    except Exception:
        scg.clear_session(session_key)
        return _safe_error("send_failed", env_var)

    try:
        timeout = int(timeout_seconds) if timeout_seconds is not None else int(scg.get_secret_capture_timeout())
    except Exception:
        timeout = scg.get_secret_capture_timeout()
    resolved = scg.wait_for_response(secret_id, timeout=float(timeout))
    if resolved is None:
        return _safe_error("missing_state", env_var)
    if resolved.cancelled:
        status = resolved.reason or "cancelled"
        return json.dumps({
            "success": False,
            "stored_as": env_var,
            "secret_capture_status": status,
            "skipped": True,
        }, ensure_ascii=False)
    if not resolved.value:
        return json.dumps({
            "success": False,
            "stored_as": env_var,
            "secret_capture_status": "empty",
            "skipped": True,
        }, ensure_ascii=False)

    try:
        from hermes_cli.config import save_env_value_secure
        receipt = save_env_value_secure(env_var, resolved.value)
    except Exception:
        # Do not echo exception details after a secret has been captured; some
        # lower layers may include values or file snippets in error strings.
        return _safe_error("store_failed", env_var)
    receipt.update({
        "secret_capture_status": "stored",
        "message": f"Stored {env_var} via protected sensitive-input capture.",
    })
    # Defense-in-depth: never include the raw value in LLM-visible output.
    receipt.pop("value", None)
    receipt.pop("secret", None)
    return json.dumps(receipt, ensure_ascii=False)


def check_sensitive_input_requirements() -> bool:
    return True


REQUEST_SENSITIVE_INPUT_SCHEMA = {
    "name": "request_sensitive_input",
    "description": (
        "Request a password, OTP, API key, token, or other secret from the user "
        "via protected local/gateway capture. The next typed payload is stored "
        "locally under env_var and is never returned to the model. Use this instead "
        "of asking for secrets in normal chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "env_var": {
                "type": "string",
                "description": "Environment variable / local alias to store, e.g. TDLIB_TEST_PASSWORD.",
            },
            "prompt": {
                "type": "string",
                "description": "User-facing prompt explaining exactly what secret to send.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Optional wait timeout in seconds. Defaults to agent.secret_capture_timeout or clarify_timeout.",
            },
        },
        "required": ["env_var", "prompt"],
    },
}


registry.register(
    name="request_sensitive_input",
    toolset="sensitive_input",
    schema=REQUEST_SENSITIVE_INPUT_SCHEMA,
    handler=lambda args, **kw: request_sensitive_input(
        env_var=args.get("env_var", ""),
        prompt=args.get("prompt", ""),
        timeout_seconds=args.get("timeout_seconds"),
    ),
    check_fn=check_sensitive_input_requirements,
    emoji="🔐",
)
