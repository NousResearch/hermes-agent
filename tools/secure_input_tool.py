#!/usr/bin/env python3
"""Secure input tool.

Allows an agent to request a secret through an interactive UI without receiving
or storing the raw value in the model transcript. The tool returns an opaque
``secret_ref`` that scoped runtime tools can consume.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from agent.secure_input_broker import SecretBrokerError, register_secret
from tools.registry import registry

logger = logging.getLogger(__name__)

_secure_input_callback: Callable[..., str] | None = None


def set_secure_input_callback(callback) -> None:
    """Register a UI callback that returns the raw secret to runtime code only."""
    global _secure_input_callback
    _secure_input_callback = callback


def check_secure_input_requirements() -> bool:
    return True


def _normalize_allowed_consumers(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ["terminal"]
    consumers = [str(item).strip() for item in value if str(item).strip()]
    return consumers or ["terminal"]


def request_secure_input(
    *,
    purpose: str,
    title: str = "Secure input required",
    description: str = "Paste the secret. It will not be sent to the model.",
    secret_type: str = "token",
    ttl_seconds: int = 600,
    single_use: bool = True,
    allowed_consumers: list[str] | None = None,
    task_id: str | None = None,
) -> str:
    """Prompt the user for a secret and return a redacted broker reference."""
    if _secure_input_callback is None:
        return json.dumps(
            {
                "success": False,
                "error": (
                    "No secure input UI is available in this session. Ask the user "
                    "to provide the secret through a local secret file or run the "
                    "auth command directly."
                ),
            },
            ensure_ascii=False,
        )

    purpose = str(purpose or "secure_input").strip() or "secure_input"
    title = str(title or "Secure input required").strip()
    description = str(description or "").strip()
    consumers = _normalize_allowed_consumers(allowed_consumers)
    metadata = {
        "purpose": purpose,
        "title": title,
        "description": description,
        "secret_type": str(secret_type or "secret"),
        "ttl_seconds": ttl_seconds,
        "single_use": bool(single_use),
        "allowed_consumers": consumers,
        "secure_input": True,
    }

    try:
        value = _secure_input_callback(purpose, title, metadata)
        info = register_secret(
            value,
            purpose=purpose,
            allowed_consumers=consumers,
            ttl_seconds=ttl_seconds,
            single_use=single_use,
            label=title,
        )
        return json.dumps(
            {
                "success": True,
                **info,
                "message": (
                    "Secret captured securely. The raw value was not exposed to "
                    "the model; pass secret_ref to a compatible tool."
                ),
            },
            ensure_ascii=False,
        )
    except SecretBrokerError as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)
    except Exception as exc:
        logger.debug("secure input capture failed", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


REQUEST_SECURE_INPUT_SCHEMA = {
    "name": "request_secure_input",
    "description": (
        "Ask the active Hermes UI to collect a password/token/API key in a secure "
        "masked prompt. The raw secret bypasses the LLM and is stored only in an "
        "ephemeral in-memory broker; this tool returns an opaque secret_ref for "
        "compatible tools such as terminal. Use this instead of asking the user to "
        "paste secrets into chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "purpose": {
                "type": "string",
                "description": "Short machine-readable purpose, e.g. github_token or npm_token.",
            },
            "title": {
                "type": "string",
                "description": "Human-facing prompt title shown by the UI.",
            },
            "description": {
                "type": "string",
                "description": "Human-facing explanation of how the secret will be used.",
            },
            "secret_type": {
                "type": "string",
                "description": "Kind of secret: token, password, api_key, oauth_code, etc.",
                "default": "token",
            },
            "ttl_seconds": {
                "type": "integer",
                "description": "How long the secret_ref remains usable. Capped at 3600 seconds.",
                "minimum": 1,
                "default": 600,
            },
            "single_use": {
                "type": "boolean",
                "description": "When true, the secret_ref can be consumed only once.",
                "default": True,
            },
            "allowed_consumers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tool consumers allowed to use the ref. Defaults to ['terminal'].",
            },
        },
        "required": ["purpose"],
    },
}


def _handle_request_secure_input(args, **kw):
    return request_secure_input(
        purpose=args.get("purpose", "secure_input"),
        title=args.get("title", "Secure input required"),
        description=args.get("description", "Paste the secret. It will not be sent to the model."),
        secret_type=args.get("secret_type", "token"),
        ttl_seconds=args.get("ttl_seconds", 600),
        single_use=args.get("single_use", True),
        allowed_consumers=args.get("allowed_consumers"),
        task_id=kw.get("task_id"),
    )


registry.register(
    name="request_secure_input",
    toolset="secure_input",
    schema=REQUEST_SECURE_INPUT_SCHEMA,
    handler=_handle_request_secure_input,
    check_fn=check_secure_input_requirements,
    emoji="🔐",
)
