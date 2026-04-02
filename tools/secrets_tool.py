#!/usr/bin/env python3
"""Secrets tool — secure secret lifecycle management.

Phase 1 implementation for issue #410. Provides an agent-facing interface for:
- listing configured secret names (never values)
- checking which keys are configured vs missing
- requesting secure input (CLI only for now)
- deleting secrets from ~/.hermes/.env
- registering env_passthrough for the next sandboxed subprocess

Important: secret values never enter the LLM context. The `request` action
handles capture internally via the existing secure secret callback path.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from tools.env_passthrough import get_all_passthrough, register_env_passthrough
from tools.registry import registry
from tools.skills_tool import SKILLS_DIR, _parse_frontmatter, _get_required_environment_variables
from hermes_cli.config import OPTIONAL_ENV_VARS, get_env_value, load_env, save_env_value, get_env_path

logger = logging.getLogger(__name__)

_ENV_VAR_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SECRET_CAPTURE_CALLBACK = None
_SECRET_SUFFIXES = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_PASSWD",
    "_PRIVATE_KEY",
    "_ACCESS_KEY",
    "_AUTH_TOKEN",
    "_REFRESH_TOKEN",
    "_CLIENT_SECRET",
    "_BOT_TOKEN",
    "_APP_TOKEN",
)
_SECRET_EXACT_NAMES = {
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_PHONE_NUMBER",
}
_KNOWN_PASSWORD_VARS = {
    name for name, info in OPTIONAL_ENV_VARS.items() if isinstance(info, dict) and info.get("password")
}


def _is_secret_like_name(name: str) -> bool:
    name = _normalize_key_name(name)
    if not name:
        return False
    if name in _SECRET_EXACT_NAMES:
        return True
    if name in _KNOWN_PASSWORD_VARS:
        return True
    return name.endswith(_SECRET_SUFFIXES)


def set_secrets_request_callback(callback) -> None:
    global _SECRET_CAPTURE_CALLBACK
    _SECRET_CAPTURE_CALLBACK = callback


def _normalize_key_name(key: str) -> str:
    key = str(key or "").strip().upper()
    if not _ENV_VAR_NAME_RE.match(key):
        return ""
    return key


def _configured_secret_names() -> List[str]:
    env_snapshot = load_env()
    names = []
    for name, value in env_snapshot.items():
        if value and _is_secret_like_name(name):
            names.append(name)
    for name, value in os.environ.items():
        if value and _is_secret_like_name(name) and name not in names:
            names.append(name)
    return sorted(names)


def _delete_env_key(key: str) -> None:
    env_path = get_env_path()
    if env_path.exists():
        try:
            lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
            kept = [line for line in lines if not line.strip().startswith(f"{key}=")]
            env_path.write_text("".join(kept), encoding="utf-8")
        except Exception:
            # Fall back to blanking the key if removal fails for any reason.
            save_env_value(key, "")
            os.environ.pop(key, None)
            return
    os.environ.pop(key, None)



def _required_secrets_for_skills() -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    configured = set(_configured_secret_names())

    if not SKILLS_DIR.exists():
        return result

    for skill_md in SKILLS_DIR.rglob("SKILL.md"):
        try:
            content = skill_md.read_text(encoding="utf-8")
            frontmatter, _body = _parse_frontmatter(content)
            required_entries = _get_required_environment_variables(frontmatter)
        except Exception:
            continue

        missing = sorted(
            {
                _normalize_key_name(entry.get("name"))
                for entry in required_entries
                if _normalize_key_name(entry.get("name")) not in configured
            }
        )
        missing = [name for name in missing if name]
        if missing:
            result[skill_md.parent.name] = missing

    return result


def _action_list(_args: Dict[str, Any], **_kwargs) -> str:
    return json.dumps(
        {
            "secrets": _configured_secret_names(),
            "missing_for_skills": _required_secrets_for_skills(),
            "passthrough_registered": sorted(get_all_passthrough()),
        },
        ensure_ascii=False,
    )


def _action_check(args: Dict[str, Any], **_kwargs) -> str:
    keys = args.get("keys") or []
    if not isinstance(keys, list):
        return json.dumps({"error": "keys must be a list"})

    configured, missing, rejected = [], [], []
    for item in keys:
        name = _normalize_key_name(item)
        if not name:
            continue
        if not _is_secret_like_name(name):
            rejected.append(name)
            continue
        if get_env_value(name):
            configured.append(name)
        else:
            missing.append(name)

    return json.dumps({"configured": configured, "missing": missing, "rejected": rejected}, ensure_ascii=False)


def _action_request(args: Dict[str, Any], **_kwargs) -> str:
    key = _normalize_key_name(args.get("key"))
    if not key:
        return json.dumps({"error": "A valid key is required"})
    if not _is_secret_like_name(key):
        return json.dumps({"error": f"{key} is not a supported secret-like variable name"})

    description = str(args.get("description") or "").strip()
    instructions = str(args.get("instructions") or "").strip()
    prompt = str(args.get("prompt") or f"Enter value for {key}").strip()

    if _SECRET_CAPTURE_CALLBACK is None:
        hint = "Use the local CLI to be prompted securely, or set it manually in ~/.hermes/.env."
        try:
            if os.getenv("HERMES_GATEWAY_SESSION") or os.getenv("HERMES_SESSION_PLATFORM"):
                from gateway.platforms.base import GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE
                hint = GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE
        except Exception:
            pass
        return json.dumps(
            {
                "success": False,
                "key": key,
                "error": "Secure secret entry is not available in this surface.",
                "hint": hint,
            },
            ensure_ascii=False,
        )

    metadata = {}
    if description:
        metadata["description"] = description
    if instructions:
        metadata["instructions"] = instructions

    try:
        result = _SECRET_CAPTURE_CALLBACK(key, prompt, metadata)
    except Exception as e:
        logger.warning("Secret capture callback failed for %s", key, exc_info=True)
        return json.dumps({"success": False, "key": key, "error": str(e)}, ensure_ascii=False)

    payload = {
        "success": bool(isinstance(result, dict) and result.get("success")),
        "key": key,
        "stored": bool(isinstance(result, dict) and result.get("success") and not result.get("skipped")),
        "skipped": bool(isinstance(result, dict) and result.get("skipped")),
        "message": (result or {}).get("message") if isinstance(result, dict) else None,
    }
    return json.dumps(payload, ensure_ascii=False)


def _action_delete(args: Dict[str, Any], **_kwargs) -> str:
    key = _normalize_key_name(args.get("key"))
    if not key:
        return json.dumps({"error": "A valid key is required"})
    if not _is_secret_like_name(key):
        return json.dumps({"error": f"{key} is not a supported secret-like variable name"})

    _delete_env_key(key)
    return json.dumps({"success": True, "deleted": key}, ensure_ascii=False)


def _action_inject(args: Dict[str, Any], **_kwargs) -> str:
    keys = args.get("keys") or []
    if not isinstance(keys, list):
        return json.dumps({"error": "keys must be a list"})

    to_register = []
    missing = []
    rejected = []
    for item in keys:
        name = _normalize_key_name(item)
        if not name:
            continue
        if not _is_secret_like_name(name):
            rejected.append(name)
            continue
        if get_env_value(name):
            to_register.append(name)
        else:
            missing.append(name)

    if to_register:
        register_env_passthrough(to_register)

    return json.dumps(
        {
            "success": True,
            "injected": sorted(set(to_register)),
            "missing": sorted(set(missing)),
            "rejected": sorted(set(rejected)),
        },
        ensure_ascii=False,
    )


def secrets_tool(args: Dict[str, Any], **kwargs) -> str:
    action = str(args.get("action") or "").strip().lower()
    if action == "list":
        return _action_list(args, **kwargs)
    if action == "check":
        return _action_check(args, **kwargs)
    if action == "request":
        return _action_request(args, **kwargs)
    if action == "delete":
        return _action_delete(args, **kwargs)
    if action == "inject":
        return _action_inject(args, **kwargs)
    return json.dumps({"error": f"Unknown action: {action}"})


registry.register(
    name="secrets",
    toolset="secrets",
    description="Securely manage API keys and other credentials without exposing values to the model. List configured secret names, check which keys are missing, request secure input in CLI, delete secrets, or register secrets for env passthrough.",
    emoji="🔐",
    schema={
        "name": "secrets",
        "description": "Secure secret lifecycle management. Secret values are never returned to the model.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "check", "request", "delete", "inject"],
                    "description": "The secrets action to perform.",
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Secret names to check or inject.",
                },
                "key": {
                    "type": "string",
                    "description": "Single secret name for request/delete.",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description shown during secure prompt.",
                },
                "instructions": {
                    "type": "string",
                    "description": "Optional user instructions for where to find the secret.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Custom secure prompt text.",
                },
            },
            "required": ["action"],
        },
    },
    handler=secrets_tool,
)
