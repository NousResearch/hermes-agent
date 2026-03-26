#!/usr/bin/env python3
from __future__ import annotations

import getpass
import json
import logging
import os
import re
from pathlib import Path
from typing import Callable

import yaml

from hermes_constants import get_hermes_home
from tools.registry import registry
from hermes_cli.config import get_env_value, save_env_value, OPTIONAL_ENV_VARS
from tools.env_passthrough import register_env_passthrough

logger = logging.getLogger(__name__)

_secret_capture_callback: Callable[[str, str, dict[str, str]], object] | None = None
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_NAME_PATTERN = re.compile(
    r"(^|_)(SECRET|TOKEN|PASSWORD|PASS|PASSWD|API_KEY|KEY|CLIENT_SECRET|ACCESS_TOKEN|PRIVATE_KEY)($|_)",
    re.IGNORECASE,
)


def set_secret_capture_callback(callback) -> None:
    global _secret_capture_callback
    _secret_capture_callback = callback


def _normalize_key(key: object | None) -> str:
    return str(key or "").strip()


def _normalize_keys(keys: list[str] | None, key: str | None = None) -> list[str]:
    merged: list[str] = []
    for item in keys or []:
        normalized = _normalize_key(item)
        if normalized:
            merged.append(normalized)
    if key:
        normalized = _normalize_key(key)
        if normalized:
            merged.append(normalized)
    return list(dict.fromkeys(merged))


def _is_gateway_surface() -> bool:
    if os.getenv("HERMES_GATEWAY_SESSION"):
        return True
    return bool(os.getenv("HERMES_SESSION_PLATFORM"))


def _env_file_path() -> Path:
    return get_hermes_home() / ".env"


def _scan_env_key_names() -> set[str]:
    env_path = _env_file_path()
    if not env_path.exists():
        return set()

    found: set[str] = set()
    try:
        for raw_line in env_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key_name = line.split("=", 1)[0].strip()
            if _ENV_VAR_NAME_RE.match(key_name):
                found.add(key_name)
    except Exception:
        logger.debug("Failed scanning %s for key names", env_path, exc_info=True)

    return found


def _known_secret_keys() -> set[str]:
    known: set[str] = set()
    for key_name, meta in OPTIONAL_ENV_VARS.items():
        if isinstance(meta, dict) and meta.get("password") is True:
            known.add(key_name)
    return known


def _looks_like_secret_key(key_name: str) -> bool:
    return bool(_SECRET_NAME_PATTERN.search(key_name))


def _discover_secret_keys() -> list[str]:
    keys = _known_secret_keys()
    for env_key in _scan_env_key_names():
        if env_key in keys or _looks_like_secret_key(env_key):
            keys.add(env_key)
    return sorted(keys)


def _parse_frontmatter(content: str) -> dict[str, object]:
    if not content.startswith("---"):
        return {}
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return {}
    yaml_content = content[3 : end_match.start() + 3]
    try:
        parsed = yaml.safe_load(yaml_content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}
    return {}


def _skill_secret_requirements() -> list[dict[str, object]]:
    skills_dir = get_hermes_home() / "skills"
    if not skills_dir.exists():
        return []

    requirements: list[dict[str, object]] = []
    for skill_md in skills_dir.rglob("SKILL.md"):
        try:
            frontmatter = _parse_frontmatter(
                skill_md.read_text(encoding="utf-8", errors="replace")
            )
            raw_requires = frontmatter.get("requires_secrets")
            if isinstance(raw_requires, str):
                raw_requires = [raw_requires]
            if not isinstance(raw_requires, list):
                continue

            required_keys = [
                _normalize_key(item)
                for item in raw_requires
                if _normalize_key(item) and _ENV_VAR_NAME_RE.match(_normalize_key(item))
            ]
            if not required_keys:
                continue

            skill_name = _normalize_key(frontmatter.get("name")) or skill_md.parent.name
            requirements.append(
                {
                    "skill": skill_name,
                    "requires_secrets": sorted(list(dict.fromkeys(required_keys))),
                }
            )
        except Exception:
            logger.debug(
                "Failed parsing skill frontmatter from %s", skill_md, exc_info=True
            )
    requirements.sort(key=lambda item: str(item.get("skill", "")))
    return requirements


def _list_action() -> dict[str, object]:
    configured = [
        key_name
        for key_name in _discover_secret_keys()
        if bool(get_env_value(key_name))
    ]

    missing_for_skills: set[str] = set()
    skills_missing: list[dict[str, object]] = []

    for requirement in _skill_secret_requirements():
        required_for_skill = requirement.get("requires_secrets")
        if not isinstance(required_for_skill, list):
            continue
        missing = [
            key_name
            for key_name in required_for_skill
            if not bool(get_env_value(key_name))
        ]
        if missing:
            missing_sorted = sorted(missing)
            missing_for_skills.update(missing_sorted)
            skills_missing.append(
                {
                    "skill": requirement["skill"],
                    "missing": missing_sorted,
                }
            )

    return {
        "configured": sorted(configured),
        "missing_for_skills": sorted(missing_for_skills),
        "skills_missing_secrets": skills_missing,
    }


def _check_action(keys: list[str] | None, key: str | None = None) -> dict[str, object]:
    normalized = _normalize_keys(keys, key)
    configured = [item for item in normalized if bool(get_env_value(item))]
    missing = [item for item in normalized if not bool(get_env_value(item))]
    return {
        "configured": configured,
        "missing": missing,
    }


def _request_action(
    key: str | None,
    description: str | None = None,
    instructions: str | None = None,
) -> dict[str, object]:
    normalized_key = _normalize_key(key)
    if not normalized_key:
        return {"error": "key is required for action='request'"}
    if not _ENV_VAR_NAME_RE.match(normalized_key):
        return {"error": f"invalid secret key: {normalized_key}"}

    prompt_label = _normalize_key(description) or normalized_key
    prompt = f"{prompt_label}"
    normalized_instructions = _normalize_key(instructions)
    if normalized_instructions:
        prompt = f"{prompt_label} ({normalized_instructions})"

    if _secret_capture_callback is not None:
        metadata = {
            "description": _normalize_key(description) or None,
            "instructions": _normalize_key(instructions) or None,
        }
        metadata = {k: v for k, v in metadata.items() if v}
        try:
            callback_result = _secret_capture_callback(normalized_key, prompt, metadata)
        except Exception:
            logger.warning(
                "Secret capture callback failed for %s", normalized_key, exc_info=True
            )
            callback_result = None

        if isinstance(callback_result, dict):
            if (
                isinstance(callback_result.get("value"), str)
                and callback_result["value"]
            ):
                save_env_value(normalized_key, callback_result["value"])
                return {"stored": True, "key": normalized_key}

            if callback_result.get("stored") is True:
                return {"stored": True, "key": normalized_key}

            if callback_result.get("success") and not callback_result.get("skipped"):
                return {"stored": True, "key": normalized_key}

    if _is_gateway_surface():
        return {
            "stored": False,
            "key": normalized_key,
            "gateway_secret_prompt": {
                "required": True,
                "channel": "dm",
                "key": normalized_key,
                "description": _normalize_key(description) or normalized_key,
                "instructions": _normalize_key(instructions) or None,
            },
        }

    try:
        entered_secret = getpass.getpass(f"Enter secret for {normalized_key}: ")
    except (EOFError, KeyboardInterrupt):
        entered_secret = ""

    if not entered_secret:
        return {"stored": False, "key": normalized_key}

    save_env_value(normalized_key, entered_secret)
    return {"stored": True, "key": normalized_key}


def _delete_action(key: str | None) -> dict[str, object]:
    normalized_key = _normalize_key(key)
    if not normalized_key:
        return {"error": "key is required for action='delete'"}
    if not _ENV_VAR_NAME_RE.match(normalized_key):
        return {"error": f"invalid secret key: {normalized_key}"}

    save_env_value(normalized_key, "")
    return {"deleted": True, "key": normalized_key}


def _inject_action(keys: list[str] | None, key: str | None = None) -> dict[str, object]:
    normalized = _normalize_keys(keys, key)
    if not normalized:
        return {"error": "keys is required for action='inject'"}

    register_env_passthrough(normalized)
    return {"injected": normalized}


def secrets_handler(
    action: str | None,
    keys: list[str] | None = None,
    key: str | None = None,
    description: str | None = None,
    instructions: str | None = None,
    task_id: str | None = None,
) -> str:
    normalized_action = _normalize_key(action).lower()

    if normalized_action == "list":
        result = _list_action()
    elif normalized_action == "check":
        result = _check_action(keys=keys, key=key)
    elif normalized_action == "request":
        result = _request_action(
            key=key,
            description=description,
            instructions=instructions,
        )
    elif normalized_action == "delete":
        result = _delete_action(key=key)
    elif normalized_action == "inject":
        result = _inject_action(keys=keys, key=key)
    else:
        result = {
            "error": "Invalid action. Use one of: list, check, request, delete, inject"
        }

    return json.dumps(result, ensure_ascii=False)


SECRETS_SCHEMA = {
    "name": "secrets",
    "description": "Securely manage secret lifecycle: list, check, request, delete, and inject secrets for terminal passthrough.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "check", "request", "delete", "inject"],
                "description": "Secret operation to perform.",
            },
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Secret keys for check/inject actions.",
            },
            "key": {
                "type": "string",
                "description": "Single secret key for request/delete actions.",
            },
            "description": {
                "type": "string",
                "description": "Human-readable label for request prompts.",
            },
            "instructions": {
                "type": "string",
                "description": "URL or help text for locating the secret.",
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="secrets",
    toolset="core",
    schema=SECRETS_SCHEMA,
    handler=lambda args, **kw: secrets_handler(
        action=args.get("action"),
        keys=args.get("keys"),
        key=args.get("key"),
        description=args.get("description"),
        instructions=args.get("instructions"),
        task_id=kw.get("task_id"),
    ),
    emoji="🔐",
)
