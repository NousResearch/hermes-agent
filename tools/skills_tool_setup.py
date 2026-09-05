"""Skill readiness: required env vars, secret capture, and setup notes. Split out of
``tools.skills_tool`` (names re-imported there); module state (``_secret_capture_callback``,
``load_env``) stays there and is read lazily at call time so origin-module patches are honored."""

import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List

from hermes_constants import display_hermes_home
from utils import env_var_enabled

logger = logging.getLogger("tools.skills_tool")

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REMOTE_ENV_BACKENDS = frozenset({"docker", "singularity", "modal", "ssh", "daytona", "vercel_sandbox"})


class SkillReadinessStatus(str, Enum):
    AVAILABLE = "available"
    SETUP_NEEDED = "setup_needed"
    UNSUPPORTED = "unsupported"


def _is_remote_env_backend(backend: str) -> bool:
    """Built-in remote backends plus plugin backends declaring is_remote."""
    if backend in _REMOTE_ENV_BACKENDS or not backend or backend == "local":
        return backend in _REMOTE_ENV_BACKENDS
    try:
        from agent.terminal_env_registry import provider_flag
        return bool(provider_flag(backend, "is_remote", False))
    except Exception:
        return False


def _as_dict_list(raw: Any) -> list:
    """Accept a single mapping or a list; anything else is treated as empty."""
    return [raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []


def _clean_str(value: Any) -> str | None:
    """Stripped string when *value* is a non-blank str, else None."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _get_required_environment_variables(frontmatter: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge required_environment_variables, setup.collect_secrets and legacy
    prerequisites.env_vars into one deduped, validated list (first entry wins)."""
    setup = frontmatter.get("setup")
    setup = setup if isinstance(setup, dict) else {}
    setup_help = _clean_str(setup.get("help"))
    prereqs = frontmatter.get("prerequisites")
    legacy = (prereqs.get("env_vars") if isinstance(prereqs, dict) else None) or []
    legacy = [legacy] if isinstance(legacy, str) else legacy
    required: Dict[str, Dict[str, Any]] = {}  # env name -> entry, insertion-ordered, first wins
    declared = _as_dict_list(frontmatter.get("required_environment_variables"))
    entries = [{"name": i} if isinstance(i, str) else i for i in declared
               if isinstance(i, (str, dict))]
    # collect_secrets entries: env_var is the name; provider_url (or url) doubles as help.
    entries += [
        {"name": i.get("env_var"), "prompt": i.get("prompt"),
         "url": str(i.get("provider_url") or i.get("url") or "").strip() or None}
        for i in _as_dict_list(setup.get("collect_secrets")) if isinstance(i, dict)]
    entries += [{"name": str(v)} for v in legacy if str(v).strip()]
    for entry in entries:
        env_name = str(entry.get("name") or entry.get("env_var") or "").strip()
        if not env_name or env_name in required or not _ENV_VAR_NAME_RE.match(env_name):
            continue
        normalized: Dict[str, Any] = {
            "name": env_name,
            "prompt": str(entry.get("prompt") or f"Enter value for {env_name}").strip()}
        if help_text := _clean_str(
                entry.get("help") or entry.get("provider_url") or entry.get("url") or setup_help):
            normalized["help"] = help_text
        if required_for := _clean_str(entry.get("required_for")):
            normalized["required_for"] = required_for
        if entry.get("optional"):
            normalized["optional"] = True
        required[env_name] = normalized
    return list(required.values())


def _capture_result(missing_names, setup_skipped=False, gateway_setup_hint=None):
    return {"missing_names": missing_names, "setup_skipped": setup_skipped, "gateway_setup_hint": gateway_setup_hint}


def _capture_required_environment_variables(
    skill_name: str, missing_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prompt for missing secrets via the registered capture callback (if any)."""
    from tools import skills_tool as _st
    if not missing_entries:
        return _capture_result([])
    missing_names = [entry["name"] for entry in missing_entries]
    # Messaging-platform gateway surfaces can't prompt for a secret, so they get the "unsupported"
    # hint. Interactive gateway surfaces (desktop app / TUI) set HERMES_INTERACTIVE (same flag
    # tools/approval.py uses) and register a callback routing to a secure secret.request overlay.
    if _is_gateway_surface() and not env_var_enabled("HERMES_INTERACTIVE"):
        try:
            from gateway.platforms.base import GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE as hint
        except Exception:
            hint = (f"Secure secret entry is not available. Load this skill in the local CLI to be "
                    f"prompted, or add the key to {display_hermes_home()}/.env manually.")
        return _capture_result(missing_names, gateway_setup_hint=hint)
    if (callback := _st._secret_capture_callback) is None:
        return _capture_result(missing_names)
    remaining_names: List[str] = []
    for entry in missing_entries:
        metadata = {"skill_name": skill_name, **{k: entry[k] for k in ("help", "required_for") if entry.get(k)}}
        try:
            callback_result = callback(entry["name"], entry["prompt"], metadata)
        except Exception:
            logger.warning(f"Secret capture callback failed for {entry['name']}", exc_info=True)
            callback_result = {"success": False, "stored_as": entry["name"], "validated": False, "skipped": True}
        ok = isinstance(callback_result, dict) and callback_result.get("success")
        if not (ok and not callback_result.get("skipped")):
            remaining_names.append(entry["name"])
    return _capture_result(remaining_names, bool(remaining_names))


def _is_gateway_surface() -> bool:
    if env_var_enabled("HERMES_GATEWAY_SESSION"):
        return True
    from gateway.session_context import get_session_env
    return bool(get_session_env("HERMES_SESSION_PLATFORM"))


def _is_env_var_persisted(var_name: str, env_snapshot: Dict[str, str]) -> bool:
    """Set in a regular-file snapshot, else in the active profile's loaded secrets."""
    from agent.secret_scope import get_secret

    return bool(env_snapshot[var_name] if var_name in env_snapshot else get_secret(var_name))


def _build_setup_note(
    readiness_status: SkillReadinessStatus, missing: List[str],
    setup_help: str | None = None) -> str | None:
    if readiness_status != SkillReadinessStatus.SETUP_NEEDED:
        return None
    note = f"Setup needed before using this skill: missing {', '.join(missing) if missing else 'required prerequisites'}."
    return f"{note} {setup_help}" if setup_help else note
