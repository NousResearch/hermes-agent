"""Direct Hermes profile-to-profile messaging tool.

This tool lets one Hermes agent ask another configured Hermes profile a
bounded question through the local Hermes CLI, instead of relying on a shared
Telegram/Slack room.  It intentionally uses subprocess argv lists (no shell)
and validates target profile names before spawning.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agent.redact import redact_sensitive_text
from hermes_constants import get_default_hermes_root
from tools.registry import registry, tool_error

_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SESSION_ID_RE = re.compile(r"^\s*session_id:\s*(\S+)\s*$", re.MULTILINE)
_DEFAULT_TIMEOUT_SECONDS = 300
_MAX_TIMEOUT_SECONDS = 1800
_ALLOWED_MODES = {"sync"}


AGENT_MESSAGE_SCHEMA = {
    "name": "agent_message",
    "description": (
        "Send a direct request to another local Hermes profile by invoking the "
        "target profile through the Hermes CLI. Use this for agent-to-agent "
        "coordination when a real response from that profile is needed; use "
        "kanban or a durable artifact for multi-step work that must survive "
        "the current turn. This does not send via Telegram/Slack."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target_profile": {
                "type": "string",
                "description": "Hermes profile name to contact, e.g. 'narvi', 'aragorn', 'emily', or 'murakami-steward'.",
            },
            "message": {
                "type": "string",
                "description": "The request/message to send to the target profile.",
            },
            "mode": {
                "type": "string",
                "enum": ["sync"],
                "description": "Delivery mode. v1 supports only 'sync': wait for the target profile's final reply.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_TIMEOUT_SECONDS,
                "description": "Maximum seconds to wait for the target profile. Defaults to 300; capped at 1800.",
            },
        },
        "required": ["target_profile", "message"],
    },
}


def _json_error(message: str, **extra: Any) -> str:
    payload = {"success": False, "error": redact_sensitive_text(str(message))}
    payload.update(extra)
    return json.dumps(payload)


def _validate_profile_name(name: str) -> str:
    profile = str(name or "").strip()
    if not profile:
        raise ValueError("target_profile is required")
    if not _PROFILE_NAME_RE.fullmatch(profile):
        raise ValueError(
            "target_profile must contain only letters, numbers, underscores, or hyphens "
            "and must start with a letter or number"
        )
    return profile


def _profile_home(profile: str, *, root: Path | None = None) -> Path:
    hermes_root = Path(root) if root is not None else get_default_hermes_root()
    if profile == "default":
        return hermes_root
    return hermes_root / "profiles" / profile


def _profile_exists(profile: str, *, root: Path | None = None) -> bool:
    home = _profile_home(profile, root=root)
    return home.exists() and home.is_dir() and (home / "config.yaml").exists()


def _available_profiles(*, root: Path | None = None) -> list[str]:
    hermes_root = Path(root) if root is not None else get_default_hermes_root()
    profiles = []
    if (hermes_root / "config.yaml").exists():
        profiles.append("default")
    profile_dir = hermes_root / "profiles"
    if profile_dir.exists():
        profiles.extend(
            sorted(
                p.name
                for p in profile_dir.iterdir()
                if p.is_dir() and _PROFILE_NAME_RE.fullmatch(p.name) and (p / "config.yaml").exists()
            )
        )
    return profiles


def _resolve_hermes_executable() -> str:
    exe = shutil.which("hermes")
    if not exe:
        raise RuntimeError("Could not find 'hermes' on PATH")
    return exe


def _build_agent_message_command(profile: str, message: str) -> list[str]:
    """Build the subprocess argv for a sync profile request.

    Keep this shell-free: callers must pass the returned list directly to
    subprocess.run/Popen with shell=False.
    """
    return [
        _resolve_hermes_executable(),
        "--profile",
        profile,
        "chat",
        "-Q",
        "-q",
        message,
    ]


def _coerce_timeout(value: Any) -> int:
    if value in (None, ""):
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        raise ValueError("timeout_seconds must be an integer")
    if timeout < 1:
        raise ValueError("timeout_seconds must be >= 1")
    return min(timeout, _MAX_TIMEOUT_SECONDS)


def _extract_session_id(output: str) -> str | None:
    match = _SESSION_ID_RE.search(output or "")
    return match.group(1) if match else None


def _strip_session_id_line(output: str) -> str:
    return _SESSION_ID_RE.sub("", output or "").strip()


def _coerce_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def agent_message_tool(args: dict, **_kw) -> str:
    """Send a direct sync request to another local Hermes profile."""
    mode = str(args.get("mode") or "sync").strip().lower()
    if mode not in _ALLOWED_MODES:
        return tool_error("agent_message v1 supports only mode='sync'")

    message = str(args.get("message") or "")
    if not message.strip():
        return tool_error("message is required")

    try:
        profile = _validate_profile_name(args.get("target_profile", ""))
        timeout = _coerce_timeout(args.get("timeout_seconds"))
    except ValueError as exc:
        return tool_error(str(exc))

    if not _profile_exists(profile):
        available = _available_profiles()
        return _json_error(
            f"Hermes profile '{profile}' was not found",
            available_profiles=available,
        )

    try:
        command = _build_agent_message_command(profile, message)
    except RuntimeError as exc:
        return _json_error(str(exc))

    env = os.environ.copy()
    env.pop("HERMES_HOME", None)

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        partial = "".join(
            part for part in [_coerce_output_text(exc.stdout), _coerce_output_text(exc.stderr)] if part
        )
        return _json_error(
            f"agent_message timed out after {timeout}s",
            target_profile=profile,
            timeout_seconds=timeout,
            partial_output=redact_sensitive_text(partial.strip())[:4000],
        )
    except OSError as exc:
        return _json_error(f"Failed to run Hermes profile '{profile}': {exc}", target_profile=profile)

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    combined = "\n".join(part.strip() for part in [stdout, stderr] if part and part.strip())
    session_id = _extract_session_id(combined)
    reply = _strip_session_id_line(stdout)
    if not reply and stderr:
        reply = _strip_session_id_line(stderr)

    payload = {
        "success": completed.returncode == 0,
        "target_profile": profile,
        "mode": "sync",
        "returncode": completed.returncode,
        "session_id": session_id,
        "reply": redact_sensitive_text(reply),
    }
    if completed.returncode != 0:
        payload["error"] = redact_sensitive_text(combined or f"Hermes exited with status {completed.returncode}")
    return json.dumps(payload)


registry.register(
    name="agent_message",
    toolset="messaging",
    schema=AGENT_MESSAGE_SCHEMA,
    handler=agent_message_tool,
    emoji="🤝",
)
