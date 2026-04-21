"""Shared helpers for tool backend selection."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict


_DEFAULT_BROWSER_PROVIDER = "local"
_DEFAULT_MODAL_MODE = "auto"
_VALID_MODAL_MODES = {"auto", "direct", "managed"}
_ELEVENLABS_KEYCHAIN_SERVICES = (
    "openclaw/elevenlabs-api-key",
    "elevenlabs-api-key",
    "ELEVENLABS_API_KEY",
    "elevenlabs",
)


def _read_macos_keychain_generic_password(service: str) -> str:
    """Return a generic-password secret from macOS Keychain, or "" if missing."""
    if platform.system() != "Darwin" or not service:
        return ""

    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return ""

    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def managed_nous_tools_enabled() -> bool:
    """Return True when the user has an active paid Nous subscription.

    The Tool Gateway is available to any Nous subscriber who is NOT on
    the free tier.  We intentionally catch all exceptions and return
    False — never block the agent startup path.
    """
    try:
        from hermes_cli.auth import get_nous_auth_status

        status = get_nous_auth_status()
        if not status.get("logged_in"):
            return False

        from hermes_cli.models import check_nous_free_tier

        if check_nous_free_tier():
            return False  # free-tier users don't get gateway access
        return True
    except Exception:
        return False


def normalize_browser_cloud_provider(value: object | None) -> str:
    """Return a normalized browser provider key."""
    provider = str(value or _DEFAULT_BROWSER_PROVIDER).strip().lower()
    return provider or _DEFAULT_BROWSER_PROVIDER


def coerce_modal_mode(value: object | None) -> str:
    """Return the requested modal mode when valid, else the default."""
    mode = str(value or _DEFAULT_MODAL_MODE).strip().lower()
    if mode in _VALID_MODAL_MODES:
        return mode
    return _DEFAULT_MODAL_MODE


def normalize_modal_mode(value: object | None) -> str:
    """Return a normalized modal execution mode."""
    return coerce_modal_mode(value)


def has_direct_modal_credentials() -> bool:
    """Return True when direct Modal credentials/config are available."""
    return bool(
        (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))
        or (Path.home() / ".modal.toml").exists()
    )


def resolve_modal_backend_state(
    modal_mode: object | None,
    *,
    has_direct: bool,
    managed_ready: bool,
) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend selection.

    Semantics:
    - ``direct`` means direct-only
    - ``managed`` means managed-only
    - ``auto`` prefers managed when available, then falls back to direct
    """
    requested_mode = coerce_modal_mode(modal_mode)
    normalized_mode = normalize_modal_mode(modal_mode)
    managed_mode_blocked = (
        requested_mode == "managed" and not managed_nous_tools_enabled()
    )

    if normalized_mode == "managed":
        selected_backend = "managed" if managed_nous_tools_enabled() and managed_ready else None
    elif normalized_mode == "direct":
        selected_backend = "direct" if has_direct else None
    else:
        selected_backend = "managed" if managed_nous_tools_enabled() and managed_ready else "direct" if has_direct else None

    return {
        "requested_mode": requested_mode,
        "mode": normalized_mode,
        "has_direct": has_direct,
        "managed_ready": managed_ready,
        "managed_mode_blocked": managed_mode_blocked,
        "selected_backend": selected_backend,
    }


def resolve_openai_audio_api_key() -> str:
    """Prefer the voice-tools key, but fall back to the normal OpenAI key."""
    return (
        os.getenv("VOICE_TOOLS_OPENAI_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    ).strip()


def resolve_elevenlabs_api_key() -> str:
    """Return ElevenLabs API key from env or macOS Keychain fallback."""
    direct = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if direct:
        return direct

    for service in _ELEVENLABS_KEYCHAIN_SERVICES:
        secret = _read_macos_keychain_generic_password(service)
        if secret:
            return secret
    return ""


def prefers_gateway(config_section: str) -> bool:
    """Return True when the user opted into the Tool Gateway for this tool.

    Reads ``<section>.use_gateway`` from config.yaml.  Never raises.
    """
    try:
        from hermes_cli.config import load_config
        section = (load_config() or {}).get(config_section)
        if isinstance(section, dict):
            return bool(section.get("use_gateway"))
    except Exception:
        pass
    return False
