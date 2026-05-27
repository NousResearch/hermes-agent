"""Shared helpers for tool backend selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from utils import is_truthy_value


_DEFAULT_BROWSER_PROVIDER = "local"
_DEFAULT_MODAL_MODE = "auto"
_VALID_MODAL_MODES = {"auto", "direct", "managed"}


def managed_nous_tools_enabled(*, force_fresh: bool = False) -> bool:
    """Return True when the user has paid Nous Portal service access.

    Tool Gateway availability fails closed on unknown/error entitlement.  We
    intentionally catch all exceptions and return False — never block startup.
    ``force_fresh=True`` is for interactive configuration flows that should
    reflect a just-purchased subscription or credits immediately.
    """
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        if force_fresh:
            account_info = get_nous_portal_account_info(force_fresh=True)
        else:
            account_info = get_nous_portal_account_info()
        if not account_info.logged_in:
            return False
        return account_info.paid_service_access is True
    except Exception:
        return False


def nous_tool_gateway_unavailable_message(
    capability: str = "the Nous Tool Gateway",
    *,
    force_fresh: bool = False,
) -> str:
    """Return account-aware guidance for an unavailable Nous Tool Gateway path."""
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )

        account_info = get_nous_portal_account_info(force_fresh=force_fresh)
        message = format_nous_portal_entitlement_message(
            account_info,
            capability=capability,
        )
        if message:
            return message
    except Exception:
        pass
    return (
        f"{capability} is unavailable. Run `hermes model` to refresh your "
        "Nous Portal login and billing status."
    )


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
    managed_enabled: bool | None = None,
) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend selection.

    Semantics:
    - ``direct`` means direct-only
    - ``managed`` means managed-only
    - ``auto`` prefers managed when available, then falls back to direct
    """
    requested_mode = coerce_modal_mode(modal_mode)
    normalized_mode = normalize_modal_mode(modal_mode)
    if managed_enabled is None:
        managed_enabled = managed_nous_tools_enabled()
    managed_mode_blocked = (
        requested_mode == "managed" and not managed_enabled
    )

    if normalized_mode == "managed":
        selected_backend = "managed" if managed_enabled and managed_ready else None
    elif normalized_mode == "direct":
        selected_backend = "direct" if has_direct else None
    else:
        selected_backend = "managed" if managed_enabled and managed_ready else "direct" if has_direct else None

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


def prefers_gateway(config_section: str) -> bool:
    """Return True when the user opted into the Tool Gateway for this tool.

    Reads ``<section>.use_gateway`` from config.yaml.  Never raises.
    """
    try:
        from hermes_cli.config import load_config
        section = (load_config() or {}).get(config_section)
        if isinstance(section, dict):
            return is_truthy_value(section.get("use_gateway"), default=False)
    except Exception:
        pass
    return False


def fal_key_is_configured() -> bool:
    """Return True when FAL_KEY is set to a non-whitespace value.

    Consults both ``os.environ`` and ``~/.hermes/.env`` (via
    ``hermes_cli.config.get_env_value`` when available) so tool-side
    checks and CLI setup-time checks agree.  A whitespace-only value
    is treated as unset everywhere.
    """
    value = os.getenv("FAL_KEY")
    if value is None:
        # Fall back to the .env file for CLI paths that may run before
        # dotenv is loaded into os.environ.
        try:
            from hermes_cli.config import get_env_value

            value = get_env_value("FAL_KEY")
        except Exception:
            value = None
    return bool(value and value.strip())
