"""Helpers for reading Tenki CLI configuration without exposing secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from utils import fast_safe_load

TENKI_DEFAULT_API_ENDPOINT = "https://api.tenki.cloud"

_SECRET_KEYS = frozenset({
    "auth_token",
    "api_key",
    "access_token",
    "session_token",
    "token",
})

_SDK_AUTH_PREFIXES = ("cookie:", "ory_st_", "sk-")


def tenki_cli_config_path() -> Path:
    """Return the Tenki CLI config path.

    ``TENKI_CONFIG_PATH`` is honored for tests and uncommon CLI installs.
    """
    override = os.getenv("TENKI_CONFIG_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "tenki" / "config.yaml"


def load_tenki_cli_config() -> dict[str, Any]:
    """Load Tenki CLI config, returning ``{}`` on missing or invalid files."""
    path = tenki_cli_config_path()
    try:
        data = fast_safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _scoped_env(name: str) -> str:
    """Read a credential env var honoring the active profile secret scope.

    Under a multiplexed gateway turn a profile scope is installed, and the
    token must come from *that* profile's secrets — never from a raw
    ``os.environ`` read that could hold another profile's value. When no
    multiplexing is active this behaves exactly like ``os.getenv``.
    """
    try:
        from agent.secret_scope import get_secret

        return _string(get_secret(name, ""))
    except Exception:
        # Fail closed: an unscoped read under active multiplexing (or any
        # resolution error) must NOT silently leak a process-global value.
        return ""


def _global_credential_fallback_allowed() -> bool:
    """Whether machine-global credential sources (the shared Tenki CLI login)
    may be consulted.

    Skipped whenever a profile secret scope is authoritative — a multiplexed
    profile without its own Tenki token must not borrow the machine-global
    ``tenki login`` credential that another profile may be relying on.
    """
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        return current_secret_scope() is None and not is_multiplex_active()
    except Exception:
        return True


def _first_string(data: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _string(data.get(key))
        if value:
            return value
    return ""


def _normalize_cli_auth_token(secret: str, key: str = "") -> str:
    """Return a Tenki SDK-compatible auth token from Tenki CLI config.

    Tenki CLI v0.6 stores its browser session cookie as a bare ``auth_token``.
    The Python SDK expects cookie credentials to be prefixed with ``cookie:``;
    otherwise it sends the value as a bearer token and the API returns
    ``sandbox: unauthorized``.
    """
    secret = _string(secret)
    if not secret or secret.startswith(_SDK_AUTH_PREFIXES):
        return secret
    if key.lower() == "auth_token":
        return f"cookie:{secret}"
    return secret


def _find_secret_value(data: Any) -> str:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(key, str) and key.lower() in _SECRET_KEYS:
                secret = _string(value)
                if secret:
                    return _normalize_cli_auth_token(secret, key)
            found = _find_secret_value(value)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_secret_value(item)
            if found:
                return found
    return ""


def resolve_tenki_api_endpoint(explicit: str = "") -> str:
    """Resolve the Tenki API endpoint from config/env/CLI defaults.

    Scope-aware (see :func:`_scoped_env`): under a multiplexed profile turn the
    active profile's setting wins, and the shared machine Tenki CLI config is
    consulted only when no profile scope is authoritative.
    """
    explicit = _string(explicit)
    if explicit:
        return explicit
    for env_name in ("TENKI_API_ENDPOINT", "TENKI_API_URL"):
        value = _scoped_env(env_name)
        if value:
            return value
    if _global_credential_fallback_allowed():
        cfg = load_tenki_cli_config()
        endpoint = _first_string(cfg, ("api_endpoint", "api_url", "endpoint"))
        if endpoint:
            return endpoint
    return TENKI_DEFAULT_API_ENDPOINT


def resolve_tenki_workspace_id(explicit: str = "") -> str:
    """Resolve the Tenki workspace id. Scope-aware; workspace/project decide
    where sandboxes are created, so a multiplexed profile must not silently
    borrow the machine-global workspace of another tenant."""
    explicit = _string(explicit)
    if explicit:
        return explicit
    for env_name in ("TENKI_WORKSPACE_ID", "TENKI_WORKSPACE"):
        value = _scoped_env(env_name)
        if value:
            return value
    if not _global_credential_fallback_allowed():
        return ""
    return _first_string(load_tenki_cli_config(), ("current_workspace_id", "workspace_id", "workspace"))


def resolve_tenki_project_id(explicit: str = "") -> str:
    """Resolve the Tenki project id. Scope-aware for the same reason as
    :func:`resolve_tenki_workspace_id`."""
    explicit = _string(explicit)
    if explicit:
        return explicit
    for env_name in ("TENKI_PROJECT_ID", "TENKI_PROJECT"):
        value = _scoped_env(env_name)
        if value:
            return value
    if not _global_credential_fallback_allowed():
        return ""
    return _first_string(load_tenki_cli_config(), ("current_project_id", "project_id", "project"))


def resolve_tenki_auth_token(explicit: str = "") -> str:
    """Resolve a Tenki auth token/API key without logging or persisting it.

    Reads are profile-scope-aware (see :func:`_scoped_env`): under a
    multiplexed gateway turn the active profile's secrets win, and the shared
    machine ``tenki login`` credential is consulted only when no profile scope
    is authoritative.
    """
    explicit = _string(explicit)
    if explicit:
        return explicit
    for env_name in ("TENKI_AUTH_TOKEN", "TENKI_API_KEY"):
        value = _scoped_env(env_name)
        if value:
            return value
    if not _global_credential_fallback_allowed():
        return ""
    return _find_secret_value(load_tenki_cli_config())


def has_tenki_auth() -> bool:
    return bool(resolve_tenki_auth_token())
