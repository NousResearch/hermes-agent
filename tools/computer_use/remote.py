"""Fail-closed configuration for authenticated remote CUA transport."""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import os
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

from agent.secret_scope import (
    current_secret_scope,
    get_secret,
    is_multiplex_active,
)

_REMOTE_TOKEN_ENV = "HERMES_CUA_REMOTE_TOKEN"
_ABSENT = object()  # sentinel: 'remote' key is absent from the config mapping


def _resolve_token(environ: Mapping[str, str]) -> str:
    """Resolve the remote CUA bearer token, honoring the profile secret scope.

    A profile-scoped token (multiplexed gateway, per-turn scope installed) wins
    over any ambient process-wide value — preventing cross-profile credential
    routing (profile A's desktop token sent to profile B's endpoint). When no
    scope is active (single-profile deployment; our fleet) or the scope does not
    provide the var, fall back to the explicit ``environ`` mapping the resolver
    receives (``os.environ`` in production, injectable for tests). An
    ``UnscopedSecretError`` (multiplexing ON, no scope installed) is a gateway
    misconfiguration that must fail closed rather than silently borrowing
    another profile's ``os.environ`` value.
    """
    if current_secret_scope() is not None:
        scoped = get_secret(_REMOTE_TOKEN_ENV)
        if scoped is not None:
            return scoped
        # Scope active but var absent. Under multiplexing, a scope miss must
        # not borrow another profile's process-wide value — fail closed. In a
        # single-profile deployment (multiplex off) the scope is just a .env
        # overlay, so fall through to the injected environ.
        if is_multiplex_active():
            return ""
        return environ.get(_REMOTE_TOKEN_ENV, "")
    if is_multiplex_active():
        raise RuntimeError(
            f"{_REMOTE_TOKEN_ENV} could not be resolved: no profile secret scope "
            f"is active while gateway multiplexing is on. The remote CUA token "
            f"read must run inside a set_secret_scope(...) block."
        )
    return environ.get(_REMOTE_TOKEN_ENV, "")


@dataclass(frozen=True)
class RemoteCuaConfig:
    url: str
    token: str = field(repr=False)


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_remote_cua_config(
    computer_use_config: Mapping[str, Any],
    *,
    permission_mode: str,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[RemoteCuaConfig]:
    """Resolve remote transport, or return None when it is not configured.

    The provider factory chooses the machine; None does not select local mode.

    A bare-host URL (empty or "/" path) is normalized to "/mcp" — the bridge serves
    a single /mcp route, so a host-only URL would 404.
    """
    raw = computer_use_config.get("remote", _ABSENT)
    if raw is _ABSENT:
        return None
    if raw is None or not isinstance(raw, Mapping):
        raise RuntimeError("remote computer use configuration must be a mapping")

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise RuntimeError("remote computer use configuration 'enabled' must be a boolean")
    if not enabled:
        return None
    if permission_mode != "standard":
        raise RuntimeError("remote computer use supports standard permission mode only")

    env = environ if environ is not None else os.environ
    token = _resolve_token(env)
    if not isinstance(token, str):
        raise RuntimeError(f"{_REMOTE_TOKEN_ENV} must contain at least 32 bytes")
    try:
        token_bytes = token.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"{_REMOTE_TOKEN_ENV} must contain only ASCII characters") from exc
    if len(token_bytes) < 32:
        raise RuntimeError(f"{_REMOTE_TOKEN_ENV} must contain at least 32 bytes")
    if any(byte < 0x20 or byte == 0x7f for byte in token_bytes):
        raise RuntimeError(f"{_REMOTE_TOKEN_ENV} must not contain control characters")

    url = raw.get("url", "")
    if not isinstance(url, str) or not url:
        raise RuntimeError("remote computer use URL is required")
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError as exc:
        raise RuntimeError("remote computer use URL is invalid") from exc

    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError("remote computer use URL must use HTTP or HTTPS")
    if not parsed.hostname:
        raise RuntimeError("remote computer use URL must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("remote computer use URL must not contain credentials")
    if parsed.query:
        raise RuntimeError("remote computer use URL must not contain a query string")
    if parsed.fragment:
        raise RuntimeError("remote computer use URL must not contain a fragment")
    if parsed.path in ("", "/"):
        # The bridge serves a single /mcp route; a bare host URL would 404.
        normalized_path = "/mcp"
    else:
        normalized_path = parsed.path
    url = urlsplit(url)._replace(path=normalized_path).geturl()
    if parsed.scheme != "https" and not _is_loopback_host(parsed.hostname):
        raise RuntimeError("remote computer use requires HTTPS for non-loopback hosts")

    return RemoteCuaConfig(url=url, token=token)
