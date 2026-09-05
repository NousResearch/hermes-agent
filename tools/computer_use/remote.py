"""Fail-closed configuration for authenticated remote CUA transport."""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import os
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

_REMOTE_TOKEN_ENV = "HERMES_CUA_REMOTE_TOKEN"


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
    """Resolve and validate remote CUA configuration, or return local mode.

    A bare-host URL (empty or "/" path) is normalized to "/mcp" — the bridge serves
    a single /mcp route, so a host-only URL would 404.
    """
    raw = computer_use_config.get("remote")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RuntimeError("remote computer use configuration must be a mapping")

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise RuntimeError("remote computer use configuration 'enabled' must be a boolean")
    if not enabled:
        return None
    if permission_mode != "standard":
        raise RuntimeError("remote computer use supports standard permission mode only")

    env = environ if environ is not None else os.environ
    token = env.get(_REMOTE_TOKEN_ENV, "")
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
