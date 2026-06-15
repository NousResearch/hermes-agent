#!/usr/bin/env python3
"""In-memory OAuth client_credentials auth for remote MCP servers.

This module supports ``mcp_servers.<name>.auth: client_credentials`` without
using the interactive OAuth browser callback flow. Access tokens minted by this
provider live only on the provider instance in this process; they are never
written to the Hermes MCP OAuth token store.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_LINEAR_TOKEN_URL = "https://api.linear.app/oauth/token"
_DEFAULT_EXPIRES_IN = 3600
_DEFAULT_REFRESH_SKEW_SECONDS = 60

_SECRET_PATTERN = re.compile(
    r"(?:"
    r"Bearer\s+\S+"
    r"|access_token[=\s:]+[^\s,;\"']+"
    r"|refresh_token[=\s:]+[^\s,;\"']+"
    r"|client_secret[=\s:]+[^\s,;\"']+"
    r"|secret[=\s:]+[^\s,;\"']+"
    r"|token[=\s:]+[^\s,;\"']+"
    r"|key[=\s:]+[^\s,;\"']+"
    r"|tok_[A-Za-z0-9_\-]+"
    r")",
    re.IGNORECASE,
)


class ClientCredentialsConfigError(ValueError):
    """Raised when an MCP client_credentials config is incomplete."""


class ClientCredentialsTokenError(RuntimeError):
    """Raised when the token endpoint does not mint a usable access token."""


@dataclass(frozen=True)
class _TokenState:
    access_token: str
    expires_at: float
    token_type: str = "Bearer"


def _redact(text: Any, *explicit_secrets: str | None) -> str:
    """Return *text* with token/secret material replaced by ``[REDACTED]``."""
    redacted = str(text)
    for secret in explicit_secrets:
        if secret:
            redacted = redacted.replace(str(secret), "[REDACTED]")
    return _SECRET_PATTERN.sub("[REDACTED]", redacted)


def _cfg_str(cfg: dict[str, Any], key: str) -> str | None:
    value = cfg.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _cfg_env_value(cfg: dict[str, Any], key: str, env_key: str) -> str | None:
    value = _cfg_str(cfg, key)
    if value:
        return value
    env_name = _cfg_str(cfg, env_key)
    if not env_name:
        return None
    # Exact, user-declared lookup only. Do not infer provider-specific fallback
    # names such as LINEAR_API_KEY; that API-key lane is intentionally not
    # supported by client_credentials.
    env_value = os.environ.get(env_name)
    if env_value is None:
        return None
    env_value = str(env_value).strip()
    return env_value or None


def _is_linear_server(server_name: str, server_url: str) -> bool:
    if server_name.strip().lower() == "linear":
        return True
    host = (urlparse(server_url).hostname or "").lower()
    return host == "linear.app" or host.endswith(".linear.app")


def _resolve_token_url(server_name: str, server_url: str, cfg: dict[str, Any]) -> str:
    token_url = _cfg_str(cfg, "token_url") or _cfg_str(cfg, "token_endpoint")
    if not token_url and _is_linear_server(server_name, server_url):
        token_url = _LINEAR_TOKEN_URL
    if not token_url:
        raise ClientCredentialsConfigError(
            "mcp_servers.<name>.auth: client_credentials requires oauth.token_url "
            "(or token_endpoint) unless the server is the built-in Linear MCP host"
        )
    parsed = urlparse(token_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ClientCredentialsConfigError(
            "mcp_servers.<name>.oauth.token_url must be an http(s) URL"
        )
    return token_url


def _token_config_from_oauth(
    server_name: str,
    server_url: str,
    oauth_config: Optional[dict[str, Any]],
) -> dict[str, Any]:
    cfg = dict(oauth_config or {})
    token_url = _resolve_token_url(server_name, server_url, cfg)
    client_id = _cfg_env_value(cfg, "client_id", "client_id_env")
    client_secret = _cfg_env_value(cfg, "client_secret", "client_secret_env")
    missing = []
    if not client_id:
        missing.append("client_id")
    if not client_secret:
        missing.append("client_secret")
    if missing:
        raise ClientCredentialsConfigError(
            "mcp_servers.<name>.auth: client_credentials requires oauth."
            + " and oauth.".join(missing)
            + " (or matching *_env keys)"
        )

    token_params = cfg.get("token_params") or {}
    if not isinstance(token_params, dict):
        raise ClientCredentialsConfigError(
            "mcp_servers.<name>.oauth.token_params must be a mapping"
        )

    return {
        "token_url": token_url,
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": _cfg_str(cfg, "scope"),
        "auth_method": (_cfg_str(cfg, "auth_method") or _cfg_str(
            cfg, "token_endpoint_auth_method"
        ) or "client_secret_basic"),
        "timeout": float(cfg.get("timeout", 30.0)),
        "refresh_skew_seconds": float(
            cfg.get("refresh_skew_seconds", _DEFAULT_REFRESH_SKEW_SECONDS)
        ),
        "token_params": {str(k): str(v) for k, v in token_params.items()},
    }


class MCPClientCredentialsAuth(httpx.Auth):
    """httpx auth provider for OAuth 2.0 client_credentials.

    The provider mints tokens lazily, stores the current access token only in
    memory, refreshes before expiry, and retries once with a fresh token after
    a 401 challenge.
    """

    def __init__(
        self,
        *,
        server_name: str,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: str | None = None,
        auth_method: str = "client_secret_basic",
        timeout: float = 30.0,
        refresh_skew_seconds: float = _DEFAULT_REFRESH_SKEW_SECONDS,
        token_params: Optional[dict[str, str]] = None,
        now: Callable[[], float] = time.time,
        token_client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
    ) -> None:
        self.server_name = server_name
        self.token_url = token_url
        self.client_id = client_id
        self._client_secret = client_secret
        self.scope = scope
        self.auth_method = auth_method
        self.timeout = timeout
        self.refresh_skew_seconds = max(float(refresh_skew_seconds), 0.0)
        self.token_params = dict(token_params or {})
        self._now = now
        self._token_client_factory = token_client_factory
        self._token: _TokenState | None = None
        self._lock = asyncio.Lock()

    def auth_flow(self, request: httpx.Request):  # pragma: no cover - sync unsupported
        raise RuntimeError("MCP client_credentials auth requires an async httpx client")

    async def async_auth_flow(self, request: httpx.Request):  # type: ignore[override]
        token = await self._ensure_token()
        request.headers["Authorization"] = f"Bearer {token}"
        response = yield request
        if response.status_code == 401:
            token = await self.force_refresh()
            request.headers["Authorization"] = f"Bearer {token}"
            yield request

    async def force_refresh(self) -> str:
        """Mint a fresh access token even if the cached one is not expired."""
        async with self._lock:
            return await self._mint_token_locked()

    async def _ensure_token(self) -> str:
        token = self._token
        if token is not None and not self._token_expiring(token):
            return token.access_token
        async with self._lock:
            token = self._token
            if token is not None and not self._token_expiring(token):
                return token.access_token
            return await self._mint_token_locked()

    def _token_expiring(self, token: _TokenState) -> bool:
        return token.expires_at <= self._now() + self.refresh_skew_seconds

    def _make_token_client(self) -> httpx.AsyncClient:
        if self._token_client_factory is not None:
            return self._token_client_factory()
        return httpx.AsyncClient(timeout=self.timeout)

    async def _mint_token_locked(self) -> str:
        data = {"grant_type": "client_credentials", **self.token_params}
        if self.scope:
            data["scope"] = self.scope

        auth: tuple[str, str] | None = None
        method = self.auth_method.lower().strip()
        if method in {"client_secret_basic", "basic"}:
            auth = (self.client_id, self._client_secret)
        elif method in {"client_secret_post", "post"}:
            data["client_id"] = self.client_id
            data["client_secret"] = self._client_secret
        else:
            raise ClientCredentialsConfigError(
                "mcp_servers.<name>.oauth.auth_method must be "
                "client_secret_basic or client_secret_post"
            )

        try:
            async with self._make_token_client() as client:
                response = await client.post(
                    self.token_url,
                    data=data,
                    auth=auth,
                    headers={"Accept": "application/json"},
                )
        except Exception as exc:
            sanitized = _redact(exc, self._client_secret)
            logger.warning(
                "MCP client_credentials '%s': token mint failed: %s",
                self.server_name,
                sanitized,
            )
            raise ClientCredentialsTokenError(
                f"client_credentials token mint failed: {sanitized}"
            ) from exc

        if response.status_code >= 400:
            detail = self._response_detail(response)
            sanitized = _redact(
                f"HTTP {response.status_code}: {detail}", self._client_secret
            )
            logger.warning(
                "MCP client_credentials '%s': token mint failed: %s",
                self.server_name,
                sanitized,
            )
            raise ClientCredentialsTokenError(
                f"client_credentials token mint failed: {sanitized}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            sanitized = _redact(response.text, self._client_secret)
            logger.warning(
                "MCP client_credentials '%s': token endpoint returned invalid JSON: %s",
                self.server_name,
                sanitized,
            )
            raise ClientCredentialsTokenError(
                "client_credentials token mint failed: token endpoint returned invalid JSON"
            ) from exc

        access_token = payload.get("access_token") if isinstance(payload, dict) else None
        if not access_token:
            logger.warning(
                "MCP client_credentials '%s': token endpoint response missing access_token",
                self.server_name,
            )
            raise ClientCredentialsTokenError(
                "client_credentials token mint failed: token endpoint response missing access_token"
            )

        token_type = str(payload.get("token_type") or "Bearer")
        try:
            expires_in = int(payload.get("expires_in", _DEFAULT_EXPIRES_IN))
        except (TypeError, ValueError):
            expires_in = _DEFAULT_EXPIRES_IN
        expires_in = max(expires_in, 1)
        self._token = _TokenState(
            access_token=str(access_token),
            expires_at=self._now() + expires_in,
            token_type=token_type,
        )
        return self._token.access_token

    @staticmethod
    def _response_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text
        if not isinstance(payload, dict):
            return str(payload)
        parts = []
        for key in ("error", "error_description", "message"):
            value = payload.get(key)
            if value:
                parts.append(f"{key}={value}")
        return "; ".join(parts) if parts else str(payload)


def build_client_credentials_auth(
    server_name: str,
    server_url: str,
    oauth_config: Optional[dict[str, Any]] = None,
) -> MCPClientCredentialsAuth:
    """Build an in-memory client_credentials auth provider for an MCP server."""
    resolved = _token_config_from_oauth(server_name, server_url, oauth_config)
    return MCPClientCredentialsAuth(server_name=server_name, **resolved)


@dataclass
class _ClientCredentialsEntry:
    server_url: str
    fingerprint: tuple[Any, ...]
    provider: MCPClientCredentialsAuth


class MCPClientCredentialsManager:
    """Process-local cache for client_credentials providers.

    The cache keeps refreshable in-memory tokens across reconnects inside the
    current Hermes process. It never writes rotating access tokens to disk.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _ClientCredentialsEntry] = {}
        self._entries_lock = threading.Lock()

    def get_or_build_provider(
        self,
        server_name: str,
        server_url: str,
        oauth_config: Optional[dict[str, Any]],
    ) -> MCPClientCredentialsAuth:
        resolved = _token_config_from_oauth(server_name, server_url, oauth_config)
        fingerprint = self._fingerprint(server_url, resolved)
        with self._entries_lock:
            entry = self._entries.get(server_name)
            if entry is None or entry.fingerprint != fingerprint:
                entry = _ClientCredentialsEntry(
                    server_url=server_url,
                    fingerprint=fingerprint,
                    provider=MCPClientCredentialsAuth(
                        server_name=server_name,
                        **resolved,
                    ),
                )
                self._entries[server_name] = entry
            return entry.provider

    async def handle_401(self, server_name: str) -> bool:
        with self._entries_lock:
            entry = self._entries.get(server_name)
        if entry is None:
            return False
        try:
            await entry.provider.force_refresh()
            return True
        except Exception as exc:
            logger.warning(
                "MCP client_credentials '%s': 401 refresh failed: %s",
                server_name,
                _redact(exc, entry.provider._client_secret),
            )
            return False

    @staticmethod
    def _fingerprint(server_url: str, resolved: dict[str, Any]) -> tuple[Any, ...]:
        token_params = tuple(sorted((resolved.get("token_params") or {}).items()))
        return (
            server_url,
            resolved.get("token_url"),
            resolved.get("client_id"),
            resolved.get("client_secret"),
            resolved.get("scope"),
            resolved.get("auth_method"),
            resolved.get("timeout"),
            resolved.get("refresh_skew_seconds"),
            token_params,
        )

    def remove(self, server_name: str) -> None:
        with self._entries_lock:
            self._entries.pop(server_name, None)


_MANAGER: MCPClientCredentialsManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_client_credentials_manager() -> MCPClientCredentialsManager:
    """Return the process-wide client_credentials manager."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = MCPClientCredentialsManager()
        return _MANAGER


def reset_client_credentials_manager_for_tests() -> None:
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = None
