"""Microsoft Graph app-only (client-credentials) authentication helpers."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_GRAPH_SCOPE = "https://graph.microsoft.com/.default"
DEFAULT_GRAPH_AUTHORITY_URL = "https://login.microsoftonline.com"
DEFAULT_TOKEN_SKEW_SECONDS = 120

_REQUIRED_ENV = ("MSGRAPH_TENANT_ID", "MSGRAPH_CLIENT_ID", "MSGRAPH_CLIENT_SECRET")


class MicrosoftGraphAuthError(RuntimeError):
    """Base class for Microsoft Graph auth failures."""


class MicrosoftGraphConfigError(MicrosoftGraphAuthError):
    """Graph credentials are missing or invalid."""


class MicrosoftGraphTokenError(MicrosoftGraphAuthError):
    """Token acquisition failed."""


def format_graph_error(error: Any) -> str | None:
    """Render Graph's ``{"error": {"code", "message"}}`` (or bare-string ``error``) as
    ``code: message``; shared by token endpoint and REST client. None if unusable."""
    if isinstance(error, str):
        return error
    if not isinstance(error, dict):
        return None
    code, message = error.get("code"), error.get("message")
    return f"{code}: {message}" if code and message else (str(message) if message else None)


@dataclass(frozen=True)
class GraphCredentials:
    """Normalized Microsoft Graph app-only credentials."""

    tenant_id: str
    client_id: str
    client_secret: str
    scope: str = DEFAULT_GRAPH_SCOPE
    authority_url: str = DEFAULT_GRAPH_AUTHORITY_URL

    @property
    def token_url(self) -> str:
        return f"{self.authority_url.rstrip('/')}/{self.tenant_id.strip().strip('/')}/oauth2/v2.0/token"

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, *, required: bool = True) -> "GraphCredentials | None":
        env = environ if environ is not None else os.environ
        values = [(env.get(name) or "").strip() for name in _REQUIRED_ENV]
        missing = [name for name, value in zip(_REQUIRED_ENV, values) if not value]
        if missing and not required:
            return None
        if missing:
            raise MicrosoftGraphConfigError(f"Missing Microsoft Graph configuration: {', '.join(missing)}")
        return cls(*values, scope=(env.get("MSGRAPH_SCOPE") or DEFAULT_GRAPH_SCOPE).strip(),
                   authority_url=(env.get("MSGRAPH_AUTHORITY_URL") or DEFAULT_GRAPH_AUTHORITY_URL).strip())


@dataclass
class CachedAccessToken:
    """Cached app-only Graph access token."""

    access_token: str
    expires_at: float
    token_type: str = "Bearer"

    def is_expired(self, *, skew_seconds: int = DEFAULT_TOKEN_SKEW_SECONDS) -> bool:
        return self.expires_at <= (time.time() + max(0, int(skew_seconds)))

    @property
    def expires_in_seconds(self) -> int:
        return max(0, int(self.expires_at - time.time()))


class MicrosoftGraphTokenProvider:
    """Acquire and cache Microsoft Graph app-only access tokens."""

    def __init__(self, credentials: GraphCredentials, *, timeout: float = 20.0,
                 skew_seconds: int = DEFAULT_TOKEN_SKEW_SECONDS,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.credentials, self.timeout, self.skew_seconds = credentials, timeout, max(0, int(skew_seconds))
        self._transport = transport
        self._cached_token: CachedAccessToken | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, **kwargs: Any) -> "MicrosoftGraphTokenProvider":
        return cls(GraphCredentials.from_env(environ), **kwargs)

    def clear_cache(self) -> None:
        self._cached_token = None

    def inspect_token_health(self) -> dict[str, Any]:
        cached, creds = self._cached_token, self.credentials
        return {"configured": True, "tenant_id": creds.tenant_id, "client_id": creds.client_id,
                "scope": creds.scope, "authority_url": creds.authority_url, "token_url": creds.token_url,
                "cached": bool(cached), "expires_in_seconds": cached.expires_in_seconds if cached else None,
                "is_expired": cached.is_expired(skew_seconds=0) if cached else None,
                "refresh_skew_seconds": self.skew_seconds}

    def _fresh_cached(self) -> CachedAccessToken | None:
        """The cached token unless it expires within ``skew_seconds``."""
        cached = self._cached_token
        return cached if cached and not cached.is_expired(skew_seconds=self.skew_seconds) else None

    async def get_access_token(self, *, force_refresh: bool = False) -> str:
        # Double-checked under the lock so concurrent callers share one fetch.
        if not force_refresh and (cached := self._fresh_cached()):
            return cached.access_token
        async with self._lock:
            if not force_refresh and (cached := self._fresh_cached()):
                return cached.access_token
            self._cached_token = await self._fetch_access_token()
            return self._cached_token.access_token

    async def _fetch_access_token(self) -> CachedAccessToken:
        data = {"grant_type": "client_credentials", "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret, "scope": self.credentials.scope}
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout), transport=self._transport) as client:
            response = await client.post(self.credentials.token_url, data=data,
                                         headers={"Content-Type": "application/x-www-form-urlencoded"})
        if response.status_code >= 400:
            raise MicrosoftGraphTokenError("Microsoft Graph token request failed with HTTP "
                                           f"{response.status_code}: {_extract_error_detail(response)}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise MicrosoftGraphTokenError("Microsoft Graph token response was not valid JSON.") from exc
        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
            raise MicrosoftGraphTokenError("Microsoft Graph token response did not include access_token.")
        try:
            expires_in_seconds = int(payload.get("expires_in"))
        except (TypeError, ValueError) as exc:
            raise MicrosoftGraphTokenError(
                "Microsoft Graph token response did not include a valid expires_in.") from exc
        return CachedAccessToken(access_token, time.time() + max(0, expires_in_seconds),
                                 str(payload.get("token_type") or "Bearer").strip() or "Bearer")


def _extract_error_detail(response: httpx.Response) -> str:
    """Best human-readable detail from a token-endpoint error body: ``error_description``,
    then the Graph-style ``error`` object/string, then a bare ``code``, then raw text."""
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "unknown error"
    if not isinstance(payload, dict):
        return str(payload)
    if isinstance(payload.get("error_description"), str):
        return payload["error_description"]
    error = payload.get("error")
    detail = format_graph_error(error)
    if detail is not None:
        return detail
    return str(error["code"]) if isinstance(error, dict) and error.get("code") else str(payload)
