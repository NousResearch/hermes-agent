"""Microsoft Graph app-only authentication helpers."""

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


class MicrosoftGraphAuthError(RuntimeError):
    """Base class for Microsoft Graph auth failures."""


class MicrosoftGraphConfigError(MicrosoftGraphAuthError):
    """Raised when Graph credentials are missing or invalid."""


class MicrosoftGraphTokenError(MicrosoftGraphAuthError):
    """Raised when token acquisition fails."""


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
        base = self.authority_url.rstrip("/")
        tenant = self.tenant_id.strip().strip("/")
        return f"{base}/{tenant}/oauth2/v2.0/token"

    @classmethod
    def from_env(
        cls,
        environ: dict[str, str] | None = None,
        *,
        required: bool = True,
    ) -> "GraphCredentials | None":
        env = environ if environ is not None else os.environ
        tenant_id = (env.get("MSGRAPH_TENANT_ID") or "").strip()
        client_id = (env.get("MSGRAPH_CLIENT_ID") or "").strip()
        client_secret = (env.get("MSGRAPH_CLIENT_SECRET") or "").strip()
        scope = (env.get("MSGRAPH_SCOPE") or DEFAULT_GRAPH_SCOPE).strip()
        authority_url = (
            env.get("MSGRAPH_AUTHORITY_URL") or DEFAULT_GRAPH_AUTHORITY_URL
        ).strip()

        missing = [
            name
            for name, value in (
                ("MSGRAPH_TENANT_ID", tenant_id),
                ("MSGRAPH_CLIENT_ID", client_id),
                ("MSGRAPH_CLIENT_SECRET", client_secret),
            )
            if not value
        ]
        if missing:
            if not required:
                return None
            raise MicrosoftGraphConfigError(
                f"Missing Microsoft Graph configuration: {', '.join(missing)}"
            )

        return cls(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            scope=scope,
            authority_url=authority_url,
        )


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

    def __init__(
        self,
        credentials: GraphCredentials,
        *,
        timeout: float = 20.0,
        skew_seconds: int = DEFAULT_TOKEN_SKEW_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.credentials = credentials
        self.timeout = timeout
        self.skew_seconds = max(0, int(skew_seconds))
        self._transport = transport
        self._cached_token: CachedAccessToken | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(
        cls,
        environ: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> "MicrosoftGraphTokenProvider":
        credentials = GraphCredentials.from_env(environ)
        return cls(credentials, **kwargs)

    def clear_cache(self) -> None:
        self._cached_token = None

    def inspect_token_health(self) -> dict[str, Any]:
        cached = self._cached_token
        return {
            "configured": True,
            "tenant_id": self.credentials.tenant_id,
            "client_id": self.credentials.client_id,
            "scope": self.credentials.scope,
            "authority_url": self.credentials.authority_url,
            "token_url": self.credentials.token_url,
            "cached": bool(cached),
            "expires_in_seconds": cached.expires_in_seconds if cached else None,
            "is_expired": cached.is_expired(skew_seconds=0) if cached else None,
            "refresh_skew_seconds": self.skew_seconds,
        }

    async def get_access_token(self, *, force_refresh: bool = False) -> str:
        cached = self._cached_token
        if not force_refresh and cached and not cached.is_expired(
            skew_seconds=self.skew_seconds
        ):
            return cached.access_token

        async with self._lock:
            cached = self._cached_token
            if not force_refresh and cached and not cached.is_expired(
                skew_seconds=self.skew_seconds
            ):
                return cached.access_token

            token = await self._fetch_access_token()
            self._cached_token = token
            return token.access_token

    async def _fetch_access_token(self) -> CachedAccessToken:
        data = {
            "grant_type": "client_credentials",
            "client_id": self.credentials.client_id,
            "client_secret": self.credentials.client_secret,
            "scope": self.credentials.scope,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            transport=self._transport,
        ) as client:
            response = await client.post(
                self.credentials.token_url,
                data=data,
                headers=headers,
            )

        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            raise MicrosoftGraphTokenError(
                "Microsoft Graph token request failed with HTTP "
                f"{response.status_code}: {detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise MicrosoftGraphTokenError(
                "Microsoft Graph token response was not valid JSON."
            ) from exc

        access_token = str(payload.get("access_token") or "").strip()
        token_type = str(payload.get("token_type") or "Bearer").strip() or "Bearer"
        expires_in = payload.get("expires_in")

        if not access_token:
            raise MicrosoftGraphTokenError(
                "Microsoft Graph token response did not include access_token."
            )

        try:
            expires_in_seconds = int(expires_in)
        except (TypeError, ValueError) as exc:
            raise MicrosoftGraphTokenError(
                "Microsoft Graph token response did not include a valid expires_in."
            ) from exc

        return CachedAccessToken(
            access_token=access_token,
            token_type=token_type,
            expires_at=time.time() + max(0, expires_in_seconds),
        )


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text or "unknown error"

    if isinstance(payload, dict):
        if isinstance(payload.get("error_description"), str):
            return payload["error_description"]
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            code = error.get("code")
            if message and code:
                return f"{code}: {message}"
            if message:
                return str(message)
            if code:
                return str(code)
        if isinstance(error, str):
            return error
    return str(payload)
