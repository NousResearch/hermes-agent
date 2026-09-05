"""Authenticated streamable-HTTP host bridge for a child CUA MCP session.

This module is intentionally not imported by ``tools.computer_use`` so lean
Hermes installs remain importable without the optional MCP server dependencies.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import secrets
from collections.abc import AsyncIterator, Sequence
from typing import Any, AsyncContextManager, Protocol

from mcp import types
from mcp.server.auth.middleware.bearer_auth import (
    BearerAuthBackend,
    RequireAuthMiddleware,
)
from mcp.server.auth.provider import AccessToken
from mcp.server.lowlevel.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import (
    TransportSecurityMiddleware,
    TransportSecuritySettings,
)
from starlette.applications import Starlette
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import HTTPConnection
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

try:
    from tools.computer_use.host_validation import validate_security_allowlists
except ImportError:
    # Standalone mode (no Hermes package installed)
    from host_validation import validate_security_allowlists


CUA_HOST_SCOPE = "cua:invoke"
_CUA_HOST_CLIENT_ID = "hermes-cua-host-bridge"


class ChildClientSession(Protocol):
    """The child ClientSession operations forwarded by this bridge."""

    async def list_tools(
        self,
        *,
        params: types.PaginatedRequestParams | None = None,
    ) -> types.ListToolsResult: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> types.CallToolResult: ...


class StaticBearerTokenVerifier:
    """Verify one configured bearer token as one least-privilege principal."""

    def __init__(self, token: str) -> None:
        try:
            token_bytes = token.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("bearer token must contain only ASCII characters") from exc
        if len(token_bytes) < 32:
            raise ValueError("bearer token must contain at least 32 bytes")
        if any(byte < 0x20 or byte == 0x7f for byte in token_bytes):
            raise ValueError("bearer token must not contain control characters")
        self._token = token_bytes

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            token_bytes = token.encode("ascii")
        except UnicodeEncodeError:
            return None
        if any(byte < 0x20 or byte == 0x7f for byte in token_bytes):
            # A control character can never be the configured token (already
            # rejected at construction); fail the comparison cleanly.
            return None
        if not secrets.compare_digest(token_bytes, self._token):
            return None
        return AccessToken(
            token=token,
            client_id=_CUA_HOST_CLIENT_ID,
            scopes=[CUA_HOST_SCOPE],
            subject=_CUA_HOST_CLIENT_ID,
        )


class _SessionManagerEndpoint:
    def __init__(self, manager_ref: list[StreamableHTTPSessionManager | None]) -> None:
        self._manager_ref = manager_ref

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        manager = self._manager_ref[0]
        if manager is None:
            raise RuntimeError("CUA host application lifespan is not running")
        await manager.handle_request(scope, receive, send)


class _TransportSecurityEndpoint:
    """Apply host/origin checks even before an MCP session is resolved."""

    def __init__(self, app: ASGIApp, settings: TransportSecuritySettings) -> None:
        self._app = app
        self._security = TransportSecurityMiddleware(settings)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        rejection = await self._security.validate_request(
            HTTPConnection(scope),
            is_post=scope.get("method") == "POST",
        )
        if rejection is not None:
            await rejection(scope, receive, send)
            return
        await self._app(scope, receive, send)


class _MCPMethodEndpoint:
    """Reject non-MCP methods outermost, before transport-security and auth checks."""

    _ALLOWED_METHODS = frozenset({"GET", "POST", "DELETE"})

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("method") not in self._ALLOWED_METHODS:
            response = Response(
                status_code=405,
                headers={"Allow": ", ".join(sorted(self._ALLOWED_METHODS))},
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)


class _NoStoreHeaderEndpoint:
    """Stamp `Cache-Control: no-store` on every bridge response.

    MCP responses carry session-scoped, desktop-derived payloads (screenshots,
    window state); no intermediary or browser may cache them.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def _send(message) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((b"cache-control", b"no-store"))
                message["headers"] = headers
            await send(message)

        await self._app(scope, receive, _send)


def create_forwarding_server(child_session: ChildClientSession) -> Server:
    """Create a low-level MCP server that serially forwards child tool RPCs."""
    child_rpc_lock = asyncio.Lock()

    async def _on_list_tools(
        ctx: Any, params: types.PaginatedRequestParams | None = None,
    ) -> types.ListToolsResult:
        async with child_rpc_lock:
            return await child_session.list_tools(params=params)

    async def _on_call_tool(
        ctx: Any, params: types.CallToolRequestParams,
    ) -> types.CallToolResult:
        async with child_rpc_lock:
            return await child_session.call_tool(params.name, params.arguments)

    server = Server(
        "hermes-cua-host-bridge",
        on_list_tools=_on_list_tools,
        on_call_tool=_on_call_tool,
    )
    return server


def create_host_bridge_app(
    *,
    child_session_context: AsyncContextManager[ChildClientSession],
    bearer_token: str,
    allowed_hosts: Sequence[str],
    allowed_origins: Sequence[str],
    # The MCP recommendation for interactive sessions is 1800s; long model
    # turns must not be reaped mid-call.
    session_idle_timeout: float = 1800,
) -> Starlette:
    """Build the authenticated, stateful streamable-HTTP CUA host app.

    The caller supplies an already-configured standard-mode child-session
    context. This factory deliberately exposes no permission-mode or bypass
    input, so HTTP clients cannot upgrade the child driver's authorization.
    """
    timeout = float(session_idle_timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("session_idle_timeout must be a finite positive number")
    hosts, origins = validate_security_allowlists(allowed_hosts, allowed_origins)

    verifier = StaticBearerTokenVerifier(bearer_token)
    manager_ref: list[StreamableHTTPSessionManager | None] = [None]
    security_settings = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )

    # Order: method filter → transport security → bearer auth → session manager.
    # Transport security runs BEFORE auth so a DNS-rebinding probe (wrong Host,
    # typically no credentials) is classified and logged as a transport violation
    # (421) rather than surfacing as auth noise (401); a correct-host request with
    # a bad token is still a clean 401.
    protected_endpoint = _MCPMethodEndpoint(
        _TransportSecurityEndpoint(
            AuthenticationMiddleware(
                RequireAuthMiddleware(
                    _NoStoreHeaderEndpoint(_SessionManagerEndpoint(manager_ref)),
                    required_scopes=[CUA_HOST_SCOPE],
                ),
                backend=BearerAuthBackend(verifier),
            ),
            security_settings,
        )
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with child_session_context as child_session:
            server = create_forwarding_server(child_session)
            manager = StreamableHTTPSessionManager(
                server,
                json_response=True,
                stateless=False,
                security_settings=security_settings,
                session_idle_timeout=timeout,
            )
            manager_ref[0] = manager
            try:
                async with manager.run():
                    yield
            finally:
                manager_ref[0] = None

    return Starlette(
        routes=[
            Route(
                "/mcp",
                protected_endpoint,
            )
        ],
        lifespan=lifespan,
    )
