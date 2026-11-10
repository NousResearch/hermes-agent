"""Early MCP OAuth renewal for requests and idle, connected transports."""

import asyncio
import contextvars
import logging
import time

logger = logging.getLogger(__name__)
_wait = asyncio.sleep


def cache_issued_lifetime(provider) -> None:
    """Keep the lifetime from storage for a loaded token, whose TTL is rebased on read."""
    tokens = provider.context.current_tokens
    lifetime = getattr(tokens, "expires_in", None)
    storage = provider.context.storage
    if tokens is not None:
        from tools.mcp_oauth import HermesTokenStorage, _read_json

        if isinstance(storage, HermesTokenStorage):
            data = _read_json(storage._tokens_path())
            if (
                isinstance(data, dict)
                and data.get("access_token") == tokens.access_token
                and type(data.get("expires_in")) is int
            ):
                lifetime = data["expires_in"]
    provider._hermes_loaded_lifetime = (
        (tokens, lifetime) if tokens is not None else None
    )


def refresh_delay(provider) -> float | None:
    """Seconds until renewal, preserving the issued lifetime after a cold load."""
    context = provider.context
    tokens = context.current_tokens
    expiry = context.token_expiry_time
    if (
        tokens is None
        or tokens.expires_in is None
        or expiry is None
        or not context.can_refresh_token()
    ):
        return None
    cached = getattr(provider, "_hermes_loaded_lifetime", None)
    lifetime = (
        cached[1] if cached is not None and cached[0] is tokens else tokens.expires_in
    )
    if lifetime is None or lifetime <= 0:
        return None
    lead = min(300, max(60, lifetime * 0.1), lifetime * 0.5)
    return max(0, expiry - time.time() - lead)


def build_refresh_client(provider, request):
    """Build the SDK-flavoured, TLS-configured, body-capped token-endpoint client."""
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_errors import _make_mcp_body_cap_transport
    from tools.mcp_tool_transport import _mcp_proxy_mounts

    httpx = sdk_httpx()
    verify = getattr(provider, "_hermes_ssl_verify", True)
    client_cert = getattr(provider, "_hermes_client_cert", None)
    transport_options = {"verify": verify}
    if client_cert is not None:
        transport_options["cert"] = client_cert
    direct_transport = httpx.AsyncHTTPTransport(**transport_options)
    direct_transport = _make_mcp_body_cap_transport(httpx, direct_transport)
    mounts = _mcp_proxy_mounts(
        httpx,
        str(request.url),
        verify,
        client_cert,
        getattr(provider, "_hermes_server_name", ""),
    )
    client_options = {
        "timeout": 15.0,
        "follow_redirects": True,
        "transport": direct_transport,
    }
    if mounts is not None:
        client_options["mounts"] = mounts
    return httpx.AsyncClient(**client_options)


async def refresh_if_due(provider, *, initialize: bool = False) -> bool:
    """Renew under both existing locks, without entering browser authorization.

    Expired request-path grants stay with the SDK's reactive flow. The idle
    worker may also try an expired grant after a delayed event-loop wake.
    """
    from tools.mcp_oauth_provider import _RefreshCompletedByPeer
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_oauth import RefreshFenceTimeout

    httpx = sdk_httpx()
    transient_errors = (
        httpx.HTTPError,
        OSError,
        RefreshFenceTimeout,
    )
    context = provider.context
    if provider._initialized:
        fast_delay = refresh_delay(provider)
        if fast_delay is None:
            return False
        if fast_delay > 0:
            return True

    async with context.lock:
        try:
            if initialize and not provider._initialized:
                await provider._initialize()
            delay = refresh_delay(provider)
            if delay is None:
                return False
            if delay > 0 or (initialize and not context.is_token_valid()):
                return True

            request = await provider._refresh_token()
            async with build_refresh_client(provider, request) as client:
                response = await client.send(request)

            prior_tokens = context.current_tokens
            prior_expiry = context.token_expiry_time
            await provider._maybe_flag_poisoned_client(response)
            refreshed = await provider._handle_refresh_response(response)
            if refreshed:
                return True

            status = response.status_code
            context.current_tokens = prior_tokens
            context.token_expiry_time = prior_expiry
            if status == 429 or 500 <= status < 600:
                raise ConnectionError(
                    "MCP OAuth token endpoint temporarily unavailable"
                )
            if not context.is_token_valid():
                context.clear_tokens()
            provider._initialized = False
            return False
        except _RefreshCompletedByPeer:
            return True
        except transient_errors:
            if initialize and context.is_token_valid():
                return True
            raise
        finally:
            await provider._hermes_release_refresh_fence()


async def maintain_tokens(provider) -> None:
    """One connection-owned waiter, cancelled and awaited at disconnect."""
    from tools.mcp_oauth import RefreshFenceTimeout
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    transient_errors = (
        httpx.HTTPError,
        OSError,
        RefreshFenceTimeout,
    )
    failures = 0
    retry_delay = None
    while True:
        delay = refresh_delay(provider)
        if delay is None:
            return
        await _wait(max(1.0, retry_delay if retry_delay is not None else delay))
        retry_delay = None
        try:
            keep_running = await refresh_if_due(provider)
        except transient_errors as exc:
            failures += 1
            server_name = getattr(provider, "_hermes_server_name", "") or "unknown"
            if failures >= 5:
                logger.warning(
                    "MCP OAuth '%s': background renewal failed (%s); stopping after five consecutive errors",
                    server_name,
                    type(exc).__name__,
                )
                return
            expiry = provider.context.token_expiry_time
            remaining = expiry - time.time() if expiry is not None else 0
            retry_delay = min(60.0, max(1.0, remaining / 3))
            logger.warning(
                "MCP OAuth '%s': background renewal failed (%s); retrying in %.0f seconds",
                server_name,
                type(exc).__name__,
                retry_delay,
            )
            continue
        if not keep_running:
            return
        failures = 0


def start_token_maintenance(server) -> list[asyncio.Task]:
    """Attach renewal to the active OAuth connection and its profile context."""
    provider = getattr(server, "_oauth_auth", None)
    if server._auth_type != "oauth" or provider is None:
        return []
    context = contextvars.copy_context()
    hermes_home = getattr(provider, "_hermes_home", "")
    if hermes_home:
        from hermes_constants import set_hermes_home_override

        context.run(set_hermes_home_override, hermes_home)
    return [asyncio.create_task(maintain_tokens(provider), context=context)]
