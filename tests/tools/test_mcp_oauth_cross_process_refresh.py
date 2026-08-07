"""Cross-process single-flight coverage for rotating MCP OAuth refresh tokens.

Two Hermes processes may share one ``HERMES_HOME``. Providers such as Notion
rotate refresh tokens, so both processes must not submit the same token: the
second replay can revoke the entire grant. These tests use two independent SDK
provider instances and the real on-disk token store to exercise that boundary.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK OAuth support is required")

from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
)
from pydantic import AnyUrl

from tools.mcp_oauth import HermesTokenStorage
from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests


pytestmark = pytest.mark.skipif(
    _HERMES_PROVIDER_CLS is None,
    reason="MCP SDK OAuth support is required",
)

_SERVER_URL = "https://mcp.example.com/mcp"
_TOKEN_URL = "https://auth.example.com/oauth/token"
_CALLBACK = AnyUrl("http://127.0.0.1:12345/callback")


async def _noop_redirect(_url: str) -> None:
    return None


async def _noop_callback() -> tuple[str, str | None]:
    raise AssertionError("interactive callback must not run during refresh")


async def _seed_expired_grant(storage: HermesTokenStorage) -> None:
    await storage.set_tokens(
        OAuthToken(
            access_token="old-access",
            token_type="Bearer",
            expires_in=0,
            refresh_token="old-refresh",
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[_CALLBACK],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )
    storage.save_oauth_metadata(
        OAuthMetadata.model_validate(
            {
                "issuer": "https://auth.example.com",
                "authorization_endpoint": "https://auth.example.com/oauth/authorize",
                "token_endpoint": _TOKEN_URL,
                "response_types_supported": ["code"],
            }
        )
    )


def _provider(storage: HermesTokenStorage):
    assert _HERMES_PROVIDER_CLS is not None
    return _HERMES_PROVIDER_CLS(
        server_name="notion",
        server_url=_SERVER_URL,
        client_metadata=OAuthClientMetadata(
            redirect_uris=[_CALLBACK],
            client_name="Hermes Agent",
        ),
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )


def _refresh_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        json={
            "access_token": "new-access",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "new-refresh",
        },
    )


async def _finish(flow, request: httpx.Request) -> None:
    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx.Response(200, request=request))


async def _drive_flow(
    flow,
    outbound: asyncio.Queue[httpx.Request],
    inbound: asyncio.Queue[httpx.Response],
) -> None:
    """Drive every generator operation from one task (AnyIO lock ownership)."""
    try:
        request = await flow.__anext__()
        while True:
            await outbound.put(request)
            response = await inbound.get()
            try:
                request = await flow.asend(response)
            except StopAsyncIteration:
                return
    finally:
        await flow.aclose()


async def _stop_driver(task: asyncio.Task[None]) -> None:
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_rotating_refresh_is_single_flight_across_provider_instances(
    tmp_path, monkeypatch
):
    """The waiter adopts the winner's rotated grant instead of replaying it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage_a = HermesTokenStorage("notion")
    storage_b = HermesTokenStorage("notion")
    await _seed_expired_grant(storage_a)

    provider_a = _provider(storage_a)
    provider_b = _provider(storage_b)
    flow_a = provider_a.async_auth_flow(httpx.Request("POST", _SERVER_URL))
    flow_b = provider_b.async_auth_flow(httpx.Request("POST", _SERVER_URL))

    first_refresh = await flow_a.__anext__()
    assert str(first_refresh.url) == _TOKEN_URL
    assert b"old-refresh" in first_refresh.content

    outbound_b: asyncio.Queue[httpx.Request] = asyncio.Queue()
    inbound_b: asyncio.Queue[httpx.Response] = asyncio.Queue()
    driver_b = asyncio.create_task(_drive_flow(flow_b, outbound_b, inbound_b))
    await asyncio.sleep(0.05)

    try:
        assert outbound_b.empty(), (
            "a second provider must wait while the rotating refresh token is in flight"
        )

        first_request = await flow_a.asend(_refresh_response(first_refresh))
        assert str(first_request.url) == _SERVER_URL
        assert first_request.headers["authorization"] == "Bearer new-access"

        second_request = await asyncio.wait_for(outbound_b.get(), timeout=2)
        assert str(second_request.url) == _SERVER_URL, (
            "the waiter must reload the winner's grant, not emit a second refresh request"
        )
        assert second_request.headers["authorization"] == "Bearer new-access"

        await _finish(flow_a, first_request)
        await inbound_b.put(httpx.Response(200, request=second_request))
        await asyncio.wait_for(driver_b, timeout=2)
    finally:
        await flow_a.aclose()
        await _stop_driver(driver_b)

    stored = await storage_a.get_tokens()
    assert stored is not None
    assert stored.access_token == "new-access"
    assert stored.refresh_token == "new-refresh"


@pytest.mark.asyncio
async def test_abandoned_refresh_releases_cross_process_lock(tmp_path, monkeypatch):
    """Disconnect/cancellation cannot strand the per-grant refresh lock."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage_a = HermesTokenStorage("notion")
    storage_b = HermesTokenStorage("notion")
    await _seed_expired_grant(storage_a)

    flow_a = _provider(storage_a).async_auth_flow(httpx.Request("POST", _SERVER_URL))
    flow_b = _provider(storage_b).async_auth_flow(httpx.Request("POST", _SERVER_URL))

    first_refresh = await flow_a.__anext__()
    assert str(first_refresh.url) == _TOKEN_URL

    outbound_b: asyncio.Queue[httpx.Request] = asyncio.Queue()
    inbound_b: asyncio.Queue[httpx.Response] = asyncio.Queue()
    driver_b = asyncio.create_task(_drive_flow(flow_b, outbound_b, inbound_b))
    await asyncio.sleep(0.05)
    assert outbound_b.empty()

    try:
        await flow_a.aclose()
        second_refresh = await asyncio.wait_for(outbound_b.get(), timeout=2)
        assert str(second_refresh.url) == _TOKEN_URL

        await inbound_b.put(_refresh_response(second_refresh))
        second_request = await asyncio.wait_for(outbound_b.get(), timeout=2)
        assert second_request.headers["authorization"] == "Bearer new-access"
        await inbound_b.put(httpx.Response(200, request=second_request))
        await asyncio.wait_for(driver_b, timeout=2)
    finally:
        await flow_a.aclose()
        await _stop_driver(driver_b)
