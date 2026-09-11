"""Regression coverage for MCP OAuth delegated-generator lifecycle ownership."""

from __future__ import annotations

import asyncio

import anyio
import httpx
import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK OAuth support required")


async def _noop_redirect(_url: str) -> None:
    return None


async def _noop_callback():
    raise AssertionError("callback must not run in generator lifecycle tests")


async def _provider(tmp_path, monkeypatch):
    from mcp.shared.auth import (
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthToken,
    )
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="access",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="refresh",
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )
    return _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            client_name="Hermes Agent",
        ),
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )


@pytest.mark.asyncio
async def test_outer_close_closes_inner_exactly_once(tmp_path, monkeypatch):
    from mcp.client.auth.oauth2 import OAuthClientProvider

    provider = await _provider(tmp_path, monkeypatch)
    close_calls = 0

    class Inner:
        async def __anext__(self):
            return httpx.Request("POST", "https://example.com/mcp")

        async def asend(self, _response):
            raise AssertionError("flow is intentionally closed before response")

        async def aclose(self):
            nonlocal close_calls
            close_calls += 1

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", lambda self, request: Inner())
    flow = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    await flow.__anext__()
    await flow.aclose()

    assert close_calls == 1


@pytest.mark.asyncio
async def test_natural_completion_closes_inner_flow(tmp_path, monkeypatch):
    from mcp.client.auth.oauth2 import OAuthClientProvider

    provider = await _provider(tmp_path, monkeypatch)
    close_calls = 0

    class Inner:
        async def __anext__(self):
            return httpx.Request("POST", "https://example.com/mcp")

        async def asend(self, _response):
            raise StopAsyncIteration

        async def aclose(self):
            nonlocal close_calls
            close_calls += 1

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", lambda self, request: Inner())
    flow = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    await flow.__anext__()

    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx.Response(200))

    assert close_calls == 1


@pytest.mark.asyncio
async def test_natural_completion_surfaces_cleanup_failure(tmp_path, monkeypatch):
    from mcp.client.auth.oauth2 import OAuthClientProvider

    provider = await _provider(tmp_path, monkeypatch)

    class Inner:
        async def __anext__(self):
            return httpx.Request("POST", "https://example.com/mcp")

        async def asend(self, _response):
            raise StopAsyncIteration

        async def aclose(self):
            raise ValueError("cleanup failure")

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", lambda self, request: Inner())
    flow = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    await flow.__anext__()

    with pytest.raises(ValueError, match="cleanup failure"):
        await flow.asend(httpx.Response(200))


@pytest.mark.asyncio
async def test_cancellation_closes_inner_flow(tmp_path, monkeypatch):
    from mcp.client.auth.oauth2 import OAuthClientProvider

    provider = await _provider(tmp_path, monkeypatch)
    closed = asyncio.Event()

    class Inner:
        async def __anext__(self):
            return httpx.Request("POST", "https://example.com/mcp")

        async def asend(self, _response):
            raise asyncio.CancelledError()

        async def aclose(self):
            closed.set()

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", lambda self, request: Inner())
    flow = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    await flow.__anext__()

    with pytest.raises(asyncio.CancelledError):
        await flow.asend(httpx.Response(200))

    assert closed.is_set()


@pytest.mark.asyncio
async def test_active_anyio_cancellation_cannot_skip_inner_close(tmp_path, monkeypatch):
    from mcp.client.auth.oauth2 import OAuthClientProvider

    provider = await _provider(tmp_path, monkeypatch)
    closed = asyncio.Event()

    class Inner:
        async def __anext__(self):
            return httpx.Request("POST", "https://example.com/mcp")

        async def asend(self, _response):
            await anyio.sleep_forever()

        async def aclose(self):
            await anyio.sleep(0)
            closed.set()

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", lambda self, request: Inner())
    flow = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    await flow.__anext__()

    with anyio.CancelScope() as cancel_scope:
        cancel_scope.cancel()
        await flow.asend(httpx.Response(200))

    assert closed.is_set()


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_mask_primary_flow_error(tmp_path, monkeypatch):
    from mcp.client.auth.oauth2 import OAuthClientProvider

    provider = await _provider(tmp_path, monkeypatch)

    class Inner:
        async def __anext__(self):
            return httpx.Request("POST", "https://example.com/mcp")

        async def asend(self, _response):
            raise RuntimeError("primary OAuth failure")

        async def aclose(self):
            raise ValueError("cleanup failure")

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", lambda self, request: Inner())
    flow = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    await flow.__anext__()

    with pytest.raises(RuntimeError, match="primary OAuth failure") as exc_info:
        await flow.asend(httpx.Response(500))

    assert exc_info.value.__notes__ == [
        "MCP OAuth inner auth-flow cleanup also raised ValueError"
    ]


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_mask_cancellation(tmp_path, monkeypatch):
    from mcp.client.auth.oauth2 import OAuthClientProvider

    provider = await _provider(tmp_path, monkeypatch)

    class Inner:
        async def __anext__(self):
            return httpx.Request("POST", "https://example.com/mcp")

        async def asend(self, _response):
            raise asyncio.CancelledError()

        async def aclose(self):
            raise ValueError("cleanup failure")

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", lambda self, request: Inner())
    flow = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    await flow.__anext__()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await flow.asend(httpx.Response(500))

    assert exc_info.value.__notes__ == [
        "MCP OAuth inner auth-flow cleanup also raised ValueError"
    ]


@pytest.mark.asyncio
async def test_outer_close_surfaces_cleanup_failure(tmp_path, monkeypatch):
    from mcp.client.auth.oauth2 import OAuthClientProvider

    provider = await _provider(tmp_path, monkeypatch)

    class Inner:
        async def __anext__(self):
            return httpx.Request("POST", "https://example.com/mcp")

        async def asend(self, _response):
            raise AssertionError("flow is intentionally closed before response")

        async def aclose(self):
            raise ValueError("cleanup failure")

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", lambda self, request: Inner())
    flow = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    await flow.__anext__()

    with pytest.raises(ValueError, match="cleanup failure"):
        await flow.aclose()


@pytest.mark.asyncio
async def test_resource_close_reacquires_semaphore_before_inner_teardown(
    tmp_path, monkeypatch
):
    from mcp.client.auth.oauth2 import OAuthClientProvider

    provider = await _provider(tmp_path, monkeypatch)
    request = httpx.Request("POST", "https://example.com/mcp")
    close_calls = 0
    close_saw_reacquired = False

    class Inner:
        async def __anext__(self):
            await provider.context.lock.acquire()
            return request

        async def asend(self, _response):
            raise AssertionError("flow is intentionally closed before response")

        async def aclose(self):
            nonlocal close_calls, close_saw_reacquired
            close_calls += 1
            close_saw_reacquired = provider.context.lock.value == 0
            provider.context.lock.release()

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", lambda self, request: Inner())
    flow = provider.async_auth_flow(request)

    assert await flow.__anext__() is request
    assert provider.context.lock.value == 1

    await flow.aclose()

    assert close_calls == 1
    assert close_saw_reacquired
    assert provider.context.lock.value == 1


@pytest.mark.asyncio
async def test_stale_resource_retry_closes_inner_exactly_once(tmp_path, monkeypatch):
    from mcp.client.auth.oauth2 import OAuthClientProvider
    from mcp.shared.auth import OAuthToken

    provider = await _provider(tmp_path, monkeypatch)
    request = httpx.Request("POST", "https://example.com/mcp")
    old_tokens = OAuthToken(
        access_token="old-access",
        token_type="Bearer",
        expires_in=3600,
        refresh_token="refresh",
    )
    provider.context.current_tokens = old_tokens
    provider.context.update_token_expiry(old_tokens)
    close_calls = 0

    class Inner:
        async def __anext__(self):
            await provider.context.lock.acquire()
            return request

        async def asend(self, _response):
            raise AssertionError("stale retry must not resume the old inner flow")

        async def aclose(self):
            nonlocal close_calls
            close_calls += 1
            provider.context.lock.release()

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", lambda self, request: Inner())
    flow = provider.async_auth_flow(request)

    assert await flow.__anext__() is request
    assert provider.context.lock.value == 1

    new_tokens = OAuthToken(
        access_token="new-access",
        token_type="Bearer",
        expires_in=3600,
        refresh_token="refresh",
    )
    provider.context.current_tokens = new_tokens
    provider.context.update_token_expiry(new_tokens)

    assert await flow.asend(httpx.Response(401, request=request)) is request
    assert request.headers["Authorization"] == "Bearer new-access"
    assert close_calls == 1
    assert provider.context.lock.value == 1

    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx.Response(200, request=request))

    assert close_calls == 1
