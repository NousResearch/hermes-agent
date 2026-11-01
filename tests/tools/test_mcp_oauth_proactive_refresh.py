"""Proactive renewal of expiring MCP grants, including an idle connection."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
)

from tools.mcp_oauth import HermesTokenStorage
from tools.mcp_oauth_manager import HermesMCPOAuthProvider
from tools.mcp_tool import sdk_httpx


async def _provider(tmp_path, monkeypatch, ttl=3600):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("srv", hermes_home=tmp_path)
    await storage.set_tokens(
        OAuthToken(
            access_token="old",
            token_type="Bearer",
            refresh_token="refresh-old",
            expires_in=ttl,
        )
    )
    await storage.set_client_info(OAuthClientInformationFull(client_id="client"))
    storage.save_oauth_metadata(
        OAuthMetadata(
            issuer="https://auth.example",
            authorization_endpoint="https://auth.example/authorize",
            token_endpoint="https://auth.example/token",
            response_types_supported=["code"],
        )
    )
    provider = HermesMCPOAuthProvider(
        server_name="srv",
        server_url="https://mcp.example/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=["http://localhost/callback"]
        ),
        storage=storage,
        redirect_handler=AsyncMock(
            side_effect=AssertionError("browser must stay closed")
        ),
        callback_handler=AsyncMock(
            side_effect=AssertionError("browser must stay closed")
        ),
    )
    await provider._initialize()
    return provider


def _mock_refresh_client(monkeypatch, handler):
    from tools import mcp_oauth_refresh as refresh

    httpx = sdk_httpx()

    monkeypatch.setattr(
        refresh,
        "build_refresh_client",
        lambda provider, request: httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ),
    )


def _success_response(httpx, calls, *, expires_in=3600):
    calls.append(True)
    return httpx.Response(
        200,
        json={
            "access_token": f"new-{len(calls)}",
            "token_type": "Bearer",
            "refresh_token": f"refresh-{len(calls)}",
            "expires_in": expires_in,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining,expected", [(200, 1), (1800, 0)])
async def test_request_refreshes_only_inside_window(
    tmp_path, monkeypatch, remaining, expected
):
    import time

    provider = await _provider(tmp_path, monkeypatch)
    provider.context.token_expiry_time = time.time() + remaining
    calls = []
    httpx = sdk_httpx()

    async def handle(request):
        return _success_response(httpx, calls)

    _mock_refresh_client(monkeypatch, handle)
    request = httpx.Request("POST", "https://mcp.example/mcp")
    flow = provider.async_auth_flow(request)
    try:
        assert await anext(flow) is request
        assert len(calls) == expected
        assert request.headers["Authorization"] == (
            "Bearer new-1" if expected else "Bearer old"
        )
    finally:
        await flow.aclose()


@pytest.mark.asyncio
async def test_cold_first_request_uses_issued_lifetime_inside_lead_window(
    tmp_path, monkeypatch
):
    import time
    from tools import mcp_oauth as oauth

    provider = await _provider(tmp_path, monkeypatch)
    stored = oauth._read_json(provider.context.storage._tokens_path())
    stored["expires_at"] = time.time() + 200
    oauth._write_json(provider.context.storage._tokens_path(), stored)
    provider._initialized = False

    httpx = sdk_httpx()
    calls = []

    async def handle(request):
        return _success_response(httpx, calls)

    _mock_refresh_client(monkeypatch, handle)
    request = httpx.Request("POST", "https://mcp.example/mcp")
    flow = provider.async_auth_flow(request)
    try:
        assert await anext(flow) is request
        assert len(calls) == 1
        assert request.headers["Authorization"] == "Bearer new-1"
    finally:
        await flow.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "no_proxy"),
    [
        ("https://10.2.3.4/token", "10.0.0.0/8"),
        ("https://mcp.example.test/token", "*.example.test"),
    ],
)
async def test_sdk_refresh_client_honours_no_proxy_and_tls_without_network(
    monkeypatch, url, no_proxy
):
    from urllib import request as urllib_request
    from types import SimpleNamespace
    from tools import mcp_oauth_refresh as refresh

    httpx = sdk_httpx()
    monkeypatch.setenv("NO_PROXY", no_proxy)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr(
        urllib_request,
        "getproxies",
        lambda: {
            "http": "http://proxy.example:8080",
            "https": "http://proxy.example:8080",
        },
    )
    monkeypatch.setattr(urllib_request, "proxy_bypass", lambda host: False)
    cert = ("/tmp/client.pem", "/tmp/client-key.pem")
    provider = SimpleNamespace(
        _hermes_ssl_verify=False,
        _hermes_client_cert=cert,
        _hermes_server_name="srv",
    )
    transport_options, delivered = [], []

    def transport(**kwargs):
        transport_options.append(kwargs)
        return httpx.MockTransport(
            lambda request: delivered.append(request) or httpx.Response(200)
        )

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    request = httpx.Request("POST", url)
    async with refresh.build_refresh_client(provider, request) as client:
        response = await client.send(request)

    assert response.status_code == 200
    assert [str(item.url) for item in delivered] == [url]
    assert transport_options == [{"verify": False, "cert": cert}]


def test_managed_oauth_provider_keeps_resolved_tls_settings(monkeypatch):
    from types import SimpleNamespace
    from tools import mcp_oauth_manager, mcp_tool_transport
    from tools.mcp_tool import MCPServerTask

    provider = SimpleNamespace()
    manager = SimpleNamespace(
        get_or_build_provider=lambda *args: provider,
    )
    cert = ("resolved-cert", "resolved-key")
    monkeypatch.setattr(mcp_oauth_manager, "get_manager", lambda: manager)
    monkeypatch.setattr(
        mcp_tool_transport, "_resolve_client_cert", lambda name, config: cert
    )
    server = MCPServerTask("srv")
    server._auth_type = "oauth"

    assert (
        server._build_oauth_auth(
            "https://mcp.example/mcp",
            {"ssl_verify": False, "client_cert": "configured"},
        )
        is provider
    )
    assert provider._hermes_ssl_verify is False
    assert provider._hermes_client_cert == cert


@pytest.mark.asyncio
async def test_refresh_client_uses_direct_cli_tls_defaults(monkeypatch):
    from urllib import request as urllib_request
    from types import SimpleNamespace
    from tools import mcp_oauth_refresh as refresh

    httpx = sdk_httpx()
    options = []
    monkeypatch.setattr(urllib_request, "getproxies", dict)
    monkeypatch.setattr(urllib_request, "proxy_bypass", lambda host: False)

    def transport(**kwargs):
        options.append(kwargs)
        return httpx.MockTransport(lambda request: httpx.Response(200))

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    provider = SimpleNamespace()
    request = httpx.Request("POST", "https://auth.example/token")
    async with refresh.build_refresh_client(provider, request) as client:
        assert (await client.send(request)).status_code == 200
    assert options == [{"verify": True}]


@pytest.mark.asyncio
async def test_idle_gateway_renews_before_both_tokens_expire_and_reschedules(
    tmp_path, monkeypatch
):
    from tools import mcp_oauth_refresh as refresh

    provider = await _provider(tmp_path, monkeypatch)
    now = [10000.0]
    monkeypatch.setattr(refresh.time, "time", lambda: now[0])
    provider.context.token_expiry_time = now[0] + 3600
    refresh_deadline = [now[0] + 3600]
    delays = []
    calls = []
    httpx = sdk_httpx()
    finished = asyncio.Event()

    async def handle(request):
        assert now[0] < refresh_deadline[0], "rotating refresh token expired"
        calls.append(request)
        refresh_deadline[0] = now[0] + 3600
        return httpx.Response(
            200,
            json={
                "access_token": "new",
                "token_type": "Bearer",
                "refresh_token": f"rotated-{len(calls)}",
                "expires_in": 3600,
            },
        )

    _mock_refresh_client(monkeypatch, handle)

    async def wait(delay):
        delays.append(delay)
        if len(delays) > 2:
            finished.set()
            await asyncio.Event().wait()
        now[0] += delay

    monkeypatch.setattr(refresh, "_wait", wait)
    task = asyncio.create_task(refresh.maintain_tokens(provider))
    try:
        await asyncio.wait_for(finished.wait(), 5)
        assert len(calls) == 2
        assert delays == [3300, 3300, 3300]
        assert provider._hermes_fence is None
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_peer_refresh_uses_fence_and_adopts_rotated_pair(tmp_path, monkeypatch):
    import time
    from tools import mcp_oauth as oauth
    from tools.mcp_oauth_refresh import refresh_if_due

    first = await _provider(tmp_path, monkeypatch)
    second = await _provider(tmp_path, monkeypatch)
    for provider in (first, second):
        provider.context.token_expiry_time = time.time() + 200
    entered, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []
    httpx = sdk_httpx()
    real_acquire = oauth.acquire_refresh_fence

    async def acquire(path):
        if entered.is_set():
            waiting.set()
        return await real_acquire(path)

    async def handle(request):
        calls.append(request)
        entered.set()
        await release.wait()
        return httpx.Response(
            200,
            json={
                "access_token": "peer",
                "token_type": "Bearer",
                "refresh_token": "rotated-peer",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr(oauth, "acquire_refresh_fence", acquire)
    _mock_refresh_client(monkeypatch, handle)
    a = asyncio.create_task(refresh_if_due(first))
    b = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        b = asyncio.create_task(refresh_if_due(second))
        await asyncio.wait_for(waiting.wait(), 5)
        assert len(calls) == 1
        release.set()
        await asyncio.wait_for(asyncio.gather(a, b), 5)
        assert len(calls) == 1
        assert second.context.current_tokens.refresh_token == "rotated-peer"
        assert first._hermes_fence is second._hermes_fence is None
    finally:
        release.set()
        for task in (a, b):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(t for t in (a, b) if t is not None), return_exceptions=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["shutdown", "reconnect"])
async def test_disconnect_cancels_idle_waiter_and_reconnect_has_one_waiter(
    tmp_path, monkeypatch, reason
):
    from tools import mcp_oauth_refresh as refresh
    from tools.mcp_tool import MCPServerTask

    provider = await _provider(tmp_path, monkeypatch)
    server = MCPServerTask("srv")
    server._config = {"url": "https://mcp.example/mcp"}
    server.session = object()
    server._auth_type = "oauth"
    server._oauth_auth = provider
    started, cancelled = asyncio.Event(), asyncio.Event()
    live = []

    async def wait(delay):
        task = asyncio.current_task()
        live.append(task)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            live.remove(task)
            cancelled.set()

    monkeypatch.setattr(refresh, "_wait", wait)
    for _ in range(2):
        started.clear()
        cancelled.clear()
        task = asyncio.create_task(server._wait_for_lifecycle_event())
        await asyncio.wait_for(started.wait(), 5)
        assert len(live) == 1
        event = (
            server._shutdown_event if reason == "shutdown" else server._reconnect_event
        )
        event.set()
        assert await asyncio.wait_for(task, 5) == reason
        assert cancelled.is_set()
        assert live == []
        event.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_home", ["inherit", "owned"])
async def test_maintenance_task_binds_provider_profile_without_changing_environment(
    tmp_path, monkeypatch, provider_home
):
    import os
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from types import SimpleNamespace
    from tools import mcp_oauth_refresh as refresh

    launch_home = tmp_path / "launch"
    owned_home = tmp_path / "owned"
    launch_home.mkdir()
    owned_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    env_home = os.environ.get("HERMES_HOME")
    caller_token = set_hermes_home_override(launch_home)
    provider = SimpleNamespace(
        _hermes_home=str(owned_home) if provider_home == "owned" else ""
    )
    server = SimpleNamespace(_auth_type="oauth", _oauth_auth=provider)
    seen = []
    finished = asyncio.Event()

    async def maintain(current_provider):
        seen.append(get_hermes_home())
        finished.set()

    monkeypatch.setattr(refresh, "maintain_tokens", maintain)
    try:
        task = refresh.start_token_maintenance(server)[0]
        await asyncio.wait_for(finished.wait(), 5)
        await task
        expected = owned_home if provider_home == "owned" else launch_home
        assert seen == [expected]
        assert get_hermes_home() == launch_home
        assert os.environ.get("HERMES_HOME") == env_home
    finally:
        reset_hermes_home_override(caller_token)


@pytest.mark.asyncio
async def test_initialised_not_due_provider_skips_locked_recheck(tmp_path, monkeypatch):
    import time
    from tools import mcp_oauth_refresh as refresh

    provider = await _provider(tmp_path, monkeypatch)
    provider.context.token_expiry_time = time.time() + 1800
    await provider.context.lock.acquire()
    try:
        assert (
            await asyncio.wait_for(refresh.refresh_if_due(provider, initialize=True), 5)
            is True
        )
    finally:
        provider.context.lock.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl,refresh_token", [(None, "refresh"), (3600, None)])
async def test_no_timer_without_expiry_or_refresh_token(
    tmp_path, monkeypatch, ttl, refresh_token
):
    from tools.mcp_oauth_refresh import maintain_tokens, refresh_delay

    provider = await _provider(tmp_path, monkeypatch)
    tokens = OAuthToken(
        access_token="valid",
        token_type="Bearer",
        expires_in=ttl,
        refresh_token=refresh_token,
    )
    provider.context.current_tokens = tokens
    provider.context.update_token_expiry(tokens)
    assert refresh_delay(provider) is None
    await maintain_tokens(provider)


@pytest.mark.asyncio
async def test_cancel_inflight_refresh_releases_fence(tmp_path, monkeypatch):
    import time
    from tools.mcp_oauth_refresh import refresh_if_due

    provider = await _provider(tmp_path, monkeypatch)
    provider.context.token_expiry_time = time.time() + 200
    entered = asyncio.Event()

    async def handle(request):
        entered.set()
        await asyncio.Event().wait()

    _mock_refresh_client(monkeypatch, handle)
    task = asyncio.create_task(refresh_if_due(provider))
    await asyncio.wait_for(entered.wait(), 5)
    assert provider._hermes_fence is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider._hermes_fence is None
    await asyncio.wait_for(provider.context.lock.acquire(), 5)
    provider.context.lock.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["429", "503", "transport"])
async def test_request_transient_refresh_failure_keeps_usable_access(
    tmp_path, monkeypatch, failure
):
    import time

    provider = await _provider(tmp_path, monkeypatch)
    provider.context.token_expiry_time = time.time() + 200
    old_tokens = provider.context.current_tokens
    old_expiry = provider.context.token_expiry_time
    httpx = sdk_httpx()
    calls = []

    async def handle(request):
        calls.append(request)
        if failure == "transport":
            raise httpx.ConnectError("temporary network error", request=request)
        return httpx.Response(int(failure), json={"error": "temporarily_unavailable"})

    _mock_refresh_client(monkeypatch, handle)
    request = httpx.Request("POST", "https://mcp.example/mcp")
    flow = provider.async_auth_flow(request)
    try:
        assert await anext(flow) is request
        assert request.headers["Authorization"] == "Bearer old"
        assert provider.context.current_tokens is old_tokens
        assert provider.context.token_expiry_time == old_expiry
        assert len(calls) == 1
        assert provider._hermes_fence is None
        provider.context.redirect_handler.assert_not_called()
    finally:
        await flow.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["400", "invalid_payload"])
async def test_permanent_early_failure_preserves_access_and_stops_idle_renewal(
    tmp_path, monkeypatch, failure
):
    import time
    from tools import mcp_oauth_refresh as refresh

    provider = await _provider(tmp_path, monkeypatch)
    provider.context.token_expiry_time = time.time() + 200
    old_tokens = provider.context.current_tokens
    old_expiry = provider.context.token_expiry_time
    httpx = sdk_httpx()
    waits, calls = [], []

    async def handle(request):
        calls.append(request)
        if failure == "invalid_payload":
            return httpx.Response(200, content=b"not a token response")
        return httpx.Response(400, json={"error": "invalid_grant"})

    async def wait(delay):
        waits.append(delay)

    _mock_refresh_client(monkeypatch, handle)
    monkeypatch.setattr(refresh, "_wait", wait)
    await refresh.maintain_tokens(provider)

    assert len(calls) == 1
    assert waits == [1.0]
    assert provider.context.current_tokens is old_tokens
    assert provider.context.token_expiry_time == old_expiry
    assert provider._initialized is False
    assert provider._hermes_fence is None
    provider.context.redirect_handler.assert_not_called()

    request = httpx.Request("POST", "https://mcp.example/mcp")
    flow = provider.async_auth_flow(request)
    try:
        assert await anext(flow) is request
        assert request.headers["Authorization"] == "Bearer old"
        assert provider.context.current_tokens.access_token == "old"
        assert len(calls) == 1
    finally:
        await flow.aclose()


@pytest.mark.asyncio
async def test_permanent_failure_does_not_restore_expired_access(tmp_path, monkeypatch):
    import time
    from tools.mcp_oauth_refresh import refresh_if_due

    provider = await _provider(tmp_path, monkeypatch)
    provider.context.token_expiry_time = time.time() - 1
    httpx = sdk_httpx()

    async def handle(request):
        return httpx.Response(400, json={"error": "invalid_grant"})

    _mock_refresh_client(monkeypatch, handle)
    assert await refresh_if_due(provider) is False
    assert provider.context.current_tokens is None
    assert provider.context.token_expiry_time is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport", "fence", "persistence"])
async def test_background_io_failures_schedule_a_bounded_retry(
    tmp_path, monkeypatch, failure
):
    from tools import mcp_oauth_refresh as refresh
    from tools import mcp_oauth as oauth

    provider = await _provider(tmp_path, monkeypatch)
    now = [10000.0]
    monkeypatch.setattr(refresh.time, "time", lambda: now[0])
    provider.context.token_expiry_time = now[0] + 200
    httpx = sdk_httpx()
    calls, waits = [], []
    retry_waiting = asyncio.Event()

    async def handle(request):
        calls.append(request)
        if failure == "transport" and len(calls) == 1:
            raise httpx.ConnectError("temporary network error", request=request)
        return httpx.Response(
            200,
            json={
                "access_token": f"new-{len(calls)}",
                "token_type": "Bearer",
                "refresh_token": f"refresh-{len(calls)}",
                "expires_in": 3600,
            },
        )

    _mock_refresh_client(monkeypatch, handle)
    if failure == "fence":
        real_acquire = oauth.acquire_refresh_fence
        attempts = [0]

        async def acquire(path):
            attempts[0] += 1
            if attempts[0] == 1:
                raise oauth.RefreshFenceTimeout("temporary fence contention")
            return await real_acquire(path)

        monkeypatch.setattr(oauth, "acquire_refresh_fence", acquire)
    elif failure == "persistence":
        storage = provider.context.storage
        real_set_tokens = storage.set_tokens
        writes = [0]

        async def set_tokens(tokens):
            writes[0] += 1
            if writes[0] == 1:
                raise OSError("temporary token-store write failure")
            await real_set_tokens(tokens)

        monkeypatch.setattr(storage, "set_tokens", set_tokens)

    async def wait(delay):
        waits.append(delay)
        if len(waits) >= 3:
            retry_waiting.set()
            await asyncio.Event().wait()
        now[0] += delay

    monkeypatch.setattr(refresh, "_wait", wait)
    task = asyncio.create_task(refresh.maintain_tokens(provider))
    try:
        await asyncio.wait_for(retry_waiting.wait(), 5)
        assert waits[:2] == [1.0, 60.0]
        assert provider._hermes_fence is None
        expected_calls = {"transport": 2, "fence": 1, "persistence": 1}[failure]
        assert len(calls) == expected_calls
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_background_stops_after_five_consecutive_failures(
    tmp_path, monkeypatch, caplog
):
    import time
    from tools import mcp_oauth_refresh as refresh

    provider = await _provider(tmp_path, monkeypatch)
    provider.context.token_expiry_time = time.time() + 200
    calls, waits = [], []

    async def fail(current_provider):
        calls.append(current_provider)
        raise ConnectionError("simulated-error-body")

    async def wait(delay):
        waits.append(delay)

    monkeypatch.setattr(refresh, "refresh_if_due", fail)
    monkeypatch.setattr(refresh, "_wait", wait)
    await refresh.maintain_tokens(provider)

    assert len(calls) == 5
    assert waits == [1.0, 60.0, 60.0, 60.0, 60.0]
    assert "srv" in caplog.text
    assert "simulated-error-body" not in caplog.text


@pytest.mark.asyncio
async def test_short_grant_retries_before_its_refresh_grant_expires(
    tmp_path, monkeypatch
):
    from tools import mcp_oauth_refresh as refresh

    provider = await _provider(tmp_path, monkeypatch, ttl=90)
    now = [10000.0]
    monkeypatch.setattr(refresh.time, "time", lambda: now[0])
    provider.context.token_expiry_time = now[0] + 90
    httpx = sdk_httpx()
    calls, waits = [], []
    retry_waiting = asyncio.Event()

    async def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": "temporarily_unavailable"})
        assert now[0] < 10090.0, "refresh token and access token co-expired"
        return httpx.Response(
            200,
            json={
                "access_token": "renewed",
                "token_type": "Bearer",
                "refresh_token": "renewed-refresh",
                "expires_in": 3600,
            },
        )

    async def wait(delay):
        waits.append(delay)
        if len(waits) >= 3:
            retry_waiting.set()
            await asyncio.Event().wait()
        now[0] += delay

    _mock_refresh_client(monkeypatch, handle)
    monkeypatch.setattr(refresh, "_wait", wait)
    task = asyncio.create_task(refresh.maintain_tokens(provider))
    try:
        await asyncio.wait_for(retry_waiting.wait(), 5)
        assert len(calls) == 2
        assert waits == [45.0, 15.0, 3300.0]
        assert provider.context.current_tokens.access_token == "renewed"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_zero_ttl_response_stops_idle_renewal(tmp_path, monkeypatch):
    from tools import mcp_oauth_refresh as refresh

    provider = await _provider(tmp_path, monkeypatch)
    now = [10000.0]
    monkeypatch.setattr(refresh.time, "time", lambda: now[0])
    provider.context.token_expiry_time = now[0] + 200
    httpx = sdk_httpx()
    calls, waits = [], []

    async def handle(request):
        calls.append(request)
        return _success_response(httpx, [], expires_in=0)

    async def wait(delay):
        waits.append(delay)
        now[0] += delay

    _mock_refresh_client(monkeypatch, handle)
    monkeypatch.setattr(refresh, "_wait", wait)
    await refresh.maintain_tokens(provider)

    assert len(calls) == 1
    assert waits == [1.0]
    assert refresh.refresh_delay(provider) is None


@pytest.mark.asyncio
async def test_short_grant_has_positive_refresh_interval(tmp_path, monkeypatch):
    import time
    from tools.mcp_oauth_refresh import refresh_delay

    provider = await _provider(tmp_path, monkeypatch, ttl=10)
    provider.context.token_expiry_time = time.time() + 10
    assert 4 < refresh_delay(provider) <= 6
