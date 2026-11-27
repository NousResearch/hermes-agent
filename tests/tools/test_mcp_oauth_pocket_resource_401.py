"""Pocket MCP resource-401 recovery contract.

Pocket intermittently rejects a still-valid Bearer token with HTTP 401. The SDK's
``async_auth_flow`` only refreshes when ``is_token_valid()`` is false, so such a 401 jumps
straight into full browser authorization, which a background cron job cannot complete.

For the exact ``pocket`` server at ``https://public.heypocketai.com/mcp`` and read-only
JSON-RPC calls only, the bridge retries once with the current token, then performs at most one
fenced refresh and one final attempt (<= 3 resource requests, <= 1 refresh). Everything else is
forwarded to the SDK unchanged.
"""
from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK required")

POCKET_URL = "https://public.heypocketai.com/mcp"
TOKEN_URL = "https://public.heypocketai.com/oauth/token"
HEADERS = {"mcp-protocol-version": "2025-06-18", "mcp-session-id": "sess-123"}
WWW_AUTH = {"www-authenticate": 'Bearer error="invalid_token"'}


def _httpx():
    from tools.mcp_tool import sdk_httpx
    return sdk_httpx()


def _tracked_stream():
    """Response body that records whether the bridge drained and closed it."""

    class _TrackedStream(_httpx().AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'{"error":"missing or invalid Authorization header"}'

        async def aclose(self):
            self.closed = True

    return _TrackedStream()


def _resp(status, request, *, headers=None, body=None):
    httpx = _httpx()
    if body is not None:
        return httpx.Response(status, request=request, json=body)
    stream = _tracked_stream()
    resp = httpx.Response(status, request=request, headers=headers or {}, stream=stream)
    resp.tracked = stream
    return resp


def _rpc(method, params=None, *, msg_id=1, url=POCKET_URL, raw=None):
    payload = {"jsonrpc": "2.0", "method": method}
    if msg_id is not None:
        payload["id"] = msg_id
    if params is not None:
        payload["params"] = params
    content = raw if raw is not None else json.dumps(payload).encode()
    return _httpx().Request("POST", url, content=content,
                            headers={**HEADERS, "content-type": "application/json"})


def _call(name):
    return _rpc("tools/call", {"name": name, "arguments": {}})


async def _no_redirect(_url):
    return None


async def _no_callback():
    raise RuntimeError("interactive authorization reached")


async def _provider(tmp_path, monkeypatch, *, server_name="pocket", server_url=POCKET_URL,
                    refresh_token="rt-1", tokens=True):
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    storage = HermesTokenStorage(server_name)
    if tokens:
        await storage.set_tokens(OAuthToken(access_token="at-1", token_type="Bearer", expires_in=3600,
                                            refresh_token=refresh_token))
    await storage.set_client_info(OAuthClientInformationFull(
        client_id="cid", redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        grant_types=["authorization_code", "refresh_token"], response_types=["code"],
        token_endpoint_auth_method="none"))
    storage.save_oauth_metadata(OAuthMetadata(
        issuer=AnyUrl("https://public.heypocketai.com"),
        authorization_endpoint=AnyUrl("https://public.heypocketai.com/oauth/authorize"),
        token_endpoint=AnyUrl(TOKEN_URL)))
    provider = _HERMES_PROVIDER_CLS(
        server_name=server_name, server_url=server_url,
        client_metadata=OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
                                            client_name="Hermes Agent"),
        storage=storage, redirect_handler=_no_redirect, callback_handler=_no_callback)
    return provider, storage


def _is_token_post(req):
    return req.method == "POST" and str(req.url) == TOKEN_URL


def _assert_idle(provider):
    assert provider.context.lock.value == 1
    assert provider._hermes_fence is None


def _assert_no_fence(provider):
    # Forwarded (unchanged) SDK paths keep the baseline cleanup contract: the abandoned inner
    # generator is finalized by the event loop, so only the fence is checked synchronously.
    assert provider._hermes_fence is None


# ── qualifying read requests ──────────────────────────────────────────────────

READ_REQUESTS = {
    "get": lambda: _httpx().Request("GET", POCKET_URL, headers=HEADERS),
    "initialize": lambda: _rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                              "clientInfo": {"name": "hermes", "version": "1"}}),
    "initialized": lambda: _rpc("notifications/initialized", msg_id=None),
    "tools_list": lambda: _rpc("tools/list", {}),
    "get_account_info": lambda: _call("get_account_info"),
    "list_pocket_folders": lambda: _call("list_pocket_folders"),
    "search_pocket_actionitems": lambda: _call("search_pocket_actionitems"),
    "search_pocket_conversations": lambda: _call("search_pocket_conversations"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(READ_REQUESTS))
async def test_read_401_retries_once_with_current_token(tmp_path, monkeypatch, kind):
    provider, _ = await _provider(tmp_path, monkeypatch)
    req = READ_REQUESTS[kind]()
    flow = provider.async_auth_flow(req)
    out = await flow.__anext__()
    assert out is req and out.headers["authorization"] == "Bearer at-1"

    first_401 = _resp(401, out, headers=WWW_AUTH)
    retry = await flow.asend(first_401)
    assert retry is req
    assert retry.headers["authorization"] == "Bearer at-1"
    assert retry.headers["mcp-session-id"] == "sess-123"
    assert retry.headers["mcp-protocol-version"] == "2025-06-18"
    assert first_401.tracked.closed

    with pytest.raises(StopAsyncIteration):
        await flow.asend(_resp(200, retry, body={"jsonrpc": "2.0", "id": 1, "result": {}}))
    _assert_idle(provider)


@pytest.mark.asyncio
async def test_401_401_refreshes_once_then_succeeds(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    provider, storage = await _provider(tmp_path, monkeypatch)
    req = _call("search_pocket_conversations")
    flow = provider.async_auth_flow(req)
    out = await flow.__anext__()
    retry = await flow.asend(_resp(401, out, headers=WWW_AUTH))
    assert retry is req
    second_401 = _resp(401, retry, headers=WWW_AUTH)
    refresh = await flow.asend(second_401)
    assert _is_token_post(refresh)
    assert second_401.tracked.closed
    assert provider._hermes_fence is not None  # refresh holds the cross-process fence

    final = await flow.asend(_resp(200, refresh, body={
        "access_token": "at-2", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "rt-2"}))
    assert final is req
    assert final.headers["authorization"] == "Bearer at-2"
    assert final.headers["mcp-session-id"] == "sess-123"
    with pytest.raises(StopAsyncIteration):
        await flow.asend(_resp(200, final, body={"jsonrpc": "2.0", "id": 1, "result": {}}))

    _assert_idle(provider)
    persisted = await storage.get_tokens()
    assert (persisted.access_token, persisted.refresh_token) == ("at-2", "rt-2")
    for secret in ("at-1", "at-2", "rt-1", "rt-2", "sess-123", "Bearer", "missing or invalid"):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_persistent_401_is_capped_at_three_requests_and_one_refresh(tmp_path, monkeypatch):
    from tools.mcp_oauth_pocket import PocketReadAuthExhausted

    provider, _ = await _provider(tmp_path, monkeypatch)
    req = _rpc("tools/list", {})
    flow = provider.async_auth_flow(req)
    sent = [await flow.__anext__()]
    sent.append(await flow.asend(_resp(401, sent[-1], headers=WWW_AUTH)))
    sent.append(await flow.asend(_resp(401, sent[-1], headers=WWW_AUTH)))
    sent.append(await flow.asend(_resp(200, sent[-1], body={
        "access_token": "at-2", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "rt-2"})))
    last_401 = _resp(401, sent[-1], headers=WWW_AUTH)
    with pytest.raises(PocketReadAuthExhausted):  # fails closed: never the SDK's interactive 401 branch
        await flow.asend(last_401)
    assert [r is req for r in sent] == [True, True, False, True]
    assert sum(1 for r in sent if _is_token_post(r)) == 1
    assert last_401.tracked.closed
    _assert_idle(provider)


async def _drive_to_second_401(provider, req):
    flow = provider.async_auth_flow(req)
    out = await flow.__anext__()
    retry = await flow.asend(_resp(401, out, headers=WWW_AUTH))
    assert retry is req
    return flow, retry


@pytest.mark.asyncio
async def test_rejected_refresh_fails_closed(tmp_path, monkeypatch):
    from tools.mcp_oauth_pocket import PocketReadAuthExhausted

    provider, _ = await _provider(tmp_path, monkeypatch)
    req = _call("get_account_info")
    flow, retry = await _drive_to_second_401(provider, req)
    refresh = await flow.asend(_resp(401, retry, headers=WWW_AUTH))
    assert _is_token_post(refresh)
    with pytest.raises(PocketReadAuthExhausted):  # no retry loop, no interactive authorization
        await flow.asend(_resp(400, refresh, body={"error": "invalid_grant"}))
    assert provider.context.current_tokens is None
    _assert_idle(provider)


@pytest.mark.asyncio
async def test_no_refresh_token_fails_closed(tmp_path, monkeypatch):
    from tools.mcp_oauth_pocket import PocketReadAuthExhausted

    provider, _ = await _provider(tmp_path, monkeypatch, refresh_token=None)
    req = _call("list_pocket_folders")
    flow, retry = await _drive_to_second_401(provider, req)
    with pytest.raises(PocketReadAuthExhausted):
        await flow.asend(_resp(401, retry, headers=WWW_AUTH))
    _assert_idle(provider)


@pytest.mark.asyncio
async def test_no_tokens_is_not_retried(tmp_path, monkeypatch):
    provider, _ = await _provider(tmp_path, monkeypatch, tokens=False)
    req = _rpc("tools/list", {})
    flow = provider.async_auth_flow(req)
    out = await flow.__anext__()
    assert "authorization" not in out.headers
    nxt = await flow.asend(_resp(401, out, headers=WWW_AUTH))
    assert nxt is not req and not _is_token_post(nxt)
    await flow.aclose()
    _assert_no_fence(provider)


# ── non-qualifying requests are forwarded to the SDK unchanged ────────────────

UNSAFE = {
    "unrelated_server": ("notion", POCKET_URL, lambda: _rpc("tools/list", {})),
    "wrong_host": ("pocket", "https://evil.example.com/mcp",
                   lambda: _rpc("tools/list", {}, url="https://evil.example.com/mcp")),
    "lookalike_host": ("pocket", POCKET_URL,
                       lambda: _rpc("tools/list", {}, url="https://public.heypocketai.com.evil.io/mcp")),
    "wrong_path": ("pocket", POCKET_URL, lambda: _rpc("tools/list", {}, url=POCKET_URL + "/x")),
    "query": ("pocket", POCKET_URL, lambda: _rpc("tools/list", {}, url=POCKET_URL + "?a=1")),
    "http_scheme": ("pocket", POCKET_URL,
                    lambda: _rpc("tools/list", {}, url="http://public.heypocketai.com/mcp")),
    "other_port": ("pocket", POCKET_URL,
                   lambda: _rpc("tools/list", {}, url="https://public.heypocketai.com:8443/mcp")),
    "delete": ("pocket", POCKET_URL, lambda: _httpx().Request("DELETE", POCKET_URL, headers=HEADERS)),
    "mutation": ("pocket", POCKET_URL, lambda: _call("update_pocket_actionitem")),
    "unknown_tool": ("pocket", POCKET_URL, lambda: _call("delete_everything")),
    "substring_tool": ("pocket", POCKET_URL, lambda: _call("get_account_info_and_update")),
    "unknown_rpc": ("pocket", POCKET_URL, lambda: _rpc("resources/read", {"uri": "x"})),
    "batch": ("pocket", POCKET_URL, lambda: _rpc("x", raw=json.dumps(
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}]).encode())),
    "malformed_json": ("pocket", POCKET_URL, lambda: _rpc("x", raw=b'{"jsonrpc":"2.0","method":"tools/list"')),
    "wrong_jsonrpc": ("pocket", POCKET_URL, lambda: _rpc("x", raw=json.dumps(
        {"jsonrpc": "1.0", "id": 1, "method": "tools/list"}).encode())),
    "extra_member": ("pocket", POCKET_URL, lambda: _rpc("x", raw=json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "result": {}}).encode())),
    "call_without_name": ("pocket", POCKET_URL, lambda: _rpc("tools/call", {"arguments": {}})),
    "call_bad_args": ("pocket", POCKET_URL,
                      lambda: _rpc("tools/call", {"name": "get_account_info", "arguments": []})),
    "get_with_body": ("pocket", POCKET_URL,
                      lambda: _httpx().Request("GET", POCKET_URL, headers=HEADERS, content=b"x")),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(UNSAFE))
async def test_unsafe_or_unrelated_401_is_not_retried(tmp_path, monkeypatch, case):
    server_name, server_url, make = UNSAFE[case]
    provider, _ = await _provider(tmp_path, monkeypatch, server_name=server_name, server_url=server_url)
    req = make()
    flow = provider.async_auth_flow(req)
    out = await flow.__anext__()
    assert out is req
    try:
        nxt = await flow.asend(_resp(401, out, headers=WWW_AUTH))
    except RuntimeError:
        nxt = None  # SDK's own validation/authorization path; still not a retry
    assert nxt is not req and (nxt is None or not _is_token_post(nxt))
    await flow.aclose()
    _assert_no_fence(provider)


@pytest.mark.asyncio
async def test_insufficient_scope_403_never_refreshes(tmp_path, monkeypatch):
    provider, _ = await _provider(tmp_path, monkeypatch)
    req = _call("get_account_info")
    flow = provider.async_auth_flow(req)
    out = await flow.__anext__()
    try:
        nxt = await flow.asend(_resp(403, out, headers={
            "www-authenticate": 'Bearer error="insufficient_scope", scope="read"'}))
    except (StopAsyncIteration, RuntimeError):
        nxt = None
    assert nxt is not req and (nxt is None or not _is_token_post(nxt))
    await flow.aclose()
    _assert_no_fence(provider)


# ── cancellation, peer rotation, zero overhead ────────────────────────────────

@pytest.mark.asyncio
async def test_cancel_during_resource_retry_releases_state(tmp_path, monkeypatch):
    provider, _ = await _provider(tmp_path, monkeypatch)
    req = _rpc("tools/list", {})
    flow, _retry = await _drive_to_second_401(provider, req)
    await flow.aclose()
    _assert_idle(provider)
    # A fresh flow acquires the semaphore immediately.
    nxt = provider.async_auth_flow(_rpc("tools/list", {}))
    assert (await nxt.__anext__()).headers["authorization"] == "Bearer at-1"
    await nxt.aclose()
    _assert_no_fence(provider)


@pytest.mark.asyncio
async def test_cancel_during_refresh_releases_fence_and_semaphore(tmp_path, monkeypatch):
    from tools.mcp_oauth import acquire_refresh_fence, release_refresh_fence

    provider, storage = await _provider(tmp_path, monkeypatch)
    req = _rpc("tools/list", {})
    flow, retry = await _drive_to_second_401(provider, req)
    refresh = await flow.asend(_resp(401, retry, headers=WWW_AUTH))
    assert _is_token_post(refresh) and provider._hermes_fence is not None
    await flow.aclose()
    _assert_idle(provider)
    fd = await acquire_refresh_fence(storage._tokens_path(), timeout=1.0)
    release_refresh_fence(fd)


@pytest.mark.asyncio
async def test_peer_rotation_is_adopted_without_refresh_post(tmp_path, monkeypatch):
    from mcp.shared.auth import OAuthToken

    provider, storage = await _provider(tmp_path, monkeypatch)
    req = _call("search_pocket_actionitems")
    flow, retry = await _drive_to_second_401(provider, req)
    # Another process refreshed and persisted a new live pair meanwhile.
    await storage.set_tokens(OAuthToken(access_token="at-peer", token_type="Bearer", expires_in=3600,
                                        refresh_token="rt-peer"))
    final = await flow.asend(_resp(401, retry, headers=WWW_AUTH))
    assert final is req and final.headers["authorization"] == "Bearer at-peer"
    with pytest.raises(StopAsyncIteration):
        await flow.asend(_resp(200, final, body={"jsonrpc": "2.0", "id": 1, "result": {}}))
    _assert_idle(provider)


@pytest.mark.asyncio
async def test_normal_success_has_zero_overhead(tmp_path, monkeypatch):
    provider, _ = await _provider(tmp_path, monkeypatch)
    req = _call("get_account_info")
    flow = provider.async_auth_flow(req)
    out = await flow.__anext__()
    assert out is req
    with pytest.raises(StopAsyncIteration):
        await flow.asend(_resp(200, out, body={"jsonrpc": "2.0", "id": 1, "result": {}}))
    _assert_idle(provider)
