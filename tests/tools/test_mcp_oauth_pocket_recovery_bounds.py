"""Pocket resource-401 recovery: whole-flow bounds, fail-closed exhaustion, real cancellation.

Driven through a real SDK ``AsyncClient`` over ``MockTransport`` (no network), so HTTPX's own
auth-flow driver, response draining and ``aclose`` on cancellation are what the provider sees.

- At most ONE refresh POST per auth flow, counting the SDK's initial expired-token refresh.
- Exhausted recovery raises ``PocketReadAuthExhausted`` at once: no browser redirect, no
  authorization-code exchange, no fourth resource request, even when the callback would succeed.
- ``task.cancel()`` / ``asyncio.wait_for`` timeouts while a peer holds ``context.lock`` never
  release the peer's lock, never swallow the cancellation, and leave the provider idle (lock
  free, no waiters, no refresh fence, no stray tasks or loop errors).
- Ambiguous payloads (non-string method, NaN/Infinity, duplicate members) are not recoverable.
"""
from __future__ import annotations

import asyncio
import gc
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from tests.waiting import async_wait_until

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK required")

POCKET_URL = "https://public.heypocketai.com/mcp"
ISSUER = "https://public.heypocketai.com"
TOKEN_URL = ISSUER + "/oauth/token"
HEADERS = {"mcp-protocol-version": "2025-06-18", "mcp-session-id": "sess-123",
           "content-type": "application/json"}
WWW_AUTH = {"www-authenticate": 'Bearer error="invalid_token"'}
OK_RPC = {"jsonrpc": "2.0", "id": 1, "result": {}}


def _httpx():
    from tools.mcp_tool import sdk_httpx
    return sdk_httpx()


def _rpc_request(method="tools/list", params=None, *, raw=None):
    body = raw if raw is not None else json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).encode()
    return _httpx().Request("POST", POCKET_URL, content=body, headers=HEADERS)


class _Server:
    """Fake Pocket resource + authorization server that records every request it receives."""

    def __init__(self):
        self.log: list[tuple[str, str]] = []
        self.resource_status: list[int] = []  # popped per resource request; 401 once exhausted
        self.gates: dict[tuple[str, int], asyncio.Event] = {}  # (kind, nth) -> reached signal
        self.block: set[tuple[str, int]] = set()  # (kind, nth) that hang until cancelled
        self.hold_until: dict[tuple[str, int], asyncio.Event] = {}  # respond only once set
        self.blocking_body: set[tuple[str, int]] = set()  # 401 whose body never finishes
        self.status_headers: dict[int, dict] = {}  # per-status challenge override (default WWW_AUTH)

    def count(self, kind):
        return sum(1 for k, _ in self.log if k == kind)

    def _kind(self, request):
        url = str(request.url)
        if url == POCKET_URL:
            return "resource"
        if url == TOKEN_URL:
            grant = parse_qs(request.content.decode()).get("grant_type", [""])[0]
            return "refresh" if grant == "refresh_token" else "code_exchange"
        if "/.well-known/" in url:
            return "discovery"
        if url.endswith("/register"):
            return "register"
        return "other"

    async def handler(self, request):
        httpx = _httpx()
        kind = self._kind(request)
        self.log.append((kind, request.headers.get("authorization", "")))
        key = (kind, self.count(kind))
        if key in self.gates:
            self.gates[key].set()
        if key in self.block:
            await asyncio.Event().wait()  # in flight until cancelled
        if key in self.hold_until:
            await self.hold_until[key].wait()
        if kind == "resource":
            status = self.resource_status.pop(0) if self.resource_status else 401
            if status == 200:
                return httpx.Response(200, json=OK_RPC)
            if key in self.blocking_body:
                return httpx.Response(401, headers=WWW_AUTH, stream=_blocking_stream_cls()())
            return httpx.Response(status, headers=self.status_headers.get(status, WWW_AUTH), json={"error": "missing or invalid Authorization header"})
        if kind in ("refresh", "code_exchange"):
            n = self.count(kind)
            return httpx.Response(200, json={"access_token": f"at-{kind}-{n}", "token_type": "Bearer",
                                             "expires_in": 3600, "refresh_token": f"rt-{kind}-{n}"})
        if kind == "discovery" and "oauth-protected-resource" in str(request.url):
            return httpx.Response(200, json={"resource": POCKET_URL, "authorization_servers": [ISSUER]})
        if kind == "discovery":
            return httpx.Response(200, json={
                "issuer": ISSUER, "authorization_endpoint": ISSUER + "/oauth/authorize",
                "token_endpoint": TOKEN_URL, "response_types_supported": ["code"],
                "code_challenge_methods_supported": ["S256"]})
        return httpx.Response(404)


class _BlockingStream:
    """A 401 body that never finishes arriving (cancellation while draining it)."""

    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield b'{"error":'
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def _blocking_stream_cls():
    base = _httpx().AsyncByteStream

    class _S(_BlockingStream, base):
        pass
    return _S


class _Interactive:
    """Redirect + callback handlers that WOULD succeed, recording whether they were reached."""

    def __init__(self):
        self.redirects: list[str] = []
        self.callbacks = 0

    async def redirect(self, url):
        self.redirects.append(url)

    async def callback(self):
        self.callbacks += 1
        from mcp.shared.auth import AuthorizationCodeResult
        query = parse_qs(urlsplit(self.redirects[-1]).query)
        return AuthorizationCodeResult(code="auth-code", state=query["state"][0], iss=ISSUER)


async def _provider(tmp_path, monkeypatch, *, expired=False, refresh_token="rt-1"):
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    storage = HermesTokenStorage("pocket")
    await storage.set_tokens(OAuthToken(access_token="at-1", token_type="Bearer",
                                        expires_in=-120 if expired else 3600, refresh_token=refresh_token))
    await storage.set_client_info(OAuthClientInformationFull(
        client_id="cid", redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        grant_types=["authorization_code", "refresh_token"], response_types=["code"],
        token_endpoint_auth_method="none"))
    storage.save_oauth_metadata(OAuthMetadata(
        issuer=AnyUrl(ISSUER), authorization_endpoint=AnyUrl(ISSUER + "/oauth/authorize"),
        token_endpoint=AnyUrl(TOKEN_URL)))
    interactive = _Interactive()
    provider = _HERMES_PROVIDER_CLS(
        server_name="pocket", server_url=POCKET_URL,
        client_metadata=OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
                                            client_name="Hermes Agent"),
        storage=storage, redirect_handler=interactive.redirect, callback_handler=interactive.callback)
    return provider, storage, interactive


def _client(provider, server):
    httpx = _httpx()
    return httpx.AsyncClient(transport=httpx.MockTransport(server.handler), auth=provider)


def _exhausted_cls():
    from tools.mcp_oauth_pocket import PocketReadAuthExhausted
    return PocketReadAuthExhausted


def _assert_no_interactive(server, interactive):
    assert interactive.redirects == [] and interactive.callbacks == 0
    assert server.count("code_exchange") == 0 and server.count("register") == 0


def _assert_idle(provider):
    lock = provider.context.lock  # binary semaphore: 1 == free
    assert lock.value == 1
    assert lock.statistics().tasks_waiting == 0
    assert provider._hermes_fence is None


# ── 1. one refresh per flow, including the SDK's initial expired-token refresh ──

@pytest.mark.asyncio
async def test_initial_refresh_counts_toward_the_single_refresh(tmp_path, monkeypatch):
    provider, _, interactive = await _provider(tmp_path, monkeypatch, expired=True)
    server = _Server()
    server.resource_status = [401, 401, 401, 401]
    async with _client(provider, server) as client:
        with pytest.raises(_exhausted_cls()):
            await client.send(_rpc_request())
    kinds = [k for k, _ in server.log]
    assert kinds == ["refresh", "resource", "resource"]  # initial refresh, 401, same-token retry, 401
    assert server.log[1][1] == server.log[2][1] == "Bearer at-refresh-1"
    _assert_no_interactive(server, interactive)
    _assert_idle(provider)


# ── 2. exhaustion fails closed, even when interactive auth would succeed ──────

@pytest.mark.asyncio
async def test_persistent_401_fails_closed_before_interactive_auth(tmp_path, monkeypatch, caplog):
    provider, _, interactive = await _provider(tmp_path, monkeypatch)
    server = _Server()
    server.resource_status = [401, 401, 401, 200]
    async with _client(provider, server) as client:
        with pytest.raises(_exhausted_cls()) as err:
            await client.send(_rpc_request())
    assert [k for k, _ in server.log] == ["resource", "resource", "refresh", "resource"]
    _assert_no_interactive(server, interactive)
    _assert_idle(provider)
    for secret in ("at-1", "rt-1", "at-refresh", "sess-123", "Bearer", "missing or invalid"):
        assert secret not in str(err.value) and secret not in caplog.text


@pytest.mark.asyncio
async def test_rejected_refresh_fails_closed(tmp_path, monkeypatch):
    provider, _, interactive = await _provider(tmp_path, monkeypatch)
    server = _Server()

    async def reject_refresh(request, _orig=server.handler):
        if str(request.url) == TOKEN_URL:
            server.log.append(("refresh", ""))
            return _httpx().Response(400, json={"error": "invalid_grant"})
        return await _orig(request)

    httpx = _httpx()
    async with httpx.AsyncClient(transport=httpx.MockTransport(reject_refresh), auth=provider) as client:
        with pytest.raises(_exhausted_cls()):
            await client.send(_rpc_request())
    assert [k for k, _ in server.log] == ["resource", "resource", "refresh"]
    _assert_no_interactive(server, interactive)
    _assert_idle(provider)


@pytest.mark.asyncio
async def test_no_refresh_token_fails_closed(tmp_path, monkeypatch):
    provider, _, interactive = await _provider(tmp_path, monkeypatch, refresh_token=None)
    server = _Server()
    server.resource_status = [401, 401, 200]
    async with _client(provider, server) as client:
        with pytest.raises(_exhausted_cls()):
            await client.send(_rpc_request())
    assert [k for k, _ in server.log] == ["resource", "resource"]
    _assert_no_interactive(server, interactive)
    _assert_idle(provider)


@pytest.mark.asyncio
async def test_recovered_read_still_succeeds_through_real_client(tmp_path, monkeypatch):
    provider, _, interactive = await _provider(tmp_path, monkeypatch)
    server = _Server()
    server.resource_status = [401, 401, 200]
    async with _client(provider, server) as client:
        resp = await client.send(_rpc_request())
    assert resp.status_code == 200 and resp.json() == OK_RPC
    assert [k for k, _ in server.log] == ["resource", "resource", "refresh", "resource"]
    assert server.log[-1][1] == "Bearer at-refresh-1"
    _assert_no_interactive(server, interactive)
    _assert_idle(provider)


INSUFFICIENT_SCOPE = {"www-authenticate": 'Bearer error="insufficient_scope", scope="read write"'}


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses,expected", [
    ([401, 403, 200], ["resource", "resource"]),
    ([401, 401, 403, 200], ["resource", "resource", "refresh", "resource"]),
], ids=["401-403", "401-401-refresh-403"])
async def test_403_during_recovery_fails_closed_before_scope_step_up(tmp_path, monkeypatch, statuses, expected):
    """A 403 insufficient_scope mid-recovery must not reach the SDK's interactive step-up, even
    when redirect + callback + code exchange would all succeed."""
    provider, storage, interactive = await _provider(tmp_path, monkeypatch)
    scope_before = provider.context.client_metadata.scope
    server = _Server()
    server.resource_status = statuses
    server.status_headers[403] = INSUFFICIENT_SCOPE
    async with _client(provider, server) as client:
        with pytest.raises(_exhausted_cls()):
            await client.send(_rpc_request())
    assert [k for k, _ in server.log] == expected
    assert server.count("resource") <= 3 and server.count("refresh") <= 1
    _assert_no_interactive(server, interactive)
    assert provider.context.client_metadata.scope == scope_before
    tokens = await storage.get_tokens()
    assert tokens.access_token == ("at-refresh-1" if "refresh" in expected else "at-1")
    _assert_idle(provider)


# ── 3. real cancellation / timeout under contention ───────────────────────────

class _Peer:
    """A concurrent flow that takes ``context.lock`` in its own task and releases it exactly once."""

    def __init__(self, lock):
        self.lock = lock
        self.holding = asyncio.Event()
        self.release = asyncio.Event()
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        await self.lock.acquire()
        self.holding.set()
        await self.release.wait()
        self.lock.release()


def _peer_still_holds(lock):
    # A binary semaphore has no owner task: a wrongful release by the interrupted flow would
    # raise its value (or hand it to a waiter) and make the peer's own release fail.
    return lock.value == 0


async def _settle(n=200):
    """Yield to the loop n times: drains in-loop callbacks only. Not equivalent to a wall-clock wait; a timer,
    worker thread or delayed finalizer can land after it (no such path in the cleanup observed here)."""
    for _ in range(n):
        await asyncio.sleep(0)


def _start(client, mode, request):
    if mode == "cancel":
        return asyncio.create_task(client.send(request))
    return asyncio.create_task(asyncio.wait_for(client.send(request), timeout=0.3))


async def _interrupt(task, mode):
    if mode == "cancel":
        task.cancel()
    else:  # wait_for's deadline cancels the outer task; it stays pending while cleanup defers it
        await async_wait_until(lambda: task.cancelling() or task.done(), timeout=5, what="wait_for deadline")


async def _finish(task, mode):
    expected = asyncio.CancelledError if mode == "cancel" else asyncio.TimeoutError
    with pytest.raises(expected):
        await task


def _capture_loop_errors():
    errors = []
    asyncio.get_running_loop().set_exception_handler(lambda _loop, ctx: errors.append(ctx))
    return errors


async def _assert_clean_after(provider, errors, baseline_tasks):
    gc.collect()
    await _settle()
    _assert_idle(provider)
    assert errors == []
    assert asyncio.all_tasks() - baseline_tasks == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cancel", "timeout"])
@pytest.mark.parametrize("nth", [1, 2, 3])
async def test_interrupt_while_reacquiring_keeps_peer_ownership(tmp_path, monkeypatch, mode, nth):
    """The original (1), retried (2) or final (3) request's response arrives while a peer holds
    ``context.lock``; the flow is interrupted while waiting to re-own it. It must keep waiting
    (cancellation deferred, not swallowed), never release the peer's lock, then close the SDK
    generator and re-raise."""
    errors = _capture_loop_errors()
    baseline = set(asyncio.all_tasks())
    provider, _, interactive = await _provider(tmp_path, monkeypatch)
    lock = provider.context.lock
    server = _Server()
    server.resource_status = [401] * (nth - 1) + [200]
    retry_reached, respond = asyncio.Event(), asyncio.Event()
    server.gates[("resource", nth)] = retry_reached
    server.hold_until[("resource", nth)] = respond
    async with _client(provider, server) as client:
        task = _start(client, mode, _rpc_request())
        await asyncio.wait_for(retry_reached.wait(), 5)
        peer = _Peer(lock)
        await asyncio.wait_for(peer.holding.wait(), 5)
        respond.set()
        await async_wait_until(lambda: lock.statistics().tasks_waiting == 1, timeout=5, what="flow waits on lock")
        assert lock.statistics().tasks_waiting == 1  # the flow is now waiting to re-own the lock
        await _interrupt(task, mode)
        await _settle()
        assert _peer_still_holds(lock)  # the peer never lost ownership
        assert not task.done()  # cancellation is deferred until the flow owns the lock again
        peer.release.set()
        await peer.task  # the peer's single release succeeds
        await _finish(task, mode)
    assert server.count("refresh") == (1 if nth == 3 else 0)
    _assert_no_interactive(server, interactive)
    await _assert_clean_after(provider, errors, baseline)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cancel", "timeout"])
@pytest.mark.parametrize("nth", [1, 2, 3])
async def test_interrupt_while_resource_request_in_flight_with_peer(tmp_path, monkeypatch, mode, nth):
    """Interrupted while the original (1), retried (2) or final (3) request is in flight and a peer
    holds ``context.lock``: the peer keeps it, then the flow cleans up and re-raises."""
    errors = _capture_loop_errors()
    baseline = set(asyncio.all_tasks())
    provider, _, _ = await _provider(tmp_path, monkeypatch)
    lock = provider.context.lock
    server = _Server()
    server.resource_status = [401, 401]
    reached = asyncio.Event()
    server.gates[("resource", nth)] = reached
    server.block.add(("resource", nth))
    async with _client(provider, server) as client:
        task = _start(client, mode, _rpc_request())
        await asyncio.wait_for(reached.wait(), 5)
        peer = _Peer(lock)
        await asyncio.wait_for(peer.holding.wait(), 5)
        await _interrupt(task, mode)
        await _settle()
        assert _peer_still_holds(lock)
        assert not task.done()
        peer.release.set()
        await peer.task
        await _finish(task, mode)
    assert server.count("refresh") == (1 if nth == 3 else 0)
    await _assert_clean_after(provider, errors, baseline)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cancel", "timeout"])
async def test_interrupt_during_refresh_releases_fence_then_peer_proceeds(tmp_path, monkeypatch, mode):
    from tools.mcp_oauth import acquire_refresh_fence, release_refresh_fence

    errors = _capture_loop_errors()
    baseline = set(asyncio.all_tasks())
    provider, storage, _ = await _provider(tmp_path, monkeypatch)
    lock = provider.context.lock
    server = _Server()
    server.resource_status = [401, 401]
    reached = asyncio.Event()
    server.gates[("refresh", 1)] = reached
    server.block.add(("refresh", 1))
    async with _client(provider, server) as client:
        task = _start(client, mode, _rpc_request())
        await asyncio.wait_for(reached.wait(), 5)
        assert provider._hermes_fence is not None
        peer = _Peer(lock)  # waits: the refreshing flow owns the lock
        await _settle()
        assert not peer.holding.is_set()
        await _interrupt(task, mode)
        await _finish(task, mode)
        await asyncio.wait_for(peer.holding.wait(), 5)
        assert _peer_still_holds(lock)
        peer.release.set()
        await peer.task
    assert server.count("resource") == 2 and server.count("refresh") == 1
    await _assert_clean_after(provider, errors, baseline)
    fd = await acquire_refresh_fence(storage._tokens_path(), timeout=1.0)
    release_refresh_fence(fd)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cancel", "timeout"])
async def test_interrupt_while_draining_discarded_401(tmp_path, monkeypatch, mode):
    errors = _capture_loop_errors()
    baseline = set(asyncio.all_tasks())
    provider, _, _ = await _provider(tmp_path, monkeypatch)
    lock = provider.context.lock
    server = _Server()
    server.resource_status = [401]
    server.blocking_body.add(("resource", 1))
    reached = asyncio.Event()
    server.gates[("resource", 1)] = reached
    async with _client(provider, server) as client:
        task = _start(client, mode, _rpc_request())
        await asyncio.wait_for(reached.wait(), 5)
        await _settle()
        peer = _Peer(lock)  # waits behind the draining flow
        await _settle()
        await _interrupt(task, mode)
        await _finish(task, mode)
        await asyncio.wait_for(peer.holding.wait(), 5)
        peer.release.set()
        await peer.task
    assert server.count("resource") == 1
    await _assert_clean_after(provider, errors, baseline)


@pytest.mark.asyncio
async def test_retry_path_stays_bidirectional(tmp_path, monkeypatch):
    """The response HTTPX sends into the flow is the one the retry logic sees."""
    provider, _, _ = await _provider(tmp_path, monkeypatch)
    req = _rpc_request()
    flow = provider.async_auth_flow(req)
    out = await flow.__anext__()
    httpx = _httpx()
    retry = await flow.asend(httpx.Response(401, request=out, headers=WWW_AUTH, content=b"{}"))
    assert retry is req
    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx.Response(200, request=retry, json=OK_RPC))
    _assert_idle(provider)


# ── 4. ambiguous payloads are never recoverable ───────────────────────────────

AMBIGUOUS = {
    "method_list": b'{"jsonrpc":"2.0","id":1,"method":[]}',
    "method_dict": b'{"jsonrpc":"2.0","id":1,"method":{}}',
    "method_number": b'{"jsonrpc":"2.0","id":1,"method":1}',
    "nan_id": b'{"jsonrpc":"2.0","id":NaN,"method":"tools/list"}',
    "infinity_param": b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"x":Infinity}}',
    "neg_infinity_arg": (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
                         b'{"name":"get_account_info","arguments":{"x":-Infinity}}}'),
    "overflow_float": b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"x":1e999}}',
    "duplicate_method": b'{"jsonrpc":"2.0","id":1,"method":"tools/list","method":"tools/call"}',
    "duplicate_method_safe_last": (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","method":"tools/list",'
                                   b'"params":{"name":"update_pocket_actionitem"}}'),
    "duplicate_name": (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
                       b'{"name":"update_pocket_actionitem","name":"get_account_info"}}'),
    "duplicate_nested_arg": (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
                             b'{"name":"get_account_info","arguments":{"a":1,"a":2}}}'),
    "duplicate_id": b'{"jsonrpc":"2.0","id":1,"id":2,"method":"tools/list"}',
    "deep_nesting": b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"x":' + b"[" * 100000
                    + b"]" * 100000 + b"}}",
}


@pytest.mark.parametrize("case", sorted(AMBIGUOUS))
def test_ambiguous_payload_is_not_recoverable(case):
    from tools.mcp_oauth_pocket import is_recoverable_read_request

    assert is_recoverable_read_request("pocket", _rpc_request(raw=AMBIGUOUS[case])) is False


@pytest.mark.parametrize("raw", [
    b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}',
    b'{"jsonrpc":"2.0","id":"a","method":"tools/call","params":{"name":"get_account_info","arguments":{"q":1.5}}}',
])
def test_unambiguous_read_payload_stays_recoverable(raw):
    from tools.mcp_oauth_pocket import is_recoverable_read_request

    assert is_recoverable_read_request("pocket", _rpc_request(raw=raw)) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["method_list", "duplicate_method_safe_last", "nan_id"])
async def test_ambiguous_payload_401_is_not_retried(tmp_path, monkeypatch, case):
    provider, _, _ = await _provider(tmp_path, monkeypatch)
    req = _rpc_request(raw=AMBIGUOUS[case])
    flow = provider.async_auth_flow(req)
    out = await flow.__anext__()
    httpx = _httpx()
    try:
        nxt = await flow.asend(httpx.Response(401, request=out, headers=WWW_AUTH, content=b"{}"))
    except RuntimeError:
        nxt = None
    assert nxt is not req and (nxt is None or str(nxt.url) != TOKEN_URL)
    await flow.aclose()
    assert provider._hermes_fence is None
