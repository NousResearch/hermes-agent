"""Regression tests for the MCP OAuth refresh-token rotation race.

Asana (and other single-use-refresh-token providers) rotates the refresh
token on every use. When two Hermes *processes* — the long-running gateway,
a one-shot ``hermes chat`` backfill, and ``hermes mcp test/login`` runs —
share one token file (``~/.hermes/mcp-tokens/<server>.json``) and both
decide to refresh the same expired token, they can both POST the same
refresh token. The provider accepts one and invalidates the other; the
losing client falls through to a full browser re-authorization even though a
fresh, valid token was just written by the winner.

The fix mirrors the cross-process single-refresher discipline already used
for Codex/xAI/Anthropic in ``agent/credential_pool.py``: a cross-process
advisory flock around the whole read-fresh -> POST -> write-back sequence,
plus a re-read of the token file once the lock is held so a waiter adopts
the winner's rotated token instead of re-POSTing the now-spent one.

These tests reproduce the race deterministically with a fake single-use
token endpoint, run two providers against one token file, and assert exactly
one rotation with no invalidation loop.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


def _fake_response(status: int, body: bytes):
    """A minimal stand-in for the httpx2.Response the SDK feeds our handlers."""
    resp = MagicMock()
    resp.status_code = status

    async def _aread():
        return body

    resp.aread = _aread
    return resp


class _SingleUseTokenEndpoint:
    """Fake token endpoint enforcing the real single-use-refresh contract.

    The first caller to present a given refresh_token gets a fresh pair; every
    subsequent caller presenting that same (now-spent) token gets
    ``invalid_grant`` — mirroring Asana's rotation behavior.
    """

    def __init__(self, latency: float = 0.02):
        self._spent: set[str] = set()
        self._rotation = 0
        self.calls: list[str] = []
        self._latency = latency

    async def post(self, refresh_token: str):
        self.calls.append(refresh_token)
        # Widen the race window so a second client is deterministically forced
        # to wait on the cross-process lock while the winner is mid-refresh.
        await asyncio.sleep(self._latency)
        if refresh_token in self._spent:
            return 400, b'{"error":"invalid_grant"}'
        self._spent.add(refresh_token)
        self._rotation += 1
        rotation = self._rotation
        body = json.dumps(
            {
                "access_token": f"at-{rotation}",
                "refresh_token": f"rt-{rotation}",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
        ).encode()
        return 200, body


async def _refresh_once(provider, endpoint):
    """Drive one refresh cycle the way the async_auth_flow bridge does.

    Mirrors the SDK's ``_refresh_token() -> POST -> _handle_refresh_response()``
    sequence, including the skip branch where a sibling already rotated the
    token and the POST must be omitted.
    """
    from urllib.parse import parse_qs

    refresh_request = await provider._refresh_token()
    if getattr(provider, "_hermes_refresh_skipped", False):
        # The bridge feeds None instead of POSTing the spent token.
        return await provider._handle_refresh_response(None)
    refresh_token = parse_qs(refresh_request.content.decode())["refresh_token"][0]
    status, body = await endpoint.post(refresh_token)
    return await provider._handle_refresh_response(_fake_response(status, body))


# ---------------------------------------------------------------------------
# Provider construction helper
# ---------------------------------------------------------------------------


def _make_provider(tmp_path, monkeypatch, *, in_memory_token):
    """Build a provider whose storage points at the shared tmp_path token file."""
    from mcp.shared.auth import OAuthToken, OAuthClientInformationFull

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    from tools.mcp_oauth_manager import MCPOAuthManager

    provider = MCPOAuthManager().get_or_build_provider(
        "srv", "https://mcp.example.com", None
    )
    assert provider is not None
    provider.context.current_tokens = OAuthToken(
        access_token=in_memory_token["access_token"],
        refresh_token=in_memory_token["refresh_token"],
        expires_in=3600,
    )
    provider.context.client_info = OAuthClientInformationFull.model_validate(
        {"client_id": "client-id"}
    )
    provider.context.oauth_metadata = SimpleNamespace(
        token_endpoint="https://idp.example.com/oauth/token"
    )
    return provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_token_adopts_rotated_disk_token(tmp_path, monkeypatch):
    """_refresh_token re-reads disk and adopts a sibling's rotated token.

    The in-memory token (rt-0) is stale; the file already holds rt-1 because
    another process rotated it. _refresh_token must adopt rt-1 and mark the
    POST to skip rather than re-POSTing the spent rt-0.
    """
    from mcp.shared.auth import OAuthToken
    from tools.mcp_oauth import HermesTokenStorage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(access_token="at-1", refresh_token="rt-1", expires_in=3600)
    )

    provider = _make_provider(
        tmp_path,
        monkeypatch,
        in_memory_token={"access_token": "at-0", "refresh_token": "rt-0"},
    )

    await provider._refresh_token()

    assert provider.context.current_tokens.refresh_token == "rt-1"
    assert provider.context.current_tokens.access_token == "at-1"
    assert getattr(provider, "_hermes_refresh_skipped", False) is True

    # The skip branch must return True (no clear) and release the lock.
    assert await provider._handle_refresh_response(None) is True
    assert getattr(provider, "_hermes_refresh_lock", None) is None


@pytest.mark.asyncio
async def test_refresh_failure_releases_cross_process_lock(tmp_path, monkeypatch):
    """A failed refresh POST must still release the cross-process lock.

    Without release-on-failure, one dead refresh wedges every other process
    on the flock and turns a transient token failure into a fleet-wide hang.
    """
    from mcp.shared.auth import OAuthToken
    from tools.mcp_oauth import HermesTokenStorage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(access_token="at-0", refresh_token="rt-0", expires_in=3600)
    )

    provider = _make_provider(
        tmp_path,
        monkeypatch,
        in_memory_token={"access_token": "at-0", "refresh_token": "rt-0"},
    )

    await provider._refresh_token()
    assert getattr(provider, "_hermes_refresh_lock", None) is not None
    assert getattr(provider, "_hermes_refresh_skipped", False) is False

    result = await provider._handle_refresh_response(
        _fake_response(400, b'{"error":"invalid_grant"}')
    )

    assert result is False
    assert provider.context.current_tokens is None  # cleared on failure
    assert getattr(provider, "_hermes_refresh_lock", None) is None  # released
    # The attribute being None is not proof: _hermes_release_refresh_lock nulls
    # it before lock.release(). Prove the flock is genuinely free by acquiring
    # it again (with a short timeout) — a wedged flock would hang/raise here.
    relock = storage.refresh_lock(timeout_seconds=2)
    await relock.acquire()
    relock.release()


@pytest.mark.asyncio
async def test_concurrent_refresh_rotates_once_without_invalidation_loop(
    tmp_path, monkeypatch
):
    """Two clients sharing one token file rotate exactly once, no invalidation loop.

    This is the core acceptance: the cross-process lock plus read-fresh means
    exactly one process POSTs the refresh token and the other adopts the
    winner's rotated pair — so the second client never burns the single-use
    token and never falls through to a browser re-authorization.
    """
    from mcp.shared.auth import OAuthToken
    from tools.mcp_oauth import HermesTokenStorage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(access_token="at-0", refresh_token="rt-0", expires_in=3600)
    )

    provider_a = _make_provider(
        tmp_path,
        monkeypatch,
        in_memory_token={"access_token": "at-0", "refresh_token": "rt-0"},
    )
    provider_b = _make_provider(
        tmp_path,
        monkeypatch,
        in_memory_token={"access_token": "at-0", "refresh_token": "rt-0"},
    )
    # Two independent "processes": separate providers, separate storages, one file.
    assert provider_a is not provider_b
    assert provider_a.context.storage is not provider_b.context.storage
    assert (
        provider_a.context.storage._tokens_path()
        == provider_b.context.storage._tokens_path()
    )

    endpoint = _SingleUseTokenEndpoint()

    await asyncio.gather(
        _refresh_once(provider_a, endpoint),
        _refresh_once(provider_b, endpoint),
    )

    # Exactly one process ever posted, and only the original token was used.
    assert endpoint.calls == ["rt-0"], f"unexpected refresh POSTs: {endpoint.calls!r}"
    # Both clients converged on the same fresh pair — no invalidation loop.
    assert provider_a.context.current_tokens.refresh_token == "rt-1"
    assert provider_b.context.current_tokens.refresh_token == "rt-1"
    # The rotation is durable on disk.
    disk = await storage.get_tokens()
    assert disk is not None
    assert disk.refresh_token == "rt-1"


def test_bridge_skips_posting_when_token_adopted(tmp_path, monkeypatch):
    """The async_auth_flow bridge omits the POST when _refresh_token skipped.

    The read-fresh/adopt logic alone is not enough: the bridge must not yield
    the (now-pointless) refresh request to httpx. This drives the real bridge
    against a patched SDK base flow and asserts the only outgoing is the
    resource request, never the refresh POST.
    """
    from mcp.shared.auth import OAuthToken, OAuthClientInformationFull

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    from tools.mcp_oauth import HermesTokenStorage

    storage = HermesTokenStorage("srv")
    asyncio.run(
        storage.set_tokens(
            OAuthToken(access_token="at-1", refresh_token="rt-1", expires_in=3600)
        )
    )

    provider = _make_provider(
        tmp_path,
        monkeypatch,
        in_memory_token={"access_token": "at-0", "refresh_token": "rt-0"},
    )
    provider.context.update_token_expiry(provider.context.current_tokens)
    # The refresh is skipped here, so no real metadata is needed; leaving the
    # SimpleNamespace in place would trip _persist_oauth_metadata_if_changed,
    # which expects a real OAuthMetadata.model_dump().
    provider.context.oauth_metadata = None

    async def fake_base_flow(self, request):
        """Mimic the SDK's refresh-then-request sequence in a controlled way."""
        async with self.context.lock:
            if self.context.can_refresh_token():
                refresh_request = await self._refresh_token()
                refresh_response = yield refresh_request
                if not await self._handle_refresh_response(refresh_response):
                    self._initialized = False
            if self.context.is_token_valid():
                self._add_auth_header(request)
            yield request

    from mcp.client.auth.oauth2 import OAuthClientProvider

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", fake_base_flow)

    sentinel = SimpleNamespace(headers={})
    yielded = []

    async def drive():
        gen = provider.async_auth_flow(sentinel)
        while True:
            try:
                out = await gen.__anext__()
            except StopAsyncIteration:
                return
            yielded.append(out)
            try:
                await gen.asend(None)
            except StopAsyncIteration:
                return

    asyncio.run(drive())

    assert yielded == [sentinel], (
        f"bridge yielded {len(yielded)} outgoing request(s); the refresh POST "
        f"should have been skipped because the token was already rotated"
    )


@pytest.mark.asyncio
async def test_stale_skip_flag_does_not_swallow_resource_request(tmp_path, monkeypatch):
    """A skip flag leaked by an aborted refresh must not swallow requests (F1).

    The bridge's skip branch is positional: before the fix, a
    ``_hermes_refresh_skipped`` left True by a cancelled refresh would suppress
    the *next* flow's first ``yield`` even when that yield was the resource
    request itself, silently dropping the MCP call for the provider's lifetime.
    This drives the real bridge (real SDK generator) with a stuck flag and a
    valid token and asserts the resource request is still delivered and the
    flag is cleared.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    provider = _make_provider(
        tmp_path,
        monkeypatch,
        in_memory_token={"access_token": "at-0", "refresh_token": "rt-0"},
    )
    # Valid token -> the SDK yields the resource request as its first outgoing.
    provider.context.update_token_expiry(provider.context.current_tokens)
    provider._initialized = True
    provider.context.oauth_metadata = None
    # Simulate the flag leaked by an aborted refresh (the F1 wedge).
    provider._hermes_refresh_skipped = True

    sentinel = SimpleNamespace(headers={})
    gen = provider.async_auth_flow(sentinel)

    # The resource request must be delivered, not swallowed.
    out = await gen.__anext__()
    assert out is sentinel

    # Completing the flow must clear the leaked flag (bridge finally).
    try:
        await gen.asend(_fake_response(200, b"{}"))
    except StopAsyncIteration:
        pass
    assert provider._hermes_refresh_skipped is False


@pytest.mark.asyncio
async def test_adopt_expired_rotated_token_still_posts_refresh(tmp_path, monkeypatch):
    """Adopting an already-expired rotated pair must NOT skip the POST (F2).

    ``get_tokens()`` rewrites ``expires_in`` to remaining seconds (0 when
    expired), so a sibling may have rotated the refresh token into a pair that
    has since lapsed. Adopting that pair and *skipping* the refresh would drop
    straight into browser re-auth; the correct action is to adopt the unspent
    refresh token and still POST it.
    """
    from urllib.parse import parse_qs

    from mcp.shared.auth import OAuthToken
    from tools.mcp_oauth import HermesTokenStorage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    storage = HermesTokenStorage("srv")
    # Disk holds a rotated pair whose access token has already expired.
    await storage.set_tokens(
        OAuthToken(access_token="at-1", refresh_token="rt-1", expires_in=0)
    )

    provider = _make_provider(
        tmp_path,
        monkeypatch,
        in_memory_token={"access_token": "at-0", "refresh_token": "rt-0"},
    )

    refresh_request = await provider._refresh_token()

    # Adopted the rotated pair (rt-1) but, because it is expired, the POST must
    # NOT be skipped — the request is still built from the adopted token.
    assert provider.context.current_tokens.refresh_token == "rt-1"
    assert getattr(provider, "_hermes_refresh_skipped", False) is False

    refresh_token = parse_qs(refresh_request.content.decode())["refresh_token"][0]
    assert refresh_token == "rt-1"  # POSTs the adopted, unspent refresh token

    # Lock stays held until _handle_refresh_response, then released on success.
    assert getattr(provider, "_hermes_refresh_lock", None) is not None
    result = await provider._handle_refresh_response(
        _fake_response(
            200,
            json.dumps(
                {
                    "access_token": "at-2",
                    "refresh_token": "rt-2",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                }
            ).encode(),
        )
    )
    assert result is True
    assert provider.context.current_tokens.refresh_token == "rt-2"
    assert getattr(provider, "_hermes_refresh_lock", None) is None


@pytest.mark.asyncio
async def test_concurrent_real_bridge_rotates_once(tmp_path, monkeypatch):
    """Two real bridges sharing one token file rotate exactly once (F3).

    Drives the actual ``async_auth_flow`` bridge (real SDK generator) for two
    providers against one token file, asserting exactly one refresh POST and
    that both clients converge on the winner's rotated pair. This exercises the
    production skip branch under concurrency — not the re-implemented helper —
    so the bridge's None-sentinel contract is validated against the real SDK.
    """
    import time
    from urllib.parse import parse_qs

    from mcp.shared.auth import OAuthToken
    from tools.mcp_oauth import HermesTokenStorage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(access_token="at-0", refresh_token="rt-0", expires_in=3600)
    )

    def _stale_provider():
        p = _make_provider(
            tmp_path,
            monkeypatch,
            in_memory_token={"access_token": "at-0", "refresh_token": "rt-0"},
        )
        # Stale in-memory token -> the SDK takes the refresh branch.
        p.context.token_expiry_time = time.time() - 3600
        p._initialized = True
        p.context.oauth_metadata = None
        return p

    provider_a = _stale_provider()
    provider_b = _stale_provider()
    assert provider_a.context.storage is not provider_b.context.storage
    assert (
        provider_a.context.storage._tokens_path()
        == provider_b.context.storage._tokens_path()
    )

    endpoint = _SingleUseTokenEndpoint()
    outcomes: list[str] = []

    async def drive(provider, sentinel):
        gen = provider.async_auth_flow(sentinel)
        out = await gen.__anext__()
        if out is sentinel:
            # Loser: adopted the winner's rotated pair, delivered the resource
            # request without POSTing the now-spent token.
            outcomes.append("skip")
        else:
            # Winner: the first outgoing is the refresh POST.
            outcomes.append("post")
            refresh_token = parse_qs(out.content.decode())["refresh_token"][0]
            status, body = await endpoint.post(refresh_token)
            out2 = await gen.asend(_fake_response(status, body))
            assert out2 is sentinel
        try:
            await gen.asend(_fake_response(200, b"{}"))
        except StopAsyncIteration:
            pass

    await asyncio.gather(
        drive(provider_a, SimpleNamespace(headers={})),
        drive(provider_b, SimpleNamespace(headers={})),
    )

    # Exactly one process ever posted, and only the original token was used.
    assert sorted(outcomes) == ["post", "skip"], f"outcomes: {outcomes!r}"
    assert endpoint.calls == ["rt-0"], f"unexpected refresh POSTs: {endpoint.calls!r}"
    # Both clients converged on the same fresh pair — no invalidation loop.
    assert provider_a.context.current_tokens.refresh_token == "rt-1"
    assert provider_b.context.current_tokens.refresh_token == "rt-1"
    disk = await storage.get_tokens()
    assert disk is not None
    assert disk.refresh_token == "rt-1"


@pytest.mark.asyncio
async def test_bridge_cancellation_releases_flock(tmp_path, monkeypatch):
    """Closing the bridge mid-refresh must release the cross-process flock.

    The bridge ``async_auth_flow`` ``finally`` releases the flock when the
    generator is cancelled after ``_refresh_token`` acquired it but before
    ``_handle_refresh_response`` released it. Without this, one cancelled
    refresh leaks the flock and wedges every other process on this token file
    for the lock timeout.
    """
    from mcp.shared.auth import OAuthToken
    from tools.mcp_oauth import HermesTokenStorage

    import time

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(access_token="at-0", refresh_token="rt-0", expires_in=3600)
    )

    provider = _make_provider(
        tmp_path,
        monkeypatch,
        in_memory_token={"access_token": "at-0", "refresh_token": "rt-0"},
    )
    provider._initialized = True
    # Expire the in-memory token so the SDK takes the refresh branch; disk and
    # memory agree (no adopt), so _refresh_token acquires the flock and yields
    # a real refresh POST while still holding it.
    provider.context.token_expiry_time = time.time() - 5.0
    provider.context.oauth_metadata = None

    sentinel = SimpleNamespace(headers={})
    gen = provider.async_auth_flow(sentinel)
    outgoing = await gen.__anext__()
    # The first outgoing is the refresh POST (not the resource request), and
    # the cross-process flock is held across it.
    assert outgoing is not sentinel
    assert getattr(provider, "_hermes_refresh_lock", None) is not None

    # Cancel the flow before feeding a response -> the bridge finally runs.
    await gen.aclose()

    assert getattr(provider, "_hermes_refresh_lock", None) is None
    # A fresh lock must acquire immediately, proving the flock was released
    # rather than merely nulling the provider's attribute.
    relock = storage.refresh_lock(timeout_seconds=2)
    await relock.acquire()


@pytest.mark.asyncio
async def test_bridge_resource_cancellation_preserves_concurrent_refresh(
    tmp_path, monkeypatch
):
    """Closing the bridge at the resource yield must not stomp a sibling's refresh.

    The bridge releases ``context.lock`` around the resource request, so a
    second flow on the same cached provider can enter ``_refresh_token`` (set
    ``_hermes_refresh_skipped`` and hold ``_hermes_refresh_lock``) while this
    flow is suspended at the resource yield. If this flow is then cancelled,
    its ``finally`` used to clear the flag and release the flock
    UNCONDITIONALLY, re-POSTing the sibling's spent refresh token and
    reproducing the browser-re-auth loop. The ownership gate must leave a
    concurrent refresh's flag+flock untouched when this flow no longer owns
    them (``resource_lock_released`` is True at the resource yield).
    """
    from mcp.shared.auth import OAuthToken
    from tools.mcp_oauth import HermesTokenStorage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(access_token="at-0", refresh_token="rt-0", expires_in=3600)
    )

    provider = _make_provider(
        tmp_path,
        monkeypatch,
        in_memory_token={"access_token": "at-0", "refresh_token": "rt-0"},
    )
    # Valid token -> the SDK yields the resource request as its first outgoing,
    # so the bridge releases context.lock (resource_lock_released=True).
    provider.context.update_token_expiry(provider.context.current_tokens)
    provider._initialized = True
    provider.context.oauth_metadata = None

    # A "concurrent" flow has acquired the cross-process flock and set the skip
    # flag; this flow must not touch either when it is cancelled at the
    # resource yield.
    held_lock = storage.refresh_lock(timeout_seconds=2)
    await held_lock.acquire()
    provider._hermes_refresh_lock = held_lock
    provider._hermes_refresh_skipped = True

    sentinel = SimpleNamespace(headers={})
    gen = provider.async_auth_flow(sentinel)
    outgoing = await gen.__anext__()
    assert outgoing is sentinel  # first outgoing is the resource request

    await gen.aclose()

    # The concurrent refresh's state survives: the skip flag is untouched and
    # the held flock was not released (release() would null the fd handle).
    assert provider._hermes_refresh_skipped is True
    assert provider._hermes_refresh_lock is held_lock
    assert held_lock._fh is not None

    # Prove the flock is genuinely still held: a fresh lock on the same path
    # must not acquire within a short window.
    with pytest.raises(TimeoutError):
        await storage.refresh_lock(timeout_seconds=1).acquire()
