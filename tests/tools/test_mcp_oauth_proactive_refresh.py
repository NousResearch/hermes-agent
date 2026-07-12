"""Tests for MCP OAuth proactive-refresh timer lifecycle (#62309).

Covers the review feedback on the original PR:
  1. schedule cancels any prior handle before installing a new one
  2. manager.remove() cancels any pending timer
  3. permanent HTTP failures do NOT re-schedule retries
  4. transient HTTP/transport failures DO re-schedule (while refreshable)
  5. successful refresh chains the next schedule
  6. re-_initialize cancels the previous timer (no orphaned duplicates)
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

from tools.mcp_oauth import HermesTokenStorage, _get_token_dir
from tools.mcp_oauth_manager import (
    _HERMES_PROVIDER_CLS,
    _PROACTIVE_REFRESH_LEAD_SECONDS,
    _PROACTIVE_REFRESH_MAX_DELAY_SECONDS,
    _PROACTIVE_REFRESH_RETRY_SECONDS,
    get_manager,
    reset_manager_for_tests,
)


async def _noop_redirect(_url: str) -> None:
    return None


async def _noop_callback() -> tuple[str, str | None]:
    raise AssertionError("callback should not run in these unit tests")


def _client_metadata() -> OAuthClientMetadata:
    return OAuthClientMetadata(
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        client_name="Hermes Agent",
    )


def _client_info() -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id="test-client",
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    )


async def _fresh_provider(tmp_path, monkeypatch, expires_in: int = 3600):
    """Build an initialized HermesMCPOAuthProvider with a stored live token."""
    assert _HERMES_PROVIDER_CLS is not None
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="live",
            token_type="Bearer",
            expires_in=expires_in,
            refresh_token="refresh-token",
        )
    )
    await storage.set_client_info(_client_info())

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=_client_metadata(),
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )
    await provider._initialize()
    return provider


# ---------------------------------------------------------------------------
# Schedule / cancel mechanics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_cancels_previous_timer(tmp_path, monkeypatch):
    """Re-scheduling must cancel the previous TimerHandle before installing.

    Without cancel-before-replace, re-initialize after disk invalidation can
    leave orphaned callbacks racing refresh_token rotation.
    """
    provider = await _fresh_provider(tmp_path, monkeypatch)
    assert provider._hermes_refresh_timer is not None
    first = provider._hermes_refresh_timer

    # Force a second schedule (e.g. what re-initialize would do after cancel).
    provider._schedule_proactive_refresh()
    second = provider._hermes_refresh_timer

    assert second is not None
    assert second is not first
    assert first.cancelled() is True
    assert second.cancelled() is False

    # Cleanup
    provider._cancel_proactive_refresh()


@pytest.mark.asyncio
async def test_reinitialize_cancels_prior_timer(tmp_path, monkeypatch):
    """A second _initialize must not leave the previous timer live."""
    provider = await _fresh_provider(tmp_path, monkeypatch)
    first = provider._hermes_refresh_timer
    assert first is not None

    await provider._initialize()
    second = provider._hermes_refresh_timer
    assert second is not None
    assert first.cancelled() is True
    assert second is not first

    provider._cancel_proactive_refresh()


@pytest.mark.asyncio
async def test_manager_remove_cancels_timer(tmp_path, monkeypatch):
    """Eviction via manager.remove must cancel the timer on the way out."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    # Seed disk + construct a provider, then hang it on a manager entry.
    provider = await _fresh_provider(tmp_path, monkeypatch)
    timer = provider._hermes_refresh_timer
    assert timer is not None and timer.cancelled() is False

    mgr = get_manager()
    # Inject the live provider into the manager entry map.
    from tools.mcp_oauth_manager import _ProviderEntry

    entry = _ProviderEntry(
        server_url="https://example.com/mcp",
        oauth_config={},
        provider=provider,
    )
    with mgr._entries_lock:
        mgr._entries["srv"] = entry

    mgr.remove("srv")

    assert timer.cancelled() is True
    assert provider._hermes_refresh_timer is None
    assert "srv" not in mgr._entries


@pytest.mark.asyncio
async def test_schedule_respects_lead_and_cap(tmp_path, monkeypatch):
    """Default delay = min(expiry - now - 300, 3300) and never negative."""
    # Very long TTL — cap should engage.
    provider = await _fresh_provider(tmp_path, monkeypatch, expires_in=100_000)
    handle = provider._hermes_refresh_timer
    assert handle is not None
    # call_later stores absolute when; we can only assert the handle exists
    # and cancelled/replaced works. Delay knowledge is exercised via a
    # controlled FakeLoop below.
    provider._cancel_proactive_refresh()

    scheduled: list[float] = []

    class _FakeHandle:
        def __init__(self):
            self._cancelled = False

        def cancel(self):
            self._cancelled = True

        def cancelled(self):
            return self._cancelled

    class _FakeLoop:
        def call_later(self, delay, cb):
            scheduled.append(float(delay))
            return _FakeHandle()

    with patch(
        "tools.mcp_oauth_manager.asyncio.get_running_loop",
        return_value=_FakeLoop(),
    ):
        # Far-future expiry → capped at 3300
        provider.context.token_expiry_time = time.time() + 100_000
        provider._schedule_proactive_refresh()
        assert scheduled[-1] == float(_PROACTIVE_REFRESH_MAX_DELAY_SECONDS)

        # Near-future expiry (200s) → max(0, 200-300) = 0
        provider.context.token_expiry_time = time.time() + 200
        provider._schedule_proactive_refresh()
        assert scheduled[-1] == 0.0

        # Mid expiry → lead of 300s
        provider.context.token_expiry_time = time.time() + 1000
        provider._schedule_proactive_refresh()
        expected = 1000 - _PROACTIVE_REFRESH_LEAD_SECONDS
        assert abs(scheduled[-1] - expected) < 1.0


# ---------------------------------------------------------------------------
# Failure / success policy in _do_proactive_refresh
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, body: bytes = b"{}"):
        self.status_code = status_code
        self._body = body

    async def aread(self) -> bytes:
        return self._body


class _FakeClient:
    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.sent = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def send(self, _req):
        self.sent += 1
        if self._error is not None:
            raise self._error
        return self._response


async def _provider_ready_for_refresh(tmp_path, monkeypatch):
    provider = await _fresh_provider(tmp_path, monkeypatch)
    # Drop the auto-scheduled timer so tests drive _do_proactive_refresh
    # themselves without auto-firing.
    provider._cancel_proactive_refresh()
    # Stub _refresh_token so we never hit real httpx construction internals.
    provider._refresh_token = AsyncMock(return_value=MagicMock())
    return provider


@pytest.mark.asyncio
async def test_success_chains_next_schedule(tmp_path, monkeypatch):
    provider = await _provider_ready_for_refresh(tmp_path, monkeypatch)
    fresh = OAuthToken(
        access_token="new",
        token_type="Bearer",
        expires_in=3600,
        refresh_token="refresh-token",
    )
    provider._handle_refresh_response = AsyncMock(return_value=True)

    # Make update paths in the real handler unreachable by short-circuiting it;
    # also seed can_refresh_token to keep truthy via the existing context.
    client = _FakeClient(response=_FakeResponse(200))
    with patch("httpx.AsyncClient", return_value=client):
        # After success we expect a schedule; patch loop.
        scheduled: list[float] = []

        class _H:
            def __init__(self):
                self._c = False

            def cancel(self):
                self._c = True

            def cancelled(self):
                return self._c

        class _L:
            def call_later(self, delay, cb):
                scheduled.append(float(delay))
                return _H()

        with patch(
            "tools.mcp_oauth_manager.asyncio.get_running_loop",
            return_value=_L(),
        ):
            # But _do uses a real httpx path after get_running_loop for schedule;
            # patch only for the schedule call by letting the body run, then
            # intercepting _schedule_proactive_refresh.
            calls: list[dict] = []

            def _track_schedule(*, delay=None):
                calls.append({"delay": delay})
                # don't install a real timer

            provider._schedule_proactive_refresh = _track_schedule  # type: ignore
            # Also make sure _handle_refresh_response still posts success and
            # leaves can_refresh_token True.
            provider.context.current_tokens = fresh
            provider.context.update_token_expiry(fresh)

            await provider._do_proactive_refresh()

    assert provider._handle_refresh_response.await_count == 1
    assert len(calls) == 1
    assert calls[0]["delay"] is None  # success → full re-schedule (lead calc)


@pytest.mark.asyncio
async def test_permanent_http_failure_does_not_retry(tmp_path, monkeypatch):
    provider = await _provider_ready_for_refresh(tmp_path, monkeypatch)
    client = _FakeClient(response=_FakeResponse(400))
    scheduled: list = []
    provider._schedule_proactive_refresh = (  # type: ignore
        lambda **kw: scheduled.append(kw)
    )

    with patch("httpx.AsyncClient", return_value=client):
        await provider._do_proactive_refresh()

    assert scheduled == []
    # Permanent failure must NOT wipe tokens (we never called the SDK
    # _handle_refresh_response on non-200 in this path).
    assert provider.context.can_refresh_token() is True


@pytest.mark.asyncio
async def test_transient_http_failure_retries(tmp_path, monkeypatch):
    provider = await _provider_ready_for_refresh(tmp_path, monkeypatch)
    client = _FakeClient(response=_FakeResponse(503))
    scheduled: list = []
    provider._schedule_proactive_refresh = (  # type: ignore
        lambda **kw: scheduled.append(kw)
    )

    with patch("httpx.AsyncClient", return_value=client):
        await provider._do_proactive_refresh()

    assert len(scheduled) == 1
    assert scheduled[0]["delay"] == float(_PROACTIVE_REFRESH_RETRY_SECONDS)


@pytest.mark.asyncio
async def test_transport_error_retries(tmp_path, monkeypatch):
    import httpx

    provider = await _provider_ready_for_refresh(tmp_path, monkeypatch)
    client = _FakeClient(error=httpx.ConnectError("boom"))
    scheduled: list = []
    provider._schedule_proactive_refresh = (  # type: ignore
        lambda **kw: scheduled.append(kw)
    )

    with patch("httpx.AsyncClient", return_value=client):
        await provider._do_proactive_refresh()

    assert len(scheduled) == 1
    assert scheduled[0]["delay"] == float(_PROACTIVE_REFRESH_RETRY_SECONDS)


@pytest.mark.asyncio
async def test_cannot_refresh_skips_without_schedule(tmp_path, monkeypatch):
    provider = await _provider_ready_for_refresh(tmp_path, monkeypatch)
    # Wipe refreshability.
    provider.context.current_tokens = OAuthToken(
        access_token="live",
        token_type="Bearer",
        expires_in=3600,
        refresh_token=None,
    )
    scheduled: list = []
    provider._schedule_proactive_refresh = (  # type: ignore
        lambda **kw: scheduled.append(kw)
    )
    await provider._do_proactive_refresh()
    assert scheduled == []


@pytest.mark.asyncio
async def test_transient_does_not_retry_after_tokens_cleared(tmp_path, monkeypatch):
    """If can_refresh_token is False mid-failure, don't re-queue."""
    import httpx

    provider = await _provider_ready_for_refresh(tmp_path, monkeypatch)

    # Make can_refresh_token False by clearing current_tokens AFTER send fails.
    client = _FakeClient(error=httpx.ReadTimeout("slow"))

    original_can = provider.context.can_refresh_token

    def _false_after_error():
        return False

    scheduled: list = []
    provider._schedule_proactive_refresh = (  # type: ignore
        lambda **kw: scheduled.append(kw)
    )

    # First can_refresh_token (pre-flight) returns True, post-error returns False.
    calls = {"n": 0}

    def _can():
        calls["n"] += 1
        # 1st call is the pre-flight gate → True; subsequent → False
        return calls["n"] == 1

    provider.context.can_refresh_token = _can  # type: ignore

    with patch("httpx.AsyncClient", return_value=client):
        await provider._do_proactive_refresh()

    assert scheduled == []


# ---------------------------------------------------------------------------
# expires_in clamp (end-to-end through SDK is_token_valid)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_clamp_neg1_makes_sdk_invalid(tmp_path, monkeypatch):
    """Fresh proof that -1 (not 0) is what makes is_token_valid return False.

    This is the Fix-A end-to-end: disk has expired token → get_tokens returns
    expires_in=-1 → update_token_expiry → is_token_valid is False.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    token_dir = _get_token_dir()
    token_dir.mkdir(parents=True, exist_ok=True)
    (token_dir / "srv.json").write_text(
        json.dumps(
            {
                "access_token": "stale",
                "token_type": "Bearer",
                "expires_in": 3600,
                "expires_at": time.time() - 60,
                "refresh_token": "fresh",
            }
        )
    )

    storage = HermesTokenStorage("srv")
    await storage.set_client_info(_client_info())
    tokens = await storage.get_tokens()
    assert tokens is not None
    assert tokens.expires_in == -1

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=_client_metadata(),
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )
    await provider._initialize()
    provider._cancel_proactive_refresh()

    assert provider.context.is_token_valid() is False
    assert provider.context.can_refresh_token() is True
