"""Tests for cold-load token expiry tracking in MCP OAuth.

PR #11383's consolidation fixed external-refresh reloading (mtime disk-watch)
and 401 dedup, but left two underlying latent bugs in place:

1. ``HermesTokenStorage.set_tokens`` persisted only relative ``expires_in``,
   which is meaningless after a process restart.
2. The MCP SDK's ``OAuthContext._initialize`` loads ``current_tokens`` from
   storage but does NOT call ``update_token_expiry``, so
   ``token_expiry_time`` stays None. ``is_token_valid()`` then returns True
   for any loaded token regardless of actual age, and the SDK's preemptive
   refresh branch at ``oauth2.py:491`` is never taken.

Consequence: a token that expired while the process was down ships to the
server with a stale Bearer header. The server's response is provider-specific
— some return HTTP 401 (caught by the consolidation's 401 handler, which
surfaces a ``needs_reauth`` error), others return HTTP 200 with an
application-level auth failure in the body (e.g. BetterStack's "No teams
found. Please check your authentication."), which the consolidation cannot
detect.

These tests pin the contract for Fix A:
- ``set_tokens`` persists an absolute ``expires_at`` wall-clock timestamp.
- ``get_tokens`` reconstructs ``expires_in`` from ``expires_at - now`` so
  the SDK's ``update_token_expiry`` computes the correct absolute expiry.
- ``HermesMCPOAuthProvider._initialize`` seeds ``context.token_expiry_time``
  after loading, so ``is_token_valid()`` reports True only for tokens that
  are actually still valid, and the SDK's preemptive refresh fires for
  expired tokens with a live refresh_token.

Reference: Claude Code solves this via an ``OAuthTokens.expiresAt`` absolute
timestamp persisted alongside the access_token (``auth.ts:~180``).
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


# ---------------------------------------------------------------------------
# HermesTokenStorage — absolute expiry persistence
# ---------------------------------------------------------------------------


class TestSetTokensAbsoluteExpiry:
    def test_set_tokens_persists_absolute_expires_at(self, tmp_path, monkeypatch):
        """Tokens round-tripped through disk must encode absolute expiry."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from mcp.shared.auth import OAuthToken

        from tools.mcp_oauth import HermesTokenStorage

        storage = HermesTokenStorage("srv")
        before = time.time()
        asyncio.run(
            storage.set_tokens(
                OAuthToken(
                    access_token="a",
                    token_type="Bearer",
                    expires_in=3600,
                    refresh_token="r",
                )
            )
        )
        after = time.time()

        on_disk = json.loads(
            (tmp_path / "mcp-tokens" / "srv.json").read_text()
        )
        assert "expires_at" in on_disk, (
            "Fix A: set_tokens must record an absolute expires_at wall-clock "
            "timestamp alongside the SDK's serialized token so cold-loads "
            "can compute correct remaining TTL."
        )
        assert before + 3600 <= on_disk["expires_at"] <= after + 3600

    def test_set_tokens_without_expires_in_omits_expires_at(
        self, tmp_path, monkeypatch
    ):
        """Tokens without a TTL must not gain a fabricated expires_at."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from mcp.shared.auth import OAuthToken

        from tools.mcp_oauth import HermesTokenStorage

        storage = HermesTokenStorage("srv")
        asyncio.run(
            storage.set_tokens(
                OAuthToken(
                    access_token="a",
                    token_type="Bearer",
                    refresh_token="r",
                )
            )
        )

        on_disk = json.loads(
            (tmp_path / "mcp-tokens" / "srv.json").read_text()
        )
        assert "expires_at" not in on_disk


class TestGetTokensReconstructsExpiresIn:
    def test_get_tokens_uses_expires_at_for_remaining_ttl(
        self, tmp_path, monkeypatch
    ):
        """Round-trip: expires_in on read must reflect time remaining."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from mcp.shared.auth import OAuthToken

        from tools.mcp_oauth import HermesTokenStorage

        storage = HermesTokenStorage("srv")
        asyncio.run(
            storage.set_tokens(
                OAuthToken(
                    access_token="a",
                    token_type="Bearer",
                    expires_in=3600,
                    refresh_token="r",
                )
            )
        )

        # Wait briefly so the remaining TTL is measurably less than 3600.
        time.sleep(0.05)

        reloaded = asyncio.run(storage.get_tokens())
        assert reloaded is not None
        assert reloaded.expires_in is not None
        # Should be slightly less than 3600 after the 50ms sleep.
        assert 3500 < reloaded.expires_in <= 3600

    def test_get_tokens_returns_zero_ttl_for_expired_token(
        self, tmp_path, monkeypatch
    ):
        """An already-expired token reloaded from disk must report expires_in=0."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import HermesTokenStorage, _get_token_dir

        token_dir = _get_token_dir()
        token_dir.mkdir(parents=True, exist_ok=True)
        # Write an already-expired token file directly.
        (token_dir / "srv.json").write_text(
            json.dumps(
                {
                    "access_token": "a",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "expires_at": time.time() - 60,  # expired 1 min ago
                    "refresh_token": "r",
                }
            )
        )

        storage = HermesTokenStorage("srv")
        reloaded = asyncio.run(storage.get_tokens())
        assert reloaded is not None
        assert reloaded.expires_in == 0, (
            "Expired token must reload with expires_in=0 so the SDK's "
            "is_token_valid() returns False and preemptive refresh fires."
        )

    def test_get_tokens_legacy_file_without_expires_at_is_loadable(
        self, tmp_path, monkeypatch
    ):
        """Existing on-disk files (pre-Fix-A) must still load without crashing.

        Pre-existing token files have ``expires_in`` but no ``expires_at``.
        Fix A falls back to the file's mtime as a best-effort wall-clock
        proxy: a file whose (mtime + expires_in) is in the past clamps
        expires_in to zero so the SDK refreshes on next request. A fresh
        legacy-format file (mtime = now) keeps most of its TTL.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import HermesTokenStorage, _get_token_dir

        token_dir = _get_token_dir()
        token_dir.mkdir(parents=True, exist_ok=True)
        # Legacy-shape file (no expires_at). Make it stale by backdating mtime
        # well past its nominal expires_in.
        legacy_path = token_dir / "srv.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "access_token": "a",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": "r",
                }
            )
        )
        stale_time = time.time() - 7200  # 2hr ago, exceeds 3600s TTL
        import os

        os.utime(legacy_path, (stale_time, stale_time))

        storage = HermesTokenStorage("srv")
        reloaded = asyncio.run(storage.get_tokens())
        assert reloaded is not None
        assert reloaded.expires_in == 0, (
            "Legacy file whose mtime + expires_in is in the past must report "
            "expires_in=0 so the SDK refreshes on next request."
        )


# ---------------------------------------------------------------------------
# HermesMCPOAuthProvider._initialize — seed token_expiry_time
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_seeds_token_expiry_time_from_stored_tokens(
    tmp_path, monkeypatch
):
    """Cold-load must populate context.token_expiry_time.

    The SDK's base ``_initialize`` loads current_tokens but doesn't seed
    token_expiry_time. Our subclass must do it so ``is_token_valid()``
    reports correctly and the preemptive-refresh path fires when needed.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="a",
            token_type="Bearer",
            expires_in=7200,
            refresh_token="r",
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )

    from mcp.shared.auth import OAuthClientMetadata

    metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        client_name="Hermes Agent",
    )
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    await provider._initialize()

    assert provider.context.token_expiry_time is not None, (
        "Fix A: _initialize must seed context.token_expiry_time so "
        "is_token_valid() correctly reports expiry on cold-load."
    )
    # Should be ~7200s in the future (fresh write).
    assert provider.context.token_expiry_time > time.time() + 7000
    assert provider.context.token_expiry_time <= time.time() + 7200 + 5


@pytest.mark.asyncio
async def test_initialize_flags_expired_token_as_invalid(tmp_path, monkeypatch):
    """After _initialize, an expired-on-disk token must report is_token_valid=False.

    This is the end-to-end assertion: cold-load an expired token, verify the
    SDK's own ``is_token_valid()`` now returns False (the consequence of
    seeding token_expiry_time correctly), so the SDK's ``async_auth_flow``
    will take the ``can_refresh_token()`` branch on the next request and
    silently refresh instead of sending the stale Bearer.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage, _get_token_dir
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None
    reset_manager_for_tests()

    # Write an already-expired token directly so we control the wall-clock.
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
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )

    metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        client_name="Hermes Agent",
    )
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    await provider._initialize()

    assert provider.context.is_token_valid() is False, (
        "After _initialize with an expired-on-disk token, is_token_valid() "
        "must return False so the SDK's async_auth_flow takes the "
        "preemptive refresh path."
    )
    assert provider.context.can_refresh_token() is True, (
        "Refresh should remain possible because refresh_token + client_info "
        "are both present."
    )


async def _noop_redirect(_url: str) -> None:
    return None


async def _noop_callback() -> tuple[str, str | None]:
    raise AssertionError("callback handler should not be invoked in these tests")


# ---------------------------------------------------------------------------
# Pre-flight OAuth metadata discovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_prefetches_oauth_metadata_when_missing(
    tmp_path, monkeypatch
):
    """Cold-load must pre-flight PRM + ASM discovery so ``_refresh_token``
    has the correct ``token_endpoint`` before the first refresh attempt.

    Without this, the SDK's ``_refresh_token`` falls back to
    ``{server_url}/token`` which is wrong for providers whose AS is at
    a different origin. BetterStack specifically: MCP at
    ``mcp.betterstack.com`` but token_endpoint at
    ``betterstack.com/oauth/token``. Without pre-flight the refresh 404s
    and we drop into full browser re-auth — visible to the user as an
    unwanted OAuth browser prompt every time the process restarts.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import httpx
    from mcp.shared.auth import (
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthToken,
    )
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="a",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="r",
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )

    # Route the AsyncClient used inside _prefetch_oauth_metadata through a
    # MockTransport that mimics BetterStack's split-origin discovery:
    #   PRM at mcp.example.com/.well-known/oauth-protected-resource -> points to auth.example.com
    #   ASM at auth.example.com/.well-known/oauth-authorization-server -> token_endpoint at auth.example.com/oauth/token
    def mock_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/.well-known/oauth-protected-resource"):
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example.com",
                    "authorization_servers": ["https://auth.example.com"],
                    "scopes_supported": ["read", "write"],
                    "bearer_methods_supported": ["header"],
                },
            )
        if url.endswith("/.well-known/oauth-authorization-server"):
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example.com",
                    "authorization_endpoint": "https://auth.example.com/oauth/authorize",
                    "token_endpoint": "https://auth.example.com/oauth/token",
                    "registration_endpoint": "https://auth.example.com/oauth/register",
                    "response_types_supported": ["code"],
                    "grant_types_supported": ["authorization_code", "refresh_token"],
                    "code_challenge_methods_supported": ["S256"],
                    "token_endpoint_auth_methods_supported": ["none"],
                    "scopes_supported": ["read", "write"],
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)

    # Patch the AsyncClient constructor used by _prefetch_oauth_metadata so
    # it uses our mock transport instead of the real network.
    import httpx as real_httpx

    original_async_client = real_httpx.AsyncClient

    def patched_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(real_httpx, "AsyncClient", patched_async_client)

    metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        client_name="Hermes Agent",
    )
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://mcp.example.com",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    await provider._initialize()

    assert provider.context.protected_resource_metadata is not None, (
        "Pre-flight must cache PRM for the SDK to reference later."
    )
    assert provider.context.oauth_metadata is not None, (
        "Pre-flight must cache ASM so _refresh_token builds the correct "
        "token_endpoint URL."
    )
    assert str(provider.context.oauth_metadata.token_endpoint) == (
        "https://auth.example.com/oauth/token"
    )


@pytest.mark.asyncio
async def test_initialize_skips_prefetch_when_no_tokens(tmp_path, monkeypatch):
    """Pre-flight must not run when there are no stored tokens yet.

    Without this guard, every fresh-install ``_initialize`` would do two
    extra network roundtrips that gain nothing (the SDK's 401-branch
    discovery will run on the first real request anyway).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import httpx
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests
    from tools.mcp_oauth import HermesTokenStorage

    assert _HERMES_PROVIDER_CLS is not None
    reset_manager_for_tests()

    calls: list[str] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    import httpx as real_httpx

    original = real_httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(real_httpx, "AsyncClient", patched)

    storage = HermesTokenStorage("srv")  # empty — no tokens on disk
    metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        client_name="Hermes Agent",
    )
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://mcp.example.com",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    await provider._initialize()

    assert calls == [], (
        f"Pre-flight must not fire when no tokens are stored, but got {calls}"
    )
