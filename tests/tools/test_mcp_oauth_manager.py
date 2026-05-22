"""Tests for the MCP OAuth manager (tools/mcp_oauth_manager.py).

The manager consolidates the eight scattered MCP-OAuth call sites into a
single object with disk-mtime watch, dedup'd 401 handling, and a provider
cache. See `tools/mcp_oauth_manager.py` for design rationale.
"""
import json
import os
import time

import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK 1.26.0+ required for OAuth support",
)


def test_manager_is_singleton():
    """get_manager() returns the same instance across calls."""
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    reset_manager_for_tests()
    m1 = get_manager()
    m2 = get_manager()
    assert m1 is m2


def test_manager_get_or_build_provider_caches(tmp_path, monkeypatch):
    """Calling get_or_build_provider twice with same name returns same provider."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    p2 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert p1 is p2


def test_manager_get_or_build_rebuilds_on_url_change(tmp_path, monkeypatch):
    """Changing the URL discards the cached provider."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://a.example.com/mcp", None)
    p2 = mgr.get_or_build_provider("srv", "https://b.example.com/mcp", None)
    assert p1 is not p2


def test_manager_remove_evicts_cache(tmp_path, monkeypatch):
    """remove(name) evicts the provider from cache AND deletes disk files."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager

    # Pre-seed tokens on disk
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(json.dumps({
        "access_token": "TOK",
        "token_type": "Bearer",
    }))

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert p1 is not None
    assert (token_dir / "srv.json").exists()

    mgr.remove("srv")

    assert not (token_dir / "srv.json").exists()
    p2 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert p1 is not p2


def test_hermes_provider_subclass_exists():
    """HermesMCPOAuthProvider is defined and subclasses OAuthClientProvider."""
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS
    from mcp.client.auth.oauth2 import OAuthClientProvider

    assert _HERMES_PROVIDER_CLS is not None
    assert issubclass(_HERMES_PROVIDER_CLS, OAuthClientProvider)


@pytest.mark.asyncio
async def test_disk_watch_invalidates_on_mtime_change(tmp_path, monkeypatch):
    """When the tokens file mtime changes, provider._initialized flips False.

    This is the behaviour Claude Code ships as
    invalidateOAuthCacheIfDiskChanged (CC-1096 / GH#24317) and is the core
    fix for Cthulhu's external-cron refresh workflow.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    reset_manager_for_tests()

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    tokens_file = token_dir / "srv.json"
    tokens_file.write_text(json.dumps({
        "access_token": "OLD",
        "token_type": "Bearer",
    }))

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None

    # First call: records mtime (zero -> real) -> returns True
    changed1 = await mgr.invalidate_if_disk_changed("srv")
    assert changed1 is True

    # No file change -> False
    changed2 = await mgr.invalidate_if_disk_changed("srv")
    assert changed2 is False

    # Touch file with a newer mtime
    future_mtime = time.time() + 10
    os.utime(tokens_file, (future_mtime, future_mtime))

    changed3 = await mgr.invalidate_if_disk_changed("srv")
    assert changed3 is True
    # _initialized flipped — next async_auth_flow will re-read from disk
    assert provider._initialized is False


def test_manager_builds_hermes_provider_subclass(tmp_path, monkeypatch):
    """get_or_build_provider returns HermesMCPOAuthProvider, not plain OAuthClientProvider."""
    from tools.mcp_oauth_manager import (
        MCPOAuthManager, _HERMES_PROVIDER_CLS, reset_manager_for_tests,
    )
    reset_manager_for_tests()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    assert _HERMES_PROVIDER_CLS is not None
    assert isinstance(provider, _HERMES_PROVIDER_CLS)
    assert provider._hermes_server_name == "srv"



# ---------------------------------------------------------------------------
# Refresh-failure recovery
# ---------------------------------------------------------------------------
#
# When the authorization server rejects a refresh token (e.g. server-side
# rotation, revocation, expired DCR client), the MCP SDK's base
# ``_handle_refresh_response`` only nulls ``current_tokens`` in memory.
# Stale tokens + dynamic client_info on disk get reloaded on the next
# process start and the cycle repeats.
#
# HermesMCPOAuthProvider overrides ``_handle_refresh_response`` to wipe
# the on-disk state and null in-memory ``client_info`` so the SDK's 401
# branch on the same async_auth_flow re-registers cleanly and prompts a
# fresh browser login. These tests pin that behavior.


@pytest.mark.asyncio
async def test_refresh_failure_clears_dynamic_client_state(tmp_path, monkeypatch):
    """Refresh rejection -> wipe tokens file + dynamic client_info + in-memory client_info."""
    import httpx

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    tokens_file = token_dir / "srv.json"
    client_info_file = token_dir / "srv.client.json"

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    # Seed tokens + client_info AFTER build so the redirect-URI invalidation
    # pass in build_oauth_auth doesn't pre-delete our fixture. We're testing
    # the refresh-failure path, not the build-time invalidation path.
    tokens_file.write_text(json.dumps({
        "access_token": "stale-access",
        "token_type": "Bearer",
        "refresh_token": "stale-refresh",
        "expires_in": 3600,
    }))
    client_info_file.write_text(json.dumps({
        "client_id": "dyn-client-id",
        "redirect_uris": ["http://127.0.0.1:31337/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "hermes_dynamic_client": True,
    }))
    from mcp.shared.auth import OAuthClientInformationFull
    provider.context.client_info = OAuthClientInformationFull.model_validate(
        json.loads(client_info_file.read_text())
    )

    # Simulate the AS rejecting the refresh token (RFC 6749 §5.2).
    failed_response = httpx.Response(
        status_code=400,
        content=b'{"error": "invalid_grant"}',
        headers={"content-type": "application/json"},
    )

    ok = await provider._handle_refresh_response(failed_response)

    assert ok is False
    assert not tokens_file.exists(), \
        "stale tokens file must be deleted so process restarts don't reload it"
    assert not client_info_file.exists(), \
        "stale dynamic client_info must be deleted so SDK re-registers via DCR"
    assert provider.context.client_info is None, \
        "in-memory client_info must be nulled so the SDK's 401 branch takes the DCR path"


@pytest.mark.asyncio
async def test_refresh_failure_preserves_preregistered_client(tmp_path, monkeypatch):
    """Refresh rejection clears tokens but preserves a configured pre-registered client_id."""
    import httpx

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    tokens_file = token_dir / "srv.json"
    client_info_file = token_dir / "srv.client.json"

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider(
        "srv",
        "https://example.com/mcp",
        {"client_id": "configured-client-id", "client_secret": "configured-secret"},
    )

    # Seed after build (see comment in the dynamic-client test).
    tokens_file.write_text(json.dumps({
        "access_token": "stale-access",
        "token_type": "Bearer",
        "refresh_token": "stale-refresh",
        "expires_in": 3600,
    }))
    client_info_file.write_text(json.dumps({
        "client_id": "configured-client-id",
        "client_secret": "configured-secret",
        "redirect_uris": ["http://127.0.0.1:31337/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_basic",
        "hermes_preregistered_client": True,
    }))
    from mcp.shared.auth import OAuthClientInformationFull
    provider.context.client_info = OAuthClientInformationFull.model_validate(
        json.loads(client_info_file.read_text())
    )

    failed_response = httpx.Response(
        status_code=400,
        content=b'{"error": "invalid_grant"}',
        headers={"content-type": "application/json"},
    )

    ok = await provider._handle_refresh_response(failed_response)

    assert ok is False
    assert not tokens_file.exists(), "stale tokens still get cleared"
    assert client_info_file.exists(), \
        "pre-registered client_info must be preserved (configured in config.yaml)"
    # In-memory client_info is left intact for pre-registered clients so the
    # configured credentials are still used in the subsequent auth flow.
    assert provider.context.client_info is not None


@pytest.mark.asyncio
async def test_refresh_success_does_not_clear_state(tmp_path, monkeypatch):
    """A successful refresh (200 + valid token JSON) leaves disk state alone."""
    import httpx

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    tokens_file = token_dir / "srv.json"
    client_info_file = token_dir / "srv.client.json"

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    # Seed after build to avoid the redirect-URI invalidation pass.
    tokens_file.write_text(json.dumps({
        "access_token": "old-access",
        "token_type": "Bearer",
        "refresh_token": "still-valid-refresh",
        "expires_in": 3600,
    }))
    client_info_file.write_text(json.dumps({
        "client_id": "dyn-client-id",
        "redirect_uris": ["http://127.0.0.1:31337/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "hermes_dynamic_client": True,
    }))

    success_response = httpx.Response(
        status_code=200,
        content=json.dumps({
            "access_token": "new-access",
            "token_type": "Bearer",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        }).encode(),
        headers={"content-type": "application/json"},
    )

    ok = await provider._handle_refresh_response(success_response)

    assert ok is True
    assert client_info_file.exists(), \
        "successful refresh must NOT delete dynamic client_info"
    # tokens file gets overwritten by the SDK's set_tokens; it must still exist.
    assert tokens_file.exists()
