"""Tests for the MCP OAuth manager (tools/mcp_oauth_manager.py).

The manager consolidates the eight scattered MCP-OAuth call sites into a
single object with disk-mtime watch, dedup'd 401 handling, and a provider
cache. See `tools/mcp_oauth_manager.py` for design rationale.
"""
import json
import os
import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK 1.26.0+ required for OAuth support",
)


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


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
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    p2 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert p1 is p2


def test_manager_get_or_build_rebuilds_on_url_change(tmp_path, monkeypatch):
    """Changing the URL discards the cached provider."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://a.example.com/mcp", None)
    p2 = mgr.get_or_build_provider("srv", "https://b.example.com/mcp", None)
    assert p1 is not p2


def test_manager_remove_evicts_cache(tmp_path, monkeypatch):
    """remove(name) evicts the provider from cache AND deletes disk files."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
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
    _set_interactive_stdin(monkeypatch)

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    assert _HERMES_PROVIDER_CLS is not None
    assert isinstance(provider, _HERMES_PROVIDER_CLS)
    assert provider._hermes_server_name == "srv"


def test_manager_fails_fast_noninteractive_without_cached_tokens(tmp_path, monkeypatch):
    """A daemon without cached MCP OAuth tokens must not enter browser auth."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch, is_tty=False)
    from tools.mcp_oauth import OAuthNonInteractiveError
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()

    with pytest.raises(OAuthNonInteractiveError, match="non-interactive"):
        mgr.get_or_build_provider("linear", "https://mcp.linear.app/mcp", None)

    assert mgr._entries["linear"].provider is None



@pytest.mark.asyncio
async def test_bearer_header_materializes_cached_valid_token(tmp_path, monkeypatch):
    """OAuth Streamable-HTTP data-plane can use a plain bearer header.

    The long-lived GET/SSE stream must not be routed through the SDK
    OAuthClientProvider auth flow, because that provider holds an auth lock
    until the response completes and can block the following tools/list POST.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch, is_tty=False)

    from mcp.shared.auth import OAuthToken

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import MCPOAuthManager

    await HermesTokenStorage("srv").set_tokens(
        OAuthToken(
            access_token="ACCESS",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="REFRESH",
        )
    )

    mgr = MCPOAuthManager()
    header = await mgr.get_bearer_authorization_header(
        "srv", "https://example.com/mcp", None,
    )

    assert header == "Bearer ACCESS"


@pytest.mark.asyncio
async def test_bearer_header_refreshes_expired_token(tmp_path, monkeypatch):
    """Expired cached tokens refresh before a data-plane bearer is returned."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch, is_tty=False)

    from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata, OAuthToken

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import MCPOAuthManager

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="OLD",
            token_type="Bearer",
            expires_in=0,
            refresh_token="REFRESH",
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="client-id",
            client_secret="client-secret",
            redirect_uris=["http://127.0.0.1/callback"],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_post",
        )
    )
    storage.save_oauth_metadata(
        OAuthMetadata(
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
        )
    )

    class _Resp:
        status_code = 200

        async def aread(self):
            return b'{"access_token":"NEW","token_type":"Bearer","expires_in":3600,"refresh_token":"REFRESH2"}'

    class _Client:
        def __init__(self, *args, **kwargs):
            self.request = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def send(self, request):
            self.request = request
            return _Resp()

    monkeypatch.setattr("httpx.AsyncClient", _Client)

    mgr = MCPOAuthManager()
    header = await mgr.get_bearer_authorization_header(
        "srv", "https://example.com/mcp", None,
    )

    assert header == "Bearer NEW"
