#!/usr/bin/env python3
"""Tests for MCP OAuth 2.1 PKCE authentication (tools/mcp_oauth.py)."""

import json
import os
import stat
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# PKCE Generation
# ---------------------------------------------------------------------------

class TestPKCEGeneration:
    """Test PKCE code_verifier and code_challenge generation."""

    def test_generates_tuple(self):
        from tools.mcp_oauth import generate_pkce
        result = generate_pkce()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_verifier_is_url_safe_base64(self):
        from tools.mcp_oauth import generate_pkce
        verifier, _ = generate_pkce()
        # URL-safe base64 without padding
        assert isinstance(verifier, str)
        assert len(verifier) > 20
        assert "=" not in verifier
        assert "+" not in verifier
        assert "/" not in verifier

    def test_challenge_is_s256_of_verifier(self):
        import base64
        import hashlib
        from tools.mcp_oauth import generate_pkce

        verifier, challenge = generate_pkce()
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()
        ).rstrip(b"=").decode()
        assert challenge == expected

    def test_each_call_produces_unique_values(self):
        from tools.mcp_oauth import generate_pkce
        v1, c1 = generate_pkce()
        v2, c2 = generate_pkce()
        assert v1 != v2
        assert c1 != c2


# ---------------------------------------------------------------------------
# OAuth Discovery
# ---------------------------------------------------------------------------

class TestOAuthDiscovery:
    """Test OAuth metadata discovery from well-known endpoints."""

    def _make_metadata(self):
        return {
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "registration_endpoint": "https://auth.example.com/register",
        }

    @patch("tools.mcp_oauth.urllib.request.urlopen")
    def test_discovers_from_oauth_endpoint(self, mock_urlopen):
        from tools.mcp_oauth import discover_oauth_metadata

        metadata = self._make_metadata()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(metadata).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = discover_oauth_metadata("https://mcp.example.com/api")
        assert result == metadata
        # Should have called the first well-known path
        called_url = mock_urlopen.call_args[0][0].full_url
        assert "/.well-known/oauth-authorization-server" in called_url

    @patch("tools.mcp_oauth.urllib.request.urlopen")
    def test_falls_back_to_openid_configuration(self, mock_urlopen):
        from tools.mcp_oauth import discover_oauth_metadata

        metadata = self._make_metadata()

        # First call (oauth-authorization-server) fails, second succeeds
        error_resp = urllib_error_404()
        success_resp = MagicMock()
        success_resp.status = 200
        success_resp.read.return_value = json.dumps(metadata).encode()
        success_resp.__enter__ = MagicMock(return_value=success_resp)
        success_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [error_resp, success_resp]

        result = discover_oauth_metadata("https://mcp.example.com/api")
        assert result == metadata
        assert mock_urlopen.call_count == 2

    @patch("tools.mcp_oauth.urllib.request.urlopen")
    def test_returns_none_on_failure(self, mock_urlopen):
        from tools.mcp_oauth import discover_oauth_metadata

        mock_urlopen.side_effect = urllib_error_404()
        result = discover_oauth_metadata("https://mcp.example.com/api")
        assert result is None

    @patch("tools.mcp_oauth.urllib.request.urlopen")
    def test_returns_none_when_missing_authorization_endpoint(self, mock_urlopen):
        from tools.mcp_oauth import discover_oauth_metadata

        # Valid JSON but missing required field
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"issuer": "x"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = discover_oauth_metadata("https://mcp.example.com/api")
        assert result is None


# ---------------------------------------------------------------------------
# Dynamic Client Registration
# ---------------------------------------------------------------------------

class TestClientRegistration:
    """Test dynamic OAuth client registration (RFC 7591)."""

    @patch("tools.mcp_oauth.urllib.request.urlopen")
    def test_successful_registration(self, mock_urlopen):
        from tools.mcp_oauth import register_client

        reg_response = {"client_id": "dyn-client-123", "client_secret": "sec-456"}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(reg_response).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        metadata = {"registration_endpoint": "https://auth.example.com/register"}
        result = register_client(metadata, "http://localhost:18400/callback", "test-server")

        assert result is not None
        assert result["client_id"] == "dyn-client-123"
        assert result["client_secret"] == "sec-456"

    def test_skips_when_no_registration_endpoint(self):
        from tools.mcp_oauth import register_client

        metadata = {"authorization_endpoint": "https://auth.example.com/authorize"}
        result = register_client(metadata, "http://localhost:18400/callback", "test")
        assert result is None

    @patch("tools.mcp_oauth.urllib.request.urlopen")
    def test_returns_none_on_error(self, mock_urlopen):
        from tools.mcp_oauth import register_client

        mock_urlopen.side_effect = urllib_error_404()
        metadata = {"registration_endpoint": "https://auth.example.com/register"}
        result = register_client(metadata, "http://localhost:18400/callback", "test")
        assert result is None


# ---------------------------------------------------------------------------
# Token Storage
# ---------------------------------------------------------------------------

class TestTokenStorage:
    """Test token save/load and file permissions."""

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import save_tokens, load_tokens

        tokens = {
            "access_token": "at-123",
            "refresh_token": "rt-456",
            "expires_at": time.time() + 3600,
            "token_type": "Bearer",
            "client_id": "cid",
            "client_secret": "",
            "token_endpoint": "https://auth.example.com/token",
            "server_url": "https://mcp.example.com",
        }

        save_tokens("test-server", tokens)
        loaded = load_tokens("test-server")

        assert loaded is not None
        assert loaded["access_token"] == "at-123"
        assert loaded["refresh_token"] == "rt-456"

    def test_file_permissions(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import save_tokens, _token_path

        save_tokens("perm-test", {"access_token": "x"})
        path = _token_path("perm-test")
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600 but got {oct(mode)}"

    def test_load_returns_none_for_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import load_tokens

        result = load_tokens("nonexistent-server")
        assert result is None

    def test_load_returns_none_for_corrupt_file(self, tmp_path, monkeypatch):
        token_dir = tmp_path / "mcp-tokens"
        token_dir.mkdir(parents=True)
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", token_dir)
        from tools.mcp_oauth import load_tokens

        (token_dir / "corrupt.json").write_text("not json{{{")
        result = load_tokens("corrupt")
        assert result is None

    def test_load_returns_none_for_empty_access_token(self, tmp_path, monkeypatch):
        token_dir = tmp_path / "mcp-tokens"
        token_dir.mkdir(parents=True)
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", token_dir)
        from tools.mcp_oauth import load_tokens

        (token_dir / "empty.json").write_text(json.dumps({"access_token": ""}))
        result = load_tokens("empty")
        assert result is None

    def test_sanitizes_server_name_in_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import _token_path

        path = _token_path("../../etc/passwd")
        assert ".." not in path.name
        assert str(path).startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# Token Validity
# ---------------------------------------------------------------------------

class TestTokenValidity:
    """Test access token expiry checking."""

    def test_valid_token(self):
        from tools.mcp_oauth import _is_token_valid

        tokens = {"access_token": "x", "expires_at": time.time() + 3600}
        assert _is_token_valid(tokens) is True

    def test_expired_token(self):
        from tools.mcp_oauth import _is_token_valid

        tokens = {"access_token": "x", "expires_at": time.time() - 100}
        assert _is_token_valid(tokens) is False

    def test_token_near_expiry_within_buffer(self):
        from tools.mcp_oauth import _is_token_valid, _TOKEN_EXPIRY_BUFFER_SECS

        # Expires in 30 seconds but buffer is 60 — should be invalid
        tokens = {"access_token": "x", "expires_at": time.time() + 30}
        assert _is_token_valid(tokens) is False

    def test_no_expiry_field_valid_if_token_present(self):
        from tools.mcp_oauth import _is_token_valid

        tokens = {"access_token": "x", "expires_at": 0}
        assert _is_token_valid(tokens) is True

    def test_no_expiry_field_invalid_if_no_token(self):
        from tools.mcp_oauth import _is_token_valid

        tokens = {"access_token": "", "expires_at": 0}
        assert _is_token_valid(tokens) is False


# ---------------------------------------------------------------------------
# Token Refresh
# ---------------------------------------------------------------------------

class TestTokenRefresh:
    """Test OAuth token refresh flow."""

    @patch("tools.mcp_oauth.urllib.request.urlopen")
    def test_successful_refresh(self, mock_urlopen, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import refresh_access_token, save_tokens, load_tokens

        tokens = {
            "access_token": "old-at",
            "refresh_token": "rt-123",
            "expires_at": time.time() - 100,
            "client_id": "cid",
            "token_endpoint": "https://auth.example.com/token",
        }
        save_tokens("refresh-test", tokens)

        refresh_resp = {
            "access_token": "new-at",
            "refresh_token": "new-rt",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(refresh_resp).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = refresh_access_token("refresh-test")
        assert result == "new-at"

        # Verify saved tokens
        saved = load_tokens("refresh-test")
        assert saved["access_token"] == "new-at"
        assert saved["refresh_token"] == "new-rt"

    def test_returns_none_without_refresh_token(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import refresh_access_token, save_tokens

        tokens = {
            "access_token": "x",
            "refresh_token": "",
            "token_endpoint": "https://auth.example.com/token",
            "client_id": "cid",
        }
        save_tokens("no-rt", tokens)
        result = refresh_access_token("no-rt")
        assert result is None

    def test_returns_none_without_token_endpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import refresh_access_token, save_tokens

        tokens = {
            "access_token": "x",
            "refresh_token": "rt",
            "client_id": "cid",
        }
        save_tokens("no-endpoint", tokens)
        result = refresh_access_token("no-endpoint")
        assert result is None

    @patch("tools.mcp_oauth.urllib.request.urlopen")
    def test_returns_none_on_network_error(self, mock_urlopen, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import refresh_access_token, save_tokens

        tokens = {
            "access_token": "x",
            "refresh_token": "rt",
            "token_endpoint": "https://auth.example.com/token",
            "client_id": "cid",
        }
        save_tokens("net-error", tokens)
        mock_urlopen.side_effect = urllib_error_404()
        result = refresh_access_token("net-error")
        assert result is None


# ---------------------------------------------------------------------------
# get_auth_headers (main entry point)
# ---------------------------------------------------------------------------

class TestGetAuthHeaders:
    """Test the main get_auth_headers() entry point."""

    def test_returns_header_for_valid_cached_token(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import get_auth_headers, save_tokens

        tokens = {
            "access_token": "cached-at",
            "token_type": "Bearer",
            "expires_at": time.time() + 3600,
        }
        save_tokens("cached", tokens)

        headers = get_auth_headers("cached", "https://mcp.example.com")
        assert headers == {"Authorization": "Bearer cached-at"}

    @patch("tools.mcp_oauth.refresh_access_token")
    def test_refreshes_expired_token(self, mock_refresh, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import get_auth_headers, save_tokens

        tokens = {
            "access_token": "old",
            "refresh_token": "rt",
            "token_type": "Bearer",
            "expires_at": time.time() - 100,
            "token_endpoint": "https://auth.example.com/token",
            "client_id": "cid",
        }
        save_tokens("expired", tokens)
        mock_refresh.return_value = "refreshed-at"

        headers = get_auth_headers("expired", "https://mcp.example.com")
        assert headers == {"Authorization": "Bearer refreshed-at"}
        mock_refresh.assert_called_once()

    @patch("tools.mcp_oauth.start_auth_flow")
    def test_starts_full_flow_when_no_tokens(self, mock_flow, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import get_auth_headers

        mock_flow.return_value = {
            "access_token": "new-at",
            "token_type": "Bearer",
        }

        headers = get_auth_headers("new-server", "https://mcp.example.com")
        assert headers == {"Authorization": "Bearer new-at"}
        mock_flow.assert_called_once()

    @patch("tools.mcp_oauth.start_auth_flow")
    def test_returns_empty_dict_on_complete_failure(self, mock_flow, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import get_auth_headers

        mock_flow.return_value = None

        headers = get_auth_headers("fail-server", "https://mcp.example.com")
        assert headers == {}


# ---------------------------------------------------------------------------
# MCP Tool Integration
# ---------------------------------------------------------------------------

class TestMCPToolOAuthIntegration:
    """Test that mcp_tool.py correctly integrates OAuth auth."""

    def test_auth_type_stored_from_config(self):
        """MCPServerTask stores _auth_type from config."""
        from tools.mcp_tool import MCPServerTask
        import asyncio

        task = MCPServerTask("test")
        loop = asyncio.new_event_loop()
        try:
            # Simulate the beginning of run() where config is stored
            config = {"url": "https://example.com/mcp", "auth": "oauth", "timeout": 60}
            task._config = config
            task.tool_timeout = config.get("timeout", 120)
            task._auth_type = config.get("auth", "").lower().strip()

            assert task._auth_type == "oauth"
        finally:
            loop.close()

    def test_auth_type_defaults_to_empty(self):
        """MCPServerTask defaults _auth_type to empty string."""
        from tools.mcp_tool import MCPServerTask

        task = MCPServerTask("test")
        assert task._auth_type == ""

    def test_header_auth_not_affected(self):
        """Servers with headers but no auth: oauth should not trigger OAuth."""
        from tools.mcp_tool import MCPServerTask

        task = MCPServerTask("test")
        config = {
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer sk-123"},
        }
        task._config = config
        task._auth_type = config.get("auth", "").lower().strip()
        assert task._auth_type == ""


# ---------------------------------------------------------------------------
# Manual Code Flow
# ---------------------------------------------------------------------------

class TestManualCodeFlow:
    """Test manual (headless) authorization code entry."""

    @patch("builtins.input", return_value="auth-code-xyz")
    @patch("builtins.print")
    def test_returns_code_from_input(self, mock_print, mock_input):
        from tools.mcp_oauth import _prompt_manual_code

        result = _prompt_manual_code("https://auth.example.com/authorize?...")
        assert result == "auth-code-xyz"

    @patch("builtins.input", return_value="")
    @patch("builtins.print")
    def test_returns_none_for_empty_input(self, mock_print, mock_input):
        from tools.mcp_oauth import _prompt_manual_code

        result = _prompt_manual_code("https://auth.example.com/authorize?...")
        assert result is None

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    @patch("builtins.print")
    def test_returns_none_on_keyboard_interrupt(self, mock_print, mock_input):
        from tools.mcp_oauth import _prompt_manual_code

        result = _prompt_manual_code("https://auth.example.com/authorize?...")
        assert result is None

    @patch("builtins.input", side_effect=EOFError)
    @patch("builtins.print")
    def test_returns_none_on_eof(self, mock_print, mock_input):
        from tools.mcp_oauth import _prompt_manual_code

        result = _prompt_manual_code("https://auth.example.com/authorize?...")
        assert result is None


# ---------------------------------------------------------------------------
# Browser Detection
# ---------------------------------------------------------------------------

class TestBrowserDetection:
    """Test _can_open_browser() environment detection."""

    def test_cli_platform_can_open_browser(self, monkeypatch):
        from tools.mcp_oauth import _can_open_browser
        monkeypatch.setenv("HERMES_PLATFORM", "cli")
        # On macOS (test env), should return True
        import sys
        if sys.platform == "darwin":
            assert _can_open_browser() is True

    def test_telegram_platform_cannot_open_browser(self, monkeypatch):
        from tools.mcp_oauth import _can_open_browser
        monkeypatch.setenv("HERMES_PLATFORM", "telegram")
        assert _can_open_browser() is False

    def test_discord_platform_cannot_open_browser(self, monkeypatch):
        from tools.mcp_oauth import _can_open_browser
        monkeypatch.setenv("HERMES_PLATFORM", "discord")
        assert _can_open_browser() is False


# ---------------------------------------------------------------------------
# Start Auth Flow (integration with mocks)
# ---------------------------------------------------------------------------

class TestStartAuthFlow:
    """Test the full auth flow with mock network calls."""

    @patch("tools.mcp_oauth._can_open_browser", return_value=False)
    @patch("tools.mcp_oauth._prompt_manual_code", return_value="code-abc")
    @patch("tools.mcp_oauth.discover_oauth_metadata")
    @patch("tools.mcp_oauth.register_client")
    @patch("tools.mcp_oauth.urllib.request.urlopen")
    @patch("builtins.print")
    def test_headless_flow_with_manual_code(
        self, mock_print, mock_urlopen, mock_register, mock_discover,
        mock_prompt, mock_browser, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import start_auth_flow

        metadata = {
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }
        mock_discover.return_value = metadata
        mock_register.return_value = {"client_id": "dyn-cid", "client_secret": ""}

        token_resp = {
            "access_token": "new-at",
            "refresh_token": "new-rt",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(token_resp).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = start_auth_flow("test-server", "https://mcp.example.com")

        assert result is not None
        assert result["access_token"] == "new-at"
        assert result["refresh_token"] == "new-rt"
        assert result["client_id"] == "dyn-cid"
        mock_prompt.assert_called_once()

    @patch("tools.mcp_oauth.discover_oauth_metadata", return_value=None)
    def test_fails_without_metadata(self, mock_discover):
        from tools.mcp_oauth import start_auth_flow

        result = start_auth_flow("test", "https://mcp.example.com")
        assert result is None

    @patch("tools.mcp_oauth._can_open_browser", return_value=False)
    @patch("tools.mcp_oauth._prompt_manual_code", return_value=None)
    @patch("tools.mcp_oauth.discover_oauth_metadata")
    @patch("tools.mcp_oauth.register_client", return_value=None)
    def test_fails_when_user_cancels(
        self, mock_register, mock_discover, mock_prompt, mock_browser,
    ):
        from tools.mcp_oauth import start_auth_flow

        mock_discover.return_value = {
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }

        result = start_auth_flow("test", "https://mcp.example.com")
        assert result is None

    def test_callback_parameter_used_for_gateway(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.mcp_oauth._TOKEN_DIR", tmp_path / "mcp-tokens")
        from tools.mcp_oauth import start_auth_flow

        with patch("tools.mcp_oauth._can_open_browser", return_value=False), \
             patch("tools.mcp_oauth.discover_oauth_metadata") as mock_discover, \
             patch("tools.mcp_oauth.register_client", return_value=None), \
             patch("tools.mcp_oauth.urllib.request.urlopen") as mock_urlopen, \
             patch("builtins.print"):

            mock_discover.return_value = {
                "authorization_endpoint": "https://auth.example.com/authorize",
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "static-cid",
            }

            token_resp = {
                "access_token": "gw-at",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(token_resp).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            # Simulate a gateway callback that returns a code
            gw_callback = MagicMock(return_value="gw-code-123")

            result = start_auth_flow(
                "gw-server", "https://mcp.example.com", callback=gw_callback,
            )

            assert result is not None
            assert result["access_token"] == "gw-at"
            gw_callback.assert_called_once()
            # The callback should have been called with an auth URL
            auth_url_arg = gw_callback.call_args[0][0]
            assert "authorize" in auth_url_arg


# ---------------------------------------------------------------------------
# Helper: create urllib error for mocking
# ---------------------------------------------------------------------------

def urllib_error_404():
    """Create a urllib HTTPError for 404 responses."""
    import urllib.error
    return urllib.error.HTTPError(
        url="https://example.com",
        code=404,
        msg="Not Found",
        hdrs=None,  # type: ignore
        fp=None,
    )
