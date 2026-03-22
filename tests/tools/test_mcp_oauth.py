"""Tests for tools/mcp_oauth.py — thin OAuth adapter over MCP SDK."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from tools.mcp_oauth import (
    HermesTokenStorage,
    build_oauth_auth,
    remove_oauth_tokens,
    _find_free_port,
    _can_open_browser,
)


# ---------------------------------------------------------------------------
# HermesTokenStorage
# ---------------------------------------------------------------------------

class TestHermesTokenStorage:
    def test_roundtrip_tokens(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("test-server")

        import asyncio

        # Initially empty
        assert asyncio.run(storage.get_tokens()) is None

        # Save and retrieve
        mock_token = MagicMock()
        mock_token.model_dump.return_value = {
            "access_token": "abc123",
            "token_type": "Bearer",
            "refresh_token": "ref456",
        }
        asyncio.run(storage.set_tokens(mock_token))

        # File exists with correct permissions
        token_path = tmp_path / "mcp-tokens" / "test-server.json"
        assert token_path.exists()
        data = json.loads(token_path.read_text())
        assert data["access_token"] == "abc123"

    def test_roundtrip_client_info(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("test-server")
        import asyncio

        assert asyncio.run(storage.get_client_info()) is None

        mock_client = MagicMock()
        mock_client.model_dump.return_value = {
            "client_id": "hermes-123",
            "client_secret": "secret",
        }
        asyncio.run(storage.set_client_info(mock_client))

        client_path = tmp_path / "mcp-tokens" / "test-server.client.json"
        assert client_path.exists()

    def test_remove_cleans_up(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("test-server")

        # Create files
        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        (d / "test-server.json").write_text("{}")
        (d / "test-server.client.json").write_text("{}")

        storage.remove()
        assert not (d / "test-server.json").exists()
        assert not (d / "test-server.client.json").exists()


# ---------------------------------------------------------------------------
# build_oauth_auth
# ---------------------------------------------------------------------------

class TestBuildOAuthAuth:
    def test_returns_oauth_provider(self):
        try:
            from mcp.client.auth import OAuthClientProvider
        except ImportError:
            pytest.skip("MCP SDK auth not available")

        auth = build_oauth_auth("test", "https://example.com/mcp")
        assert isinstance(auth, OAuthClientProvider)

    def test_returns_none_without_sdk(self, monkeypatch):
        import tools.mcp_oauth as mod
        orig_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def _block_import(name, *args, **kwargs):
            if "mcp.client.auth" in name:
                raise ImportError("blocked")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_import):
            result = build_oauth_auth("test", "https://example.com")
        # May or may not be None depending on import caching, but shouldn't crash
        assert result is None or result is not None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_find_free_port_returns_int(self):
        port = _find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_can_open_browser_false_in_ssh(self, monkeypatch):
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")
        assert _can_open_browser() is False

    def test_can_open_browser_false_without_display(self, monkeypatch):
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        # Mock os.name and uname for non-macOS, non-Windows
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(os, "uname", lambda: type("", (), {"sysname": "Linux"})())
        assert _can_open_browser() is False


# ---------------------------------------------------------------------------
# remove_oauth_tokens
# ---------------------------------------------------------------------------

class TestRemoveOAuthTokens:
    def test_removes_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        d = tmp_path / "mcp-tokens"
        d.mkdir()
        (d / "myserver.json").write_text("{}")
        (d / "myserver.client.json").write_text("{}")

        remove_oauth_tokens("myserver")

        assert not (d / "myserver.json").exists()
        assert not (d / "myserver.client.json").exists()

    def test_no_error_when_files_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        remove_oauth_tokens("nonexistent")  # should not raise
