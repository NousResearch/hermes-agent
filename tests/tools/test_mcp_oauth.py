"""Tests for tools/mcp_oauth.py — OAuth 2.1 PKCE support for MCP servers."""

import json
import os
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from tools.mcp_oauth import (
    HermesTokenStorage,
    OAuthNonInteractiveError,
    build_oauth_auth,
    remove_oauth_tokens,
    _find_free_port,
    _can_open_browser,
    _is_interactive,
    _wait_for_callback,
    _make_callback_handler,
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

    def test_has_cached_tokens(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("my-server")

        assert not storage.has_cached_tokens()

        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        (d / "my-server.json").write_text('{"access_token": "x", "token_type": "Bearer"}')

        assert storage.has_cached_tokens()

    def test_corrupt_tokens_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("bad-server")

        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        (d / "bad-server.json").write_text("NOT VALID JSON{{{")

        import asyncio
        assert asyncio.run(storage.get_tokens()) is None

    def test_corrupt_client_info_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("bad-server")

        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        (d / "bad-server.client.json").write_text("GARBAGE")

        import asyncio
        assert asyncio.run(storage.get_client_info()) is None


# ---------------------------------------------------------------------------
# build_oauth_auth
# ---------------------------------------------------------------------------

class TestBuildOAuthAuth:
    def test_returns_oauth_provider(self, tmp_path, monkeypatch):
        try:
            from mcp.client.auth import OAuthClientProvider
        except ImportError:
            pytest.skip("MCP SDK auth not available")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        auth = build_oauth_auth("test", "https://example.com/mcp")
        assert isinstance(auth, OAuthClientProvider)

    def test_returns_none_without_sdk(self, monkeypatch):
        import tools.mcp_oauth as mod
        monkeypatch.setattr(mod, "_OAUTH_AVAILABLE", False)
        result = build_oauth_auth("test", "https://example.com")
        assert result is None

    def test_pre_registered_client_id_stored(self, tmp_path, monkeypatch):
        try:
            from mcp.client.auth import OAuthClientProvider
        except ImportError:
            pytest.skip("MCP SDK auth not available")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        build_oauth_auth("slack", "https://slack.example.com/mcp", {
            "client_id": "my-app-id",
            "client_secret": "my-secret",
            "scope": "channels:read",
        })

        client_path = tmp_path / "mcp-tokens" / "slack.client.json"
        assert client_path.exists()
        data = json.loads(client_path.read_text())
        assert data["client_id"] == "my-app-id"
        assert data["client_secret"] == "my-secret"

    def test_scope_passed_through(self, tmp_path, monkeypatch):
        try:
            from mcp.client.auth import OAuthClientProvider
        except ImportError:
            pytest.skip("MCP SDK auth not available")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = build_oauth_auth("scoped", "https://example.com/mcp", {
            "scope": "read write admin",
        })
        assert provider is not None
        assert provider.context.client_metadata.scope == "read write admin"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_find_free_port_returns_int(self):
        port = _find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_find_free_port_unique(self):
        """Two consecutive calls should return different ports (usually)."""
        ports = {_find_free_port() for _ in range(5)}
        # At least 2 different ports out of 5 attempts
        assert len(ports) >= 2

    def test_can_open_browser_false_in_ssh(self, monkeypatch):
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")
        assert _can_open_browser() is False

    def test_can_open_browser_false_without_display(self, monkeypatch):
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        # Mock os.name and uname for non-macOS, non-Windows
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(os, "uname", lambda: type("", (), {"sysname": "Linux"})())
        assert _can_open_browser() is False

    def test_can_open_browser_true_with_display(self, monkeypatch):
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr(os, "name", "posix")
        assert _can_open_browser() is True


# ---------------------------------------------------------------------------
# Path traversal protection
# ---------------------------------------------------------------------------

class TestPathTraversal:
    """Verify server_name is sanitized to prevent path traversal."""

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("../../.ssh/config")
        path = storage._tokens_path()
        # Should stay within mcp-tokens directory
        assert "mcp-tokens" in str(path)
        assert ".ssh" not in str(path.resolve())

    def test_dots_and_slashes_sanitized(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("../../../etc/passwd")
        path = storage._tokens_path()
        resolved = path.resolve()
        assert resolved.is_relative_to((tmp_path / "mcp-tokens").resolve())

    def test_normal_name_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("my-mcp-server")
        assert "my-mcp-server.json" in str(storage._tokens_path())

    def test_special_chars_sanitized(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("server@host:8080/path")
        path = storage._tokens_path()
        assert "@" not in path.name
        assert ":" not in path.name
        assert "/" not in path.stem


# ---------------------------------------------------------------------------
# Callback handler isolation
# ---------------------------------------------------------------------------

class TestCallbackHandlerIsolation:
    """Verify concurrent OAuth flows don't share state."""

    def test_independent_result_dicts(self):
        _, result_a = _make_callback_handler()
        _, result_b = _make_callback_handler()

        result_a["auth_code"] = "code_A"
        result_b["auth_code"] = "code_B"

        assert result_a["auth_code"] == "code_A"
        assert result_b["auth_code"] == "code_B"

    def test_handler_writes_to_own_result(self):
        HandlerClass, result = _make_callback_handler()
        assert result["auth_code"] is None

        # Simulate a GET request
        handler = HandlerClass.__new__(HandlerClass)
        handler.path = "/callback?code=test123&state=mystate"
        handler.wfile = BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.do_GET()

        assert result["auth_code"] == "test123"
        assert result["state"] == "mystate"

    def test_handler_captures_error(self):
        HandlerClass, result = _make_callback_handler()

        handler = HandlerClass.__new__(HandlerClass)
        handler.path = "/callback?error=access_denied"
        handler.wfile = BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.do_GET()

        assert result["auth_code"] is None
        assert result["error"] == "access_denied"


# ---------------------------------------------------------------------------
# Port sharing
# ---------------------------------------------------------------------------

class TestOAuthPortSharing:
    """Verify build_oauth_auth and _wait_for_callback use the same port."""

    def test_port_stored_globally(self, tmp_path, monkeypatch):
        import tools.mcp_oauth as mod
        mod._oauth_port = None

        try:
            from mcp.client.auth import OAuthClientProvider
        except ImportError:
            pytest.skip("MCP SDK auth not available")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        build_oauth_auth("test-port", "https://example.com/mcp")
        assert mod._oauth_port is not None
        assert isinstance(mod._oauth_port, int)
        assert 1024 <= mod._oauth_port <= 65535


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


# ---------------------------------------------------------------------------
# Non-interactive / startup-safety tests
# ---------------------------------------------------------------------------

class TestIsInteractive:
    """_is_interactive() detects headless/daemon/container environments."""

    def test_false_when_stdin_not_tty(self, monkeypatch):
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)
        assert _is_interactive() is False

    def test_true_when_stdin_is_tty(self, monkeypatch):
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)
        assert _is_interactive() is True

    def test_false_when_stdin_has_no_isatty(self, monkeypatch):
        """Some environments replace stdin with an object without isatty()."""
        mock_stdin = object()  # no isatty attribute
        monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)
        assert _is_interactive() is False


class TestWaitForCallbackNoBlocking:
    """_wait_for_callback() must never call input() — it raises instead."""

    def test_raises_on_timeout_instead_of_input(self):
        """When no auth code arrives, raises OAuthNonInteractiveError."""
        import tools.mcp_oauth as mod
        import asyncio

        mod._oauth_port = _find_free_port()

        async def instant_sleep(_seconds):
            pass

        with patch.object(mod.asyncio, "sleep", instant_sleep):
            with patch("builtins.input", side_effect=AssertionError("input() must not be called")):
                with pytest.raises(OAuthNonInteractiveError, match="callback timed out"):
                    asyncio.run(_wait_for_callback())


class TestBuildOAuthAuthNonInteractive:
    """build_oauth_auth() in non-interactive mode."""

    def test_noninteractive_without_cached_tokens_warns(self, tmp_path, monkeypatch, caplog):
        """Without cached tokens, non-interactive mode logs a clear warning."""
        try:
            from mcp.client.auth import OAuthClientProvider
        except ImportError:
            pytest.skip("MCP SDK auth not available")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)

        import logging
        with caplog.at_level(logging.WARNING, logger="tools.mcp_oauth"):
            auth = build_oauth_auth("atlassian", "https://mcp.atlassian.com/v1/mcp")

        assert auth is not None
        assert "no cached tokens found" in caplog.text.lower()
        assert "non-interactive" in caplog.text.lower()

    def test_noninteractive_with_cached_tokens_no_warning(self, tmp_path, monkeypatch, caplog):
        """With cached tokens, non-interactive mode logs no 'no cached tokens' warning."""
        try:
            from mcp.client.auth import OAuthClientProvider
        except ImportError:
            pytest.skip("MCP SDK auth not available")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)

        # Pre-populate cached tokens
        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        (d / "atlassian.json").write_text(json.dumps({
            "access_token": "cached",
            "token_type": "Bearer",
        }))

        import logging
        with caplog.at_level(logging.WARNING, logger="tools.mcp_oauth"):
            auth = build_oauth_auth("atlassian", "https://mcp.atlassian.com/v1/mcp")

        assert auth is not None
        assert "no cached tokens found" not in caplog.text.lower()


# ---------------------------------------------------------------------------
# Extracted helper tests (Task 3 of MCP OAuth consolidation)
# ---------------------------------------------------------------------------


def test_build_client_metadata_basic():
    """_build_client_metadata returns metadata with expected defaults."""
    from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

    cfg = {"client_name": "Test Client"}
    _configure_callback_port(cfg)
    md = _build_client_metadata(cfg)

    assert md.client_name == "Test Client"
    assert "authorization_code" in md.grant_types
    assert "refresh_token" in md.grant_types


def test_build_client_metadata_without_secret_is_public():
    """Without client_secret, token endpoint auth is 'none' (public client)."""
    from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

    cfg = {}
    _configure_callback_port(cfg)
    md = _build_client_metadata(cfg)
    assert md.token_endpoint_auth_method == "none"


def test_build_client_metadata_with_secret_is_confidential():
    """With client_secret, token endpoint auth is 'client_secret_post'."""
    from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

    cfg = {"client_secret": "shh"}
    _configure_callback_port(cfg)
    md = _build_client_metadata(cfg)
    assert md.token_endpoint_auth_method == "client_secret_post"


def test_configure_callback_port_picks_free_port():
    """_configure_callback_port(0) picks a free port in the ephemeral range."""
    from tools.mcp_oauth import _configure_callback_port

    cfg = {"redirect_port": 0}
    port = _configure_callback_port(cfg)
    assert 1024 < port < 65536
    assert cfg["_resolved_port"] == port


def test_configure_callback_port_uses_explicit_port():
    """An explicit redirect_port is preserved."""
    from tools.mcp_oauth import _configure_callback_port

    cfg = {"redirect_port": 54321}
    port = _configure_callback_port(cfg)
    assert port == 54321
    assert cfg["_resolved_port"] == 54321


def test_parse_base_url_strips_path():
    """_parse_base_url drops path components for OAuth discovery."""
    from tools.mcp_oauth import _parse_base_url

    assert _parse_base_url("https://example.com/mcp/v1") == "https://example.com"
    assert _parse_base_url("https://example.com") == "https://example.com"
    assert _parse_base_url("https://host.example.com:8080/api") == "https://host.example.com:8080"

