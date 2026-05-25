"""Tests for tools/mcp_oauth.py — OAuth 2.1 PKCE support for MCP servers."""

import json
import os
import stat
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

import asyncio

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
    _redirect_handler,
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

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode bits not enforced on Windows")
    def test_token_file_created_with_0o600(self, tmp_path, monkeypatch):
        """Tokens must land on disk at 0o600 with no umask-default exposure window.

        Regression for the TOCTOU race where ``write_text`` + post-write
        ``chmod`` briefly left credentials at the process umask (commonly
        0o644 = world-readable) before tightening to owner-only. Mirrors
        the fix shipped for ``agent/google_oauth.py`` in #19673.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("perm-test-server")

        import asyncio
        mock_token = MagicMock()
        mock_token.model_dump.return_value = {
            "access_token": "secret-abc",
            "token_type": "Bearer",
            "refresh_token": "secret-ref",
        }
        asyncio.run(storage.set_tokens(mock_token))

        token_path = tmp_path / "mcp-tokens" / "perm-test-server.json"
        assert token_path.exists()
        mode = stat.S_IMODE(token_path.stat().st_mode)
        assert mode == 0o600, f"token file mode {oct(mode)} != 0o600 — TOCTOU race regressed"

        parent_mode = stat.S_IMODE(token_path.parent.stat().st_mode)
        assert parent_mode == 0o700, (
            f"token parent dir mode {oct(parent_mode)} != 0o700 — siblings can traverse"
        )

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
        assert json.loads(client_path.read_text())["hermes_dynamic_client"] is True

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


class TestRedirectHandlerSshHint:
    """_redirect_handler must print an SSH tunnel hint on remote sessions."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_ssh_hint_shown_on_ssh_session(self, monkeypatch, capsys):
        import tools.mcp_oauth as mco
        monkeypatch.setattr(mco, "_oauth_port", 49200)
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.setattr(mco, "_can_open_browser", lambda: False)

        self._run(_redirect_handler("https://example.com/auth?foo=bar"))

        err = capsys.readouterr().err
        assert "49200" in err
        assert "ssh -N -L" in err
        assert "Remote session detected" in err

    def test_ssh_hint_shown_via_ssh_tty(self, monkeypatch, capsys):
        import tools.mcp_oauth as mco
        monkeypatch.setattr(mco, "_oauth_port", 49201)
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.setenv("SSH_TTY", "/dev/pts/1")
        monkeypatch.setattr(mco, "_can_open_browser", lambda: False)

        self._run(_redirect_handler("https://example.com/auth"))

        err = capsys.readouterr().err
        assert "49201" in err
        assert "ssh -N -L" in err

    def test_no_ssh_hint_on_local_session(self, monkeypatch, capsys):
        import tools.mcp_oauth as mco
        monkeypatch.setattr(mco, "_oauth_port", 49202)
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.setattr(mco, "_can_open_browser", lambda: True)
        monkeypatch.setattr("webbrowser.open", lambda url, **kw: True)

        self._run(_redirect_handler("https://example.com/auth"))

        err = capsys.readouterr().err
        assert "ssh -N -L" not in err

    def test_no_ssh_hint_when_port_not_set(self, monkeypatch, capsys):
        import tools.mcp_oauth as mco
        monkeypatch.setattr(mco, "_oauth_port", None)
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")
        monkeypatch.setattr(mco, "_can_open_browser", lambda: False)

        self._run(_redirect_handler("https://example.com/auth"))

        err = capsys.readouterr().err
        assert "ssh -N -L" not in err


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

    def test_log_message_redacts_callback_code_and_state(self, caplog):
        HandlerClass, _ = _make_callback_handler()
        handler = object.__new__(HandlerClass)
        caplog.set_level("DEBUG", logger="tools.mcp_oauth")

        handler.log_message(
            '"%s" %s %s',
            "GET /callback?code=secret-code&state=secret-state&scope=read HTTP/1.1",
            "200",
            "-",
        )

        assert "code=[REDACTED]" in caplog.text
        assert "state=[REDACTED]" in caplog.text
        assert "scope=read" in caplog.text
        assert "secret-code" not in caplog.text
        assert "secret-state" not in caplog.text


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
    pytest.importorskip("mcp")
    from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

    cfg = {"client_name": "Test Client"}
    _configure_callback_port(cfg)
    md = _build_client_metadata(cfg)

    assert md.client_name == "Test Client"
    assert "authorization_code" in md.grant_types
    assert "refresh_token" in md.grant_types


def test_build_client_metadata_without_secret_is_public():
    """Without client_secret, token endpoint auth is 'none' (public client)."""
    pytest.importorskip("mcp")
    from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

    cfg = {}
    _configure_callback_port(cfg)
    md = _build_client_metadata(cfg)
    assert md.token_endpoint_auth_method == "none"


def test_build_client_metadata_with_secret_is_confidential():
    """With client_secret, token endpoint auth is 'client_secret_post'."""
    pytest.importorskip("mcp")
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


# ---------------------------------------------------------------------------
# redirect_host / redirect_port plumbing
# ---------------------------------------------------------------------------


class TestRedirectHostPlumbing:
    """``oauth.redirect_host`` and ``oauth.redirect_port`` plumb through to
    the redirect_uri, the pre-registered client, and the listener bind."""

    def test_default_redirect_host_is_loopback(self):
        """Without redirect_host configured, the URL stays on 127.0.0.1."""
        pytest.importorskip("mcp")
        import tools.mcp_oauth as mod
        from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

        cfg: dict = {}
        _configure_callback_port(cfg)
        md = _build_client_metadata(cfg)

        port = cfg["_resolved_port"]
        assert str(md.redirect_uris[0]).startswith(f"http://127.0.0.1:{port}/")
        assert mod._oauth_bind_host == "127.0.0.1"

    def test_redirect_host_localhost_in_uri_and_bind(self):
        """``redirect_host: localhost`` shows up in URI and binds to localhost.

        ``localhost`` resolves to a loopback address — the listener must
        not silently broaden to ``0.0.0.0``.
        """
        pytest.importorskip("mcp")
        import tools.mcp_oauth as mod
        from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

        cfg = {"redirect_host": "localhost"}
        _configure_callback_port(cfg)
        md = _build_client_metadata(cfg)

        port = cfg["_resolved_port"]
        assert str(md.redirect_uris[0]) == f"http://localhost:{port}/callback"
        assert mod._oauth_bind_host == "localhost"
        assert mod._oauth_bind_host != "0.0.0.0"

    def test_redirect_host_in_preregistered_client_info(self, tmp_path, monkeypatch):
        """A pre-registered client_id stores redirect_uri using redirect_host."""
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import (
            HermesTokenStorage,
            _build_client_metadata,
            _configure_callback_port,
            _maybe_preregister_client,
        )

        cfg = {
            "client_id": "preregistered-id",
            "redirect_host": "localhost",
            "redirect_port": 0,
        }
        _configure_callback_port(cfg)
        client_metadata = _build_client_metadata(cfg)
        storage = HermesTokenStorage("hostcfg")
        _maybe_preregister_client(storage, cfg, client_metadata)

        client_path = tmp_path / "mcp-tokens" / "hostcfg.client.json"
        assert client_path.exists()
        data = json.loads(client_path.read_text())
        port = cfg["_resolved_port"]
        assert data["redirect_uris"] == [f"http://localhost:{port}/callback"]
        assert data["hermes_preregistered_client"] is True

    def test_removed_preregistered_client_secret_clears_cached_client_info(
        self, tmp_path, monkeypatch
    ):
        """Removing client_id/client_secret from config must not reuse cached secrets."""
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import (
            HermesTokenStorage,
            _build_client_metadata,
            _configure_callback_port,
            _invalidate_stale_dynamic_client_info,
            _maybe_preregister_client,
        )

        cfg = {
            "client_id": "old-confidential-client",
            "client_secret": "removed-secret",
            "redirect_host": "127.0.0.1",
            "redirect_port": 54321,
        }
        _configure_callback_port(cfg)
        storage = HermesTokenStorage("removed-secret")
        _maybe_preregister_client(storage, cfg, _build_client_metadata(cfg))
        client_path = tmp_path / "mcp-tokens" / "removed-secret.client.json"
        assert json.loads(client_path.read_text())["client_secret"] == "removed-secret"

        public_cfg = {"redirect_host": "127.0.0.1", "redirect_port": 54321}
        _configure_callback_port(public_cfg)
        _invalidate_stale_dynamic_client_info(
            storage, public_cfg, _build_client_metadata(public_cfg)
        )

        assert not client_path.exists()

    def test_confidential_cached_client_info_cleared_when_config_becomes_public(
        self, tmp_path, monkeypatch
    ):
        """Legacy cached confidential client_info is stale for public-client config."""
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import (
            HermesTokenStorage,
            _build_client_metadata,
            _configure_callback_port,
            _invalidate_stale_dynamic_client_info,
        )

        cfg = {"redirect_host": "127.0.0.1", "redirect_port": 54321}
        _configure_callback_port(cfg)
        storage = HermesTokenStorage("legacy-confidential")
        client_path = tmp_path / "mcp-tokens" / "legacy-confidential.client.json"
        client_path.parent.mkdir(parents=True)
        client_path.write_text(json.dumps({
            "client_id": "legacy-confidential-client",
            "client_secret": "removed-secret",
            "redirect_uris": ["http://127.0.0.1:54321/callback"],
            "token_endpoint_auth_method": "client_secret_post",
        }))

        _invalidate_stale_dynamic_client_info(storage, cfg, _build_client_metadata(cfg))

        assert not client_path.exists()

    def test_legacy_unmarked_public_client_info_is_cleared_when_config_has_no_client_id(
        self, tmp_path, monkeypatch
    ):
        """Unmarked legacy files may be removed pre-registered clients; re-register."""
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import (
            HermesTokenStorage,
            _build_client_metadata,
            _configure_callback_port,
            _invalidate_stale_dynamic_client_info,
        )

        cfg = {"redirect_host": "127.0.0.1", "redirect_port": 54321}
        _configure_callback_port(cfg)
        storage = HermesTokenStorage("legacy-public")
        client_path = tmp_path / "mcp-tokens" / "legacy-public.client.json"
        client_path.parent.mkdir(parents=True)
        client_path.write_text(json.dumps({
            "client_id": "maybe-removed-preregistered-client",
            "redirect_uris": ["http://127.0.0.1:54321/callback"],
            "token_endpoint_auth_method": "none",
        }))

        _invalidate_stale_dynamic_client_info(storage, cfg, _build_client_metadata(cfg))

        assert not client_path.exists()

    def test_legacy_unmarked_confidential_client_info_with_refresh_is_cleared_when_public(
        self, tmp_path, monkeypatch
    ):
        """A removed confidential secret must not survive just because tokens refresh."""
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import (
            HermesTokenStorage,
            _build_client_metadata,
            _configure_callback_port,
            _invalidate_stale_dynamic_client_info,
        )

        cfg = {"redirect_host": "127.0.0.1", "redirect_port": 54321}
        _configure_callback_port(cfg)
        storage = HermesTokenStorage("legacy-confidential-refreshable")
        token_dir = tmp_path / "mcp-tokens"
        token_dir.mkdir(parents=True)
        client_path = token_dir / "legacy-confidential-refreshable.client.json"
        token_path = token_dir / "legacy-confidential-refreshable.json"
        client_path.write_text(json.dumps({
            "client_id": "legacy-confidential-client",
            "client_secret": "removed-secret",
            "redirect_uris": ["http://127.0.0.1:54321/callback"],
            "token_endpoint_auth_method": "client_secret_post",
        }))
        token_path.write_text(json.dumps({
            "access_token": "expired-access",
            "refresh_token": "refresh-me",
            "token_type": "Bearer",
            "expires_in": 0,
        }))

        _invalidate_stale_dynamic_client_info(storage, cfg, _build_client_metadata(cfg))

        assert not client_path.exists()

    def test_legacy_unmarked_client_info_is_removed_when_redirect_changes_even_with_refresh(
        self, tmp_path, monkeypatch
    ):
        """Legacy unmarked client_info is unsafe once redirect_uri changes."""
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import (
            HermesTokenStorage,
            _build_client_metadata,
            _configure_callback_port,
            _invalidate_stale_dynamic_client_info,
        )

        cfg = {"redirect_host": "127.0.0.1", "redirect_port": 54321}
        _configure_callback_port(cfg)
        storage = HermesTokenStorage("legacy-refreshable")
        token_dir = tmp_path / "mcp-tokens"
        token_dir.mkdir(parents=True)
        client_path = token_dir / "legacy-refreshable.client.json"
        token_path = token_dir / "legacy-refreshable.json"
        client_path.write_text(json.dumps({
            "client_id": "legacy-dynamic-client",
            "redirect_uris": ["http://127.0.0.1:49000/callback"],
            "token_endpoint_auth_method": "none",
        }))
        token_path.write_text(json.dumps({
            "access_token": "expired-access",
            "refresh_token": "refresh-me",
            "token_type": "Bearer",
            "expires_in": 0,
        }))

        _invalidate_stale_dynamic_client_info(storage, cfg, _build_client_metadata(cfg))

        assert not client_path.exists()

    def test_matching_public_dynamic_client_info_is_kept(self, tmp_path, monkeypatch):
        """A same-redirect public dynamic registration remains reusable."""
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import (
            HermesTokenStorage,
            _build_client_metadata,
            _configure_callback_port,
            _invalidate_stale_dynamic_client_info,
        )

        cfg = {"redirect_host": "127.0.0.1", "redirect_port": 54321}
        _configure_callback_port(cfg)
        storage = HermesTokenStorage("public-dynamic")
        client_path = tmp_path / "mcp-tokens" / "public-dynamic.client.json"
        client_path.parent.mkdir(parents=True)
        client_path.write_text(json.dumps({
            "client_id": "dynamic-public-client",
            "redirect_uris": ["http://127.0.0.1:54321/callback"],
            "token_endpoint_auth_method": "none",
            "hermes_dynamic_client": True,
        }))

        _invalidate_stale_dynamic_client_info(storage, cfg, _build_client_metadata(cfg))

        assert client_path.exists()

    def test_redirect_port_zero_picks_free_port(self):
        """``redirect_port: 0`` keeps the auto-pick behavior."""
        from tools.mcp_oauth import _configure_callback_port

        cfg = {"redirect_port": 0}
        port = _configure_callback_port(cfg)
        assert 1024 < port < 65536
        assert cfg["_resolved_port"] == port

    def test_explicit_redirect_port_preserved_in_uri(self):
        """An explicit ``redirect_port`` flows through to the URL."""
        pytest.importorskip("mcp")
        from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

        cfg = {"redirect_port": 54321, "redirect_host": "127.0.0.1"}
        _configure_callback_port(cfg)
        md = _build_client_metadata(cfg)

        assert str(md.redirect_uris[0]) == "http://127.0.0.1:54321/callback"

    def test_wait_for_callback_binds_to_configured_host(self, monkeypatch):
        """``_wait_for_callback`` for ``localhost`` binds BOTH loopback
        families on the same port and never broadens to ``0.0.0.0``.

        Single ``"localhost"`` advertised in the redirect_uri can be
        resolved by the browser to either ``127.0.0.1`` or ``::1``
        depending on the OS resolver order. Listening on only one would
        silently drop the callback whenever the browser picked the
        other family. We verify the listener covers both — the single
        ``("localhost", port)`` bind tuple from the previous design is
        now a regression to guard against.
        """
        import tools.mcp_oauth as mod
        import socket
        import asyncio

        captured: dict = {"addrs": [], "families": []}

        class _FakeServer:
            def __init__(self, addr, handler):
                captured["addrs"].append(addr)
                captured["families"].append(getattr(type(self), "address_family", socket.AF_INET))
                captured["handler"] = handler

            def handle_request(self):
                pass

            def server_close(self):
                pass

        async def instant_sleep(_seconds):
            pass

        monkeypatch.setattr(mod, "HTTPServer", _FakeServer)
        monkeypatch.setattr(mod.asyncio, "sleep", instant_sleep)
        monkeypatch.setattr(
            mod.threading, "Thread", lambda target, daemon: MagicMock()
        )

        mod._oauth_port = 51234
        mod._oauth_bind_host = "localhost"
        try:
            with pytest.raises(OAuthNonInteractiveError):
                asyncio.run(mod._wait_for_callback())
        finally:
            mod._oauth_bind_host = "127.0.0.1"

        assert ("127.0.0.1", 51234) in captured["addrs"]
        assert ("::1", 51234) in captured["addrs"]
        assert socket.AF_INET in captured["families"]
        assert socket.AF_INET6 in captured["families"]
        # No spurious bind to anything but the configured loopbacks.
        for addr, _ in captured["addrs"]:
            assert addr in {"127.0.0.1", "::1"}, (
                f"unexpected bind host {addr!r} — listener must stay loopback-only"
            )


def test_build_oauth_auth_preserves_server_url_path():
    """server_url with path is forwarded to OAuthClientProvider unmodified.

    Regression for #16015: previously ``_parse_base_url`` stripped the path,
    collapsing ``https://mcp.notion.com/mcp`` to ``https://mcp.notion.com`` and
    breaking RFC 9728 protected-resource validation against servers whose PRM
    advertises a path-scoped resource (Notion). The MCP SDK strips the path
    itself for authorization-server discovery via
    ``OAuthContext.get_authorization_base_url``; Hermes must not pre-strip.
    """
    from tools import mcp_oauth

    captured: dict = {}

    class _FakeProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch.object(mcp_oauth, "_OAUTH_AVAILABLE", True), \
         patch.object(mcp_oauth, "OAuthClientProvider", _FakeProvider), \
         patch.object(mcp_oauth, "_is_interactive", return_value=True), \
         patch.object(mcp_oauth, "_invalidate_stale_dynamic_client_info"), \
         patch.object(mcp_oauth, "_maybe_preregister_client"), \
         patch.object(mcp_oauth, "HermesTokenStorage") as mock_storage_cls:
        mock_storage_cls.return_value = MagicMock(has_cached_tokens=lambda: True)
        build_oauth_auth(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            oauth_config={},
        )

    assert captured["server_url"] == "https://mcp.notion.com/mcp"


# ---------------------------------------------------------------------------
# redirect_host validation: reject non-loopback / malformed values
# ---------------------------------------------------------------------------


class TestRedirectHostValidation:
    """``_validate_redirect_host`` (called from ``_configure_callback_port``)
    must reject any value that would broaden the callback listener beyond
    loopback or that smuggles a scheme/port/path into the host field."""

    @pytest.mark.parametrize(
        "bad_host",
        ["0.0.0.0", "192.168.1.2", "example.com"],
    )
    def test_rejects_non_loopback_host(self, bad_host):
        """Public, LAN, or all-interfaces hosts must be rejected.

        ``0.0.0.0`` would make the callback reachable from any host on the
        network, exposing the authorization-code redirect; LAN literals and
        public hostnames similarly point the listener at a non-loopback
        interface. All three must fail at config time.
        """
        from tools.mcp_oauth import _configure_callback_port

        with pytest.raises(ValueError, match="loopback"):
            _configure_callback_port({"redirect_host": bad_host})

    @pytest.mark.parametrize(
        "bad_host",
        ["http://localhost", "localhost:3000", "localhost/callback"],
    )
    def test_rejects_malformed_host_with_scheme_port_or_path(self, bad_host):
        """``redirect_host`` is a bare hostname; scheme/port/path are invalid.

        ``redirect_port`` is its own field, the path is hardcoded to
        ``/callback``, and the scheme is always ``http``. Accepting any of
        these would either silently drop user intent or build a malformed
        ``redirect_uri``.
        """
        from tools.mcp_oauth import _configure_callback_port

        with pytest.raises(ValueError):
            _configure_callback_port({"redirect_host": bad_host})


# ---------------------------------------------------------------------------
# IPv6 loopback support
# ---------------------------------------------------------------------------


class TestIPv6LoopbackRedirectHost:
    """``redirect_host: ::1`` (or ``[::1]``) builds an RFC-3986-bracketed
    redirect_uri and binds the listener to the bare IPv6 loopback."""

    @pytest.mark.parametrize("raw_host", ["::1", "[::1]"])
    def test_ipv6_loopback_in_uri_and_bind_host(self, raw_host):
        pytest.importorskip("mcp")
        import tools.mcp_oauth as mod
        from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

        cfg = {"redirect_host": raw_host}
        _configure_callback_port(cfg)
        md = _build_client_metadata(cfg)

        port = cfg["_resolved_port"]
        # URI authority must bracket the IPv6 literal.
        assert str(md.redirect_uris[0]) == f"http://[::1]:{port}/callback"
        # HTTPServer bind tuple takes the bare address (no brackets).
        assert mod._oauth_bind_host == "::1"
        assert cfg["_resolved_bind_host"] == "::1"
        assert cfg["_resolved_uri_host"] == "[::1]"

    def test_auto_port_pick_uses_ipv6_for_ipv6_bind_host(self, monkeypatch):
        """``redirect_port: 0`` with ``redirect_host: ::1`` must probe the
        IPv6 loopback (AF_INET6, bind ``::1``) when picking a free port.

        Probing IPv4 ``127.0.0.1`` and then binding IPv6 ``::1`` would only
        prove the port is free on the IPv4 loopback — the IPv6 bind could
        still race or collide. The address family of the probe must match
        the family the callback server will use.
        """
        import socket as _socket
        import tools.mcp_oauth as mod

        captured: dict = {"families": [], "binds": []}

        real_socket = _socket.socket

        class _FakeSocket:
            def __init__(self, family, kind):
                captured["families"].append(family)
                self._family = family
                self._inner = real_socket(family, kind)

            def setsockopt(self, *_a, **_k):
                pass

            def bind(self, addr):
                captured["binds"].append(addr)
                # Use a fixed port to avoid touching the network stack.
                self._sockname = (addr[0], 51999)

            def getsockname(self):
                return self._sockname

            def close(self):
                self._inner.close()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._inner.close()
                return False

        monkeypatch.setattr(mod.socket, "socket", _FakeSocket)

        cfg = {"redirect_host": "::1", "redirect_port": 0}
        port = mod._configure_callback_port(cfg)

        assert port == 51999
        assert captured["families"] == [_socket.AF_INET6]
        assert captured["binds"] == [("::1", 0)]
        assert cfg["_resolved_bind_host"] == "::1"
        assert mod._oauth_bind_host == "::1"

    def test_explicit_port_skips_probe_for_ipv6_bind_host(self, monkeypatch):
        """An explicit ``redirect_port`` must NOT probe a socket — the user
        supplied the port intentionally, and probing IPv4 to validate an
        IPv6 bind would be wrong anyway. Preserves explicit-port behavior.
        """
        import tools.mcp_oauth as mod

        called = {"socket": 0}

        def _boom(*a, **kw):
            called["socket"] += 1
            raise AssertionError(
                "explicit redirect_port must not open a probe socket"
            )

        monkeypatch.setattr(mod.socket, "socket", _boom)

        cfg = {"redirect_host": "::1", "redirect_port": 56789}
        port = mod._configure_callback_port(cfg)

        assert port == 56789
        assert called["socket"] == 0
        assert cfg["_resolved_bind_host"] == "::1"

    def test_wait_for_callback_uses_ipv6_address_family_for_ipv6_bind(
        self, monkeypatch
    ):
        """When ``_oauth_bind_host == "::1"``, ``_wait_for_callback`` must
        instantiate an HTTPServer subclass with ``address_family =
        socket.AF_INET6`` so the listener actually binds the IPv6 socket.

        Without the AF_INET6 switch, ``HTTPServer`` defaults to AF_INET and
        ``bind(("::1", port))`` raises ``OSError``, so the OAuth flow can't
        receive the callback at all.
        """
        import asyncio
        import socket as _socket
        import tools.mcp_oauth as mod

        captured: dict = {}

        class _FakeServer:
            # No address_family set — the IPv4 path would use the default
            # (AF_INET); the IPv6 subclass overrides to AF_INET6.
            def __init__(self, addr, handler):
                captured["addr"] = addr
                captured["server_cls"] = type(self)
                captured["address_family"] = getattr(
                    type(self), "address_family", None
                )

            def handle_request(self):
                pass

            def server_close(self):
                pass

        async def instant_sleep(_seconds):
            pass

        monkeypatch.setattr(mod, "HTTPServer", _FakeServer)
        monkeypatch.setattr(mod.asyncio, "sleep", instant_sleep)
        monkeypatch.setattr(
            mod.threading, "Thread", lambda target, daemon: MagicMock()
        )

        mod._oauth_port = 51234
        mod._oauth_bind_host = "::1"
        try:
            with pytest.raises(OAuthNonInteractiveError):
                asyncio.run(mod._wait_for_callback())
        finally:
            mod._oauth_bind_host = "127.0.0.1"

        # Bind tuple uses the bare ``::1``; an IPv6 subclass was selected.
        assert captured["addr"] == ("::1", 51234)
        assert captured["address_family"] == _socket.AF_INET6
        # The class actually instantiated must be a subclass of our fake
        # HTTPServer — i.e. the local _IPv6HTTPServer wrapper, not the
        # plain HTTPServer reference.
        assert captured["server_cls"] is not _FakeServer
        assert issubclass(captured["server_cls"], _FakeServer)


# ---------------------------------------------------------------------------
# Dual-stack ``localhost``: listener and free-port probe must cover both
# IPv4 ``127.0.0.1`` and IPv6 ``::1`` on the same port.
# ---------------------------------------------------------------------------


class TestLocalhostDualLoopback:
    """``redirect_host: localhost`` advertises a single hostname whose
    A/AAAA records resolve to either ``127.0.0.1`` or ``::1`` depending
    on platform (and on Linux, on glibc resolver tuning + ``/etc/hosts``).
    A browser following the OAuth redirect picks whichever the resolver
    returned — and on macOS / Windows / many Linux distros that's IPv6
    first. So the callback listener has to cover BOTH families on the
    same port. Listening only on AF_INET (the previous default whenever
    the bind_host had no colon) silently drops every IPv6-resolved
    callback.
    """

    def test_auto_port_probes_both_loopback_families(self, monkeypatch):
        """``redirect_port: 0`` with ``redirect_host: localhost`` probes
        AF_INET 127.0.0.1 AND AF_INET6 ::1 simultaneously, returning a
        port confirmed free on both. Probing only one family would let
        the listener bind succeed on that family while colliding on the
        other.
        """
        import socket as _socket
        import tools.mcp_oauth as mod

        captured: dict = {"sockets": []}
        real_socket = _socket.socket

        class _FakeSocket:
            def __init__(self, family, kind):
                captured["sockets"].append({"family": family, "binds": []})
                self._idx = len(captured["sockets"]) - 1
                self._inner = real_socket(family, kind)

            def setsockopt(self, *_a, **_k):
                pass

            def bind(self, addr):
                captured["sockets"][self._idx]["binds"].append(addr)
                # Pretend the kernel handed us port 51999 every time so
                # the multi-family-same-port assertion is checkable
                # without touching the real network stack.
                self._sockname = (addr[0], 51999)

            def getsockname(self):
                return self._sockname

            def close(self):
                self._inner.close()

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                self._inner.close()
                return False

        monkeypatch.setattr(mod.socket, "socket", _FakeSocket)

        cfg = {"redirect_host": "localhost", "redirect_port": 0}
        port = mod._configure_callback_port(cfg)

        assert port == 51999
        # Two probe sockets: one AF_INET on 127.0.0.1:0, one AF_INET6 on
        # ::1:51999 (the IPv6 verifier must bind the SAME port the IPv4
        # head returned, otherwise it isn't actually verifying coverage).
        families = [s["family"] for s in captured["sockets"]]
        assert _socket.AF_INET in families, (
            "localhost probe must include IPv4 loopback"
        )
        assert _socket.AF_INET6 in families, (
            "localhost probe must include IPv6 loopback — without this "
            "the IPv6 listener can collide on the chosen port"
        )
        # Must verify the SAME port on both families.
        ipv4_binds = next(s["binds"] for s in captured["sockets"] if s["family"] == _socket.AF_INET)
        ipv6_binds = next(s["binds"] for s in captured["sockets"] if s["family"] == _socket.AF_INET6)
        assert ipv4_binds == [("127.0.0.1", 0)]
        assert ipv6_binds == [("::1", 51999)]
        assert cfg["_resolved_bind_host"] == "localhost"
        assert cfg["_resolved_uri_host"] == "localhost"

    def test_auto_port_retries_when_ipv6_collides_with_chosen_ipv4_port(
        self, monkeypatch
    ):
        """If the port that came back free on IPv4 turns out to be in use
        on IPv6, the probe must retry rather than return a port the
        listener will fail to bind on the IPv6 family.

        Previously ``_find_free_port`` only probed one family — so
        ``localhost`` could return an IPv4-free port that the IPv6
        listener then fails to claim, and the OAuth flow silently dropped
        every IPv6-resolved callback.
        """
        import socket as _socket
        import tools.mcp_oauth as mod

        # First IPv4 attempt → port 60001. IPv6 bind on 60001 → EADDRINUSE.
        # Retry: IPv4 → 60002. IPv6 bind on 60002 → success.
        ipv4_ports = iter([60001, 60002])
        ipv6_should_fail = iter([True, False])

        class _FakeSocket:
            def __init__(self, family, kind):
                self._family = family

            def setsockopt(self, *_a, **_k):
                pass

            def bind(self, addr):
                if self._family == _socket.AF_INET:
                    self._sockname = (addr[0], next(ipv4_ports))
                else:
                    if next(ipv6_should_fail):
                        raise OSError(
                            "[Errno 98] Address already in use"
                        )
                    self._sockname = (addr[0], addr[1])

            def getsockname(self):
                return self._sockname

            def close(self):
                pass

        monkeypatch.setattr(mod.socket, "socket", _FakeSocket)

        cfg = {"redirect_host": "localhost", "redirect_port": 0}
        port = mod._configure_callback_port(cfg)

        # On the second attempt the IPv4 head returned 60002 and the IPv6
        # verify succeeded — that's the port we must surface.
        assert port == 60002

    def test_listener_starts_both_ipv4_and_ipv6_specs(self, monkeypatch):
        """``_wait_for_callback`` for ``localhost`` instantiates one
        HTTPServer per loopback family, sharing a single result dict, so
        whichever family the browser hits writes into the same callback
        result.
        """
        import asyncio
        import socket as _socket
        import tools.mcp_oauth as mod

        captured: dict = {"calls": []}

        class _FakeServer:
            def __init__(self, addr, handler):
                captured["calls"].append({
                    "addr": addr,
                    "address_family": getattr(
                        type(self), "address_family", _socket.AF_INET
                    ),
                    "handler": handler,
                })

            def handle_request(self):
                pass

            def server_close(self):
                pass

        async def instant_sleep(_s):
            pass

        monkeypatch.setattr(mod, "HTTPServer", _FakeServer)
        monkeypatch.setattr(mod.asyncio, "sleep", instant_sleep)
        monkeypatch.setattr(
            mod.threading, "Thread", lambda target, daemon: MagicMock()
        )

        mod._oauth_port = 51234
        mod._oauth_bind_host = "localhost"
        try:
            with pytest.raises(OAuthNonInteractiveError):
                asyncio.run(mod._wait_for_callback())
        finally:
            mod._oauth_bind_host = "127.0.0.1"

        addrs = [c["addr"] for c in captured["calls"]]
        families = [c["address_family"] for c in captured["calls"]]
        assert ("127.0.0.1", 51234) in addrs
        assert ("::1", 51234) in addrs
        assert _socket.AF_INET in families
        assert _socket.AF_INET6 in families
        # Every listener must share the SAME handler class so callbacks on
        # either family resolve to one shared result dict.
        handler_classes = {c["handler"] for c in captured["calls"]}
        assert len(handler_classes) == 1, (
            "dual-listener setup must share a single handler class so the "
            "browser's callback (whichever family it lands on) writes into "
            "the same result the polling loop watches"
        )

    def test_localhost_explicit_port_fails_fast_when_ipv6_bind_fails(
        self, monkeypatch
    ):
        """Explicit ``redirect_port`` with ``redirect_host: localhost``
        must fail fast (before the polling timeout) if either loopback
        family can't bind. Half-coverage would silently drop callbacks
        on the missing family for the full 5-minute timeout.
        """
        import asyncio
        import socket as _socket
        import tools.mcp_oauth as mod

        closed: list = []

        class _IPv4FakeServer:
            def __init__(self, addr, handler):
                self.addr = addr

            def handle_request(self):
                pass

            def server_close(self):
                closed.append(self.addr)

        def _server_factory(family):
            if family == _socket.AF_INET:
                return _IPv4FakeServer

            class _IPv6FailServer:
                def __init__(self, addr, handler):
                    raise OSError("[Errno 98] Address already in use")

            return _IPv6FailServer

        # Patch the factory so AF_INET succeeds and AF_INET6 raises on
        # construction (mimicking ``socket.bind`` failing inside
        # ``HTTPServer.__init__``).
        monkeypatch.setattr(mod, "_make_http_server_cls", _server_factory)

        async def instant_sleep(_s):
            raise AssertionError(
                "fail-fast: must not reach the polling sleep when one "
                "loopback family failed to bind"
            )

        monkeypatch.setattr(mod.asyncio, "sleep", instant_sleep)

        mod._oauth_port = 51234
        mod._oauth_bind_host = "localhost"
        try:
            with pytest.raises(OAuthNonInteractiveError, match="could not bind"):
                asyncio.run(mod._wait_for_callback())
        finally:
            mod._oauth_bind_host = "127.0.0.1"

        # The IPv4 server that bound successfully must be closed before
        # we raise — otherwise we'd leak the loopback port.
        assert closed == [("127.0.0.1", 51234)]


# ---------------------------------------------------------------------------
# MCPOAuthManager: rebuild provider when oauth_config changes
# ---------------------------------------------------------------------------


class TestManagerRebuildOnOAuthConfigChange:
    """``MCPOAuthManager.get_or_build_provider`` must discard and rebuild
    a cached provider when ``oauth_config`` changes for the same
    ``server_name``/``server_url``. Otherwise a user editing
    ``redirect_host``/``redirect_port`` in config.yaml would still see the
    old listener bind / redirect_uri until the process restarts."""

    def test_rebuilds_when_redirect_port_changes(self, tmp_path, monkeypatch):
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth_manager import MCPOAuthManager

        url = "https://example.com/mcp"
        mgr = MCPOAuthManager()
        p1 = mgr.get_or_build_provider("srv", url, {"redirect_port": 51111})
        p2 = mgr.get_or_build_provider("srv", url, {"redirect_port": 52222})

        assert p1 is not None and p2 is not None
        assert p1 is not p2, (
            "config change must discard the cached provider so the new "
            "redirect_port takes effect"
        )
        # New metadata reflects the new port — proves we didn't just hand
        # back a fresh wrapper around stale state.
        assert (
            str(p2.context.client_metadata.redirect_uris[0])
            == "http://127.0.0.1:52222/callback"
        )

    def test_rebuilds_when_redirect_host_changes(self, tmp_path, monkeypatch):
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth_manager import MCPOAuthManager

        url = "https://example.com/mcp"
        mgr = MCPOAuthManager()
        p1 = mgr.get_or_build_provider(
            "srv", url, {"redirect_host": "127.0.0.1", "redirect_port": 51111}
        )
        p2 = mgr.get_or_build_provider(
            "srv", url, {"redirect_host": "localhost", "redirect_port": 51111}
        )

        assert p1 is not None and p2 is not None
        assert p1 is not p2
        assert (
            str(p2.context.client_metadata.redirect_uris[0])
            == "http://localhost:51111/callback"
        )

    def test_rebuilds_on_in_place_mutation_of_same_oauth_config_dict(
        self, tmp_path, monkeypatch
    ):
        """In-place mutation of the caller's ``oauth_config`` dict must still
        invalidate the cached provider.

        If the manager stored the caller's dict by reference, comparing the
        cached entry against the (now mutated) same object would always
        evaluate equal and the manager would silently keep handing back the
        stale provider. Snapshotting via ``_snapshot_oauth_config`` defeats
        that aliasing.
        """
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth_manager import MCPOAuthManager

        url = "https://example.com/mcp"
        shared_cfg = {"redirect_host": "127.0.0.1", "redirect_port": 51111}

        mgr = MCPOAuthManager()
        p1 = mgr.get_or_build_provider("srv", url, shared_cfg)

        # Mutate the SAME dict the manager already saw — this is the case
        # the snapshot has to defend against.
        shared_cfg["redirect_port"] = 52222
        p2 = mgr.get_or_build_provider("srv", url, shared_cfg)

        assert p1 is not None and p2 is not None
        assert p1 is not p2, (
            "in-place mutation of the same oauth_config dict object must "
            "still trigger rebuild — the manager must snapshot, not alias"
        )
        assert (
            str(p2.context.client_metadata.redirect_uris[0])
            == "http://127.0.0.1:52222/callback"
        )

    def test_none_vs_empty_dict_oauth_config_is_stable(
        self, tmp_path, monkeypatch
    ):
        """``oauth_config=None`` and ``oauth_config={}`` are distinct stable
        states (snapshotting must not coerce one to the other).

        Caching a fresh entry with ``None`` then querying with ``{}`` should
        behave deterministically — currently rebuild, since the manager
        treats them as different inputs. The contract here is *stability*:
        we don't silently flip them.
        """
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth_manager import MCPOAuthManager, _snapshot_oauth_config

        # Snapshot helper preserves the None/{} distinction.
        assert _snapshot_oauth_config(None) is None
        assert _snapshot_oauth_config({}) == {}
        assert _snapshot_oauth_config({}) is not None

        url = "https://example.com/mcp"
        mgr = MCPOAuthManager()
        p_none_a = mgr.get_or_build_provider("srv", url, None)
        p_none_b = mgr.get_or_build_provider("srv", url, None)
        # Two None calls in a row reuse the same cached provider.
        assert p_none_a is p_none_b


# ---------------------------------------------------------------------------
# Per-provider callback handler isolation
# ---------------------------------------------------------------------------


class TestProviderLocalCallbackHandlerIsolation:
    """Each provider's ``callback_handler`` must bind to ITS resolved
    host/port, not whichever was last written to the module globals
    ``_oauth_port`` / ``_oauth_bind_host``.

    The cache in :class:`MCPOAuthManager` lazily builds providers, so a
    later ``_configure_callback_port`` for provider B overwrites the
    globals while provider A's advertised ``redirect_uri`` still points
    at A's original values. Without provider-local closures, calling
    A's ``callback_handler`` would bind B's host/port and the OAuth
    server's redirect would never reach a live listener.
    """

    def test_handler_a_binds_a_host_port_after_b_is_built(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        import asyncio
        from tools import mcp_oauth as mod
        from tools.mcp_oauth_manager import MCPOAuthManager

        captured_addrs: list = []

        class _RecordingServer:
            def __init__(self, addr, handler):
                captured_addrs.append(addr)

            def handle_request(self):
                pass

            def server_close(self):
                pass

        async def instant_sleep(_seconds):
            pass

        monkeypatch.setattr(mod, "HTTPServer", _RecordingServer)
        monkeypatch.setattr(mod.asyncio, "sleep", instant_sleep)
        monkeypatch.setattr(
            mod.threading, "Thread", lambda target, daemon: MagicMock()
        )

        mgr = MCPOAuthManager()
        p_a = mgr.get_or_build_provider(
            "srv-a",
            "https://a.example.com/mcp",
            {"redirect_host": "127.0.0.1", "redirect_port": 51111},
        )
        p_b = mgr.get_or_build_provider(
            "srv-b",
            "https://b.example.com/mcp",
            {"redirect_host": "localhost", "redirect_port": 52222},
        )

        assert p_a is not None and p_b is not None

        # After building B, the module globals point at B's host/port.
        assert mod._oauth_port == 52222
        assert mod._oauth_bind_host == "localhost"

        handler_a = p_a.context.callback_handler
        handler_b = p_b.context.callback_handler

        # Calling A's handler must STILL bind A's original host/port even
        # though the globals now reflect B's. This is the regression the
        # provider-local closure prevents.
        with pytest.raises(OAuthNonInteractiveError):
            asyncio.run(handler_a())
        addrs_after_a = list(captured_addrs)
        assert addrs_after_a == [("127.0.0.1", 51111)], (
            "provider A's callback_handler must bind A's original host/port "
            "regardless of any later provider's _configure_callback_port"
        )

        with pytest.raises(OAuthNonInteractiveError):
            asyncio.run(handler_b())
        # Provider B used ``redirect_host: localhost`` so the listener
        # must cover BOTH 127.0.0.1 and ::1 on B's port; preserves the
        # closure isolation guarantee for the dual-listener path.
        addrs_after_b = captured_addrs[len(addrs_after_a):]
        assert ("127.0.0.1", 52222) in addrs_after_b
        assert ("::1", 52222) in addrs_after_b


class TestProviderLocalRedirectHandlerSshHint:
    """Each provider's ``redirect_handler`` must print its SSH port-forward
    hint using the host/port resolved for THAT provider — not the module
    globals ``_oauth_port`` / ``_oauth_uri_host`` that a later
    ``build_oauth_auth`` for a different server overwrites.

    Covers two regressions: an IPv6 ``redirect_host`` must surface ``[::1]``
    (never the hard-coded ``127.0.0.1``), and a second provider built after
    the first must not retarget the first's hint.
    """

    def test_ipv6_redirect_host_hint_uses_bracketed_v6_not_v4(
        self, tmp_path, monkeypatch, capsys
    ):
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")
        monkeypatch.delenv("SSH_TTY", raising=False)

        import asyncio
        from tools import mcp_oauth as mod
        from tools.mcp_oauth_manager import MCPOAuthManager

        monkeypatch.setattr(mod, "_can_open_browser", lambda: False)

        mgr = MCPOAuthManager()
        provider = mgr.get_or_build_provider(
            "srv-v6",
            "https://v6.example.com/mcp",
            {"redirect_host": "::1", "redirect_port": 49600},
        )
        assert provider is not None

        asyncio.run(
            provider.context.redirect_handler("https://example.com/auth")
        )
        err = capsys.readouterr().err

        assert "Remote session detected" in err
        # IPv6 loopback must appear in BOTH the callback URL and ssh -L spec,
        # bracketed per RFC 3986 / OpenSSH's host:port parsing.
        assert "http://[::1]:49600/callback" in err
        assert "ssh -N -L '49600:[::1]:49600'" in err
        # The old global-reading handler hard-coded 127.0.0.1 — it must not
        # leak into a hint for a provider configured with an IPv6 host.
        assert "127.0.0.1" not in err

    def test_two_providers_keep_their_own_ports_in_ssh_hint(
        self, tmp_path, monkeypatch, capsys
    ):
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("SSH_TTY", "/dev/pts/3")
        monkeypatch.delenv("SSH_CLIENT", raising=False)

        import asyncio
        from tools import mcp_oauth as mod
        from tools.mcp_oauth_manager import MCPOAuthManager

        monkeypatch.setattr(mod, "_can_open_browser", lambda: False)

        mgr = MCPOAuthManager()
        p_a = mgr.get_or_build_provider(
            "srv-a",
            "https://a.example.com/mcp",
            {"redirect_host": "127.0.0.1", "redirect_port": 51111},
        )
        p_b = mgr.get_or_build_provider(
            "srv-b",
            "https://b.example.com/mcp",
            {"redirect_host": "127.0.0.1", "redirect_port": 52222},
        )
        assert p_a is not None and p_b is not None

        # After building B, the module globals point at B's port.
        assert mod._oauth_port == 52222

        # A's handler must STILL advertise A's port — the provider-local
        # closure shields it from B's later _configure_callback_port.
        asyncio.run(p_a.context.redirect_handler("https://example.com/auth"))
        err_a = capsys.readouterr().err
        assert "51111" in err_a
        assert "52222" not in err_a

        asyncio.run(p_b.context.redirect_handler("https://example.com/auth"))
        err_b = capsys.readouterr().err
        assert "52222" in err_b
        assert "51111" not in err_b


# ---------------------------------------------------------------------------
# Stale dynamic client_info invalidation when redirect host/port change
# ---------------------------------------------------------------------------


class TestStaleDynamicClientInfoInvalidation:
    """When dynamic registration is in use (no pre-registered ``client_id``)
    and the on-disk ``<server>.client.json`` redirect_uris no longer match
    the configured redirect_host/redirect_port, the stale file must be
    removed before provider construction so the SDK re-registers and does
    not reuse the old ``client_id`` against the new ``redirect_uri``."""

    def test_stale_client_info_removed_when_redirect_port_changes(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import build_oauth_auth

        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        client_path = d / "srv.client.json"
        client_path.write_text(json.dumps({
            "client_id": "stale-dyn-id",
            "redirect_uris": ["http://127.0.0.1:11111/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }))

        # No pre-registered client_id — purely dynamic. New port differs
        # from the stored value, so the stale file must be removed.
        provider = build_oauth_auth(
            "srv",
            "https://example.com/mcp",
            {"redirect_port": 22222},
        )
        assert provider is not None
        assert not client_path.exists(), (
            "stale dynamic client_info must be removed when redirect_port "
            "changes and no tokens are cached — otherwise the SDK reuses the old "
            "client_id with the new redirect_uri"
        )

    def test_stale_marked_dynamic_client_info_removed_even_when_tokens_cached(
        self, tmp_path, monkeypatch
    ):
        """Redirect changes require re-registration even with cached refresh tokens."""
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import build_oauth_auth

        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        client_path = d / "srv.client.json"
        token_path = d / "srv.json"
        client_path.write_text(json.dumps({
            "client_id": "refreshable-dyn-id",
            "redirect_uris": ["http://127.0.0.1:11111/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "hermes_dynamic_client": True,
        }))
        token_path.write_text(json.dumps({
            "access_token": "expired-access",
            "refresh_token": "refresh-token",
            "token_type": "Bearer",
        }))

        provider = build_oauth_auth(
            "srv",
            "https://example.com/mcp",
            {"redirect_port": 22222},
        )
        assert provider is not None
        assert not client_path.exists(), (
            "stale dynamic client_info must be removed when redirect_port changes "
            "even if a refresh token is cached; otherwise a later browser flow can "
            "reuse an old client_id with the new redirect_uri"
        )

    def test_stale_marked_dynamic_client_info_removed_without_refresh_token(
        self, tmp_path, monkeypatch
    ):
        """Access-token-only cache cannot refresh, so stale client_info is unsafe."""
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import build_oauth_auth

        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        client_path = d / "srv.client.json"
        token_path = d / "srv.json"
        client_path.write_text(json.dumps({
            "client_id": "nonrefreshable-dyn-id",
            "redirect_uris": ["http://127.0.0.1:11111/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "hermes_dynamic_client": True,
        }))
        token_path.write_text(json.dumps({
            "access_token": "access-only",
            "token_type": "Bearer",
        }))

        provider = build_oauth_auth(
            "srv",
            "https://example.com/mcp",
            {"redirect_port": 22222},
        )
        assert provider is not None
        assert not client_path.exists()

    def test_stale_client_info_removed_when_redirect_host_changes(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import build_oauth_auth

        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        client_path = d / "srv.client.json"
        client_path.write_text(json.dumps({
            "client_id": "stale-dyn-id",
            "redirect_uris": ["http://127.0.0.1:33333/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }))

        provider = build_oauth_auth(
            "srv",
            "https://example.com/mcp",
            {"redirect_host": "localhost", "redirect_port": 33333},
        )
        assert provider is not None
        assert not client_path.exists()

    def test_matching_client_info_preserved(self, tmp_path, monkeypatch):
        """A marked dynamic file with matching redirect_uris is reusable."""
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import build_oauth_auth

        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        client_path = d / "srv.client.json"
        client_path.write_text(json.dumps({
            "client_id": "good-dyn-id",
            "redirect_uris": ["http://127.0.0.1:44444/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "hermes_dynamic_client": True,
        }))

        provider = build_oauth_auth(
            "srv",
            "https://example.com/mcp",
            {"redirect_port": 44444},
        )
        assert provider is not None
        assert client_path.exists()
        data = json.loads(client_path.read_text())
        assert data["client_id"] == "good-dyn-id"

    def test_preregistered_client_id_overwrites_regardless(
        self, tmp_path, monkeypatch
    ):
        """The pre-registered ``client_id`` path is unaffected by the
        invalidation helper: ``_maybe_preregister_client`` always writes
        the file with the current redirect_uri, replacing any prior content.
        """
        pytest.importorskip("mcp")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_oauth import build_oauth_auth

        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        client_path = d / "srv.client.json"
        client_path.write_text(json.dumps({
            "client_id": "old-static-id",
            "redirect_uris": ["http://127.0.0.1:11111/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }))

        provider = build_oauth_auth(
            "srv",
            "https://example.com/mcp",
            {"client_id": "new-static-id", "redirect_port": 22222},
        )
        assert provider is not None
        assert client_path.exists()
        data = json.loads(client_path.read_text())
        assert data["client_id"] == "new-static-id"
        assert data["redirect_uris"] == ["http://127.0.0.1:22222/callback"]

