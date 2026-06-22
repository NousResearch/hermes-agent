"""Tests for tools/mcp_oauth.py — OAuth 2.1 PKCE support for MCP servers."""

import json
import os
import stat
import sys
from io import BytesIO
from unittest.mock import patch, MagicMock

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
    _paste_callback_reader,
)


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


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
        _set_interactive_stdin(monkeypatch)
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
        _set_interactive_stdin(monkeypatch)
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
        _set_interactive_stdin(monkeypatch)
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
        _set_interactive_stdin(monkeypatch)
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

    def test_noninteractive_without_cached_tokens_fails_fast(self, tmp_path, monkeypatch):
        """Without cached tokens, non-interactive mode skips browser auth."""
        pytest.importorskip("mcp.client.auth")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)

        with pytest.raises(OAuthNonInteractiveError, match="non-interactive"):
            build_oauth_auth("atlassian", "https://mcp.atlassian.com/v1/mcp")

    def test_noninteractive_with_cached_tokens_no_warning(self, tmp_path, monkeypatch, caplog):
        """With cached tokens, non-interactive mode logs no 'no cached tokens' warning."""
        pytest.importorskip("mcp.client.auth")

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
         patch.object(mcp_oauth, "_maybe_preregister_client"), \
         patch.object(mcp_oauth, "HermesTokenStorage") as mock_storage_cls:
        mock_storage_cls.return_value = MagicMock(has_cached_tokens=lambda: True)
        build_oauth_auth(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            oauth_config={},
        )

    assert captured["server_url"] == "https://mcp.notion.com/mcp"



class TestPasteCallbackReader:
    """_paste_callback_reader parses redirect URLs / query strings from stdin."""

    def _empty_result(self):
        return {"auth_code": None, "state": None, "error": None}

    def test_parses_full_local_redirect_url(self, monkeypatch):
        result = self._empty_result()
        monkeypatch.setattr(
            "sys.stdin",
            MagicMock(readline=lambda: "http://127.0.0.1:37949/callback?code=abc&state=xyz\n"),
        )
        _paste_callback_reader(result)
        assert result["auth_code"] == "abc"
        assert result["state"] == "xyz"
        assert result["error"] is None

    def test_parses_remote_provider_url(self, monkeypatch):
        """User pastes the URL their browser ended up on, including a real host."""
        result = self._empty_result()
        url = "https://mcp.linear.app/callback?code=deadbeef&state=eyJ0ZXN0Ijoi"
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: url + "\n"))
        _paste_callback_reader(result)
        assert result["auth_code"] == "deadbeef"
        assert result["state"] == "eyJ0ZXN0Ijoi"

    def test_parses_bare_query_string(self, monkeypatch):
        result = self._empty_result()
        monkeypatch.setattr(
            "sys.stdin",
            MagicMock(readline=lambda: "code=token123&state=st1\n"),
        )
        _paste_callback_reader(result)
        assert result["auth_code"] == "token123"
        assert result["state"] == "st1"

    def test_parses_leading_question_mark(self, monkeypatch):
        result = self._empty_result()
        monkeypatch.setattr(
            "sys.stdin",
            MagicMock(readline=lambda: "?code=tok&state=stA\n"),
        )
        _paste_callback_reader(result)
        assert result["auth_code"] == "tok"
        assert result["state"] == "stA"

    def test_captures_error_param(self, monkeypatch):
        result = self._empty_result()
        monkeypatch.setattr(
            "sys.stdin",
            MagicMock(readline=lambda: "https://example/cb?error=access_denied\n"),
        )
        _paste_callback_reader(result)
        assert result["auth_code"] is None
        assert result["error"] == "access_denied"

    def test_empty_input_noop(self, monkeypatch):
        result = self._empty_result()
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: ""))
        _paste_callback_reader(result)
        assert result["auth_code"] is None
        assert result["error"] is None

    def test_garbage_input_noop(self, monkeypatch, capsys):
        result = self._empty_result()
        monkeypatch.setattr(
            "sys.stdin", MagicMock(readline=lambda: "not a url at all\n")
        )
        _paste_callback_reader(result)
        assert result["auth_code"] is None
        assert result["error"] is None
        err = capsys.readouterr().err
        assert "did not contain" in err or "Could not parse" in err

    def test_skips_when_http_listener_already_won(self, monkeypatch):
        """If HTTP listener filled the result first, paste must not overwrite."""
        result = {"auth_code": "from_http", "state": "http_state", "error": None}
        monkeypatch.setattr(
            "sys.stdin",
            MagicMock(readline=lambda: "code=from_paste&state=paste_state\n"),
        )
        _paste_callback_reader(result)
        assert result["auth_code"] == "from_http"
        assert result["state"] == "http_state"

    def test_swallows_stdin_errors(self, monkeypatch):
        """OSError / interrupt on readline must not propagate."""
        result = self._empty_result()
        def raise_oserror():
            raise OSError("stdin closed")
        monkeypatch.setattr("sys.stdin", MagicMock(readline=raise_oserror))
        _paste_callback_reader(result)  # must not raise
        assert result["auth_code"] is None


class TestWaitForCallbackPasteIntegration:
    """_wait_for_callback offers the paste prompt only when interactive."""

    def test_paste_prompt_shown_on_tty(self, monkeypatch, capsys):
        import tools.mcp_oauth as mod
        mod._oauth_port = _find_free_port()
        monkeypatch.setattr(mod, "_is_interactive", lambda: True)
        # Make stdin readline block forever so HTTP listener path drives the test;
        # we just want to verify the prompt was printed and the thread spawned.
        def block_forever():
            import threading
            threading.Event().wait()
        monkeypatch.setattr("sys.stdin", MagicMock(readline=block_forever))

        async def instant_sleep(_):
            pass
        with patch.object(mod.asyncio, "sleep", instant_sleep):
            with pytest.raises(OAuthNonInteractiveError):
                asyncio.run(_wait_for_callback())
        err = capsys.readouterr().err
        assert "paste the redirect URL" in err

    def test_paste_prompt_NOT_shown_when_noninteractive(self, monkeypatch, capsys):
        """Preserves existing invariant: no input() / paste prompt in headless runs."""
        import tools.mcp_oauth as mod
        mod._oauth_port = _find_free_port()
        monkeypatch.setattr(mod, "_is_interactive", lambda: False)

        async def instant_sleep(_):
            pass
        with patch.object(mod.asyncio, "sleep", instant_sleep):
            with patch("builtins.input", side_effect=AssertionError("input() must not be called")):
                with pytest.raises(OAuthNonInteractiveError):
                    asyncio.run(_wait_for_callback())
        err = capsys.readouterr().err
        assert "paste the redirect URL" not in err


class TestPasteCallbackSkipToken:
    """User can type `skip` (or similar) at the paste prompt to bail out."""

    def _empty_result(self):
        return {"auth_code": None, "state": None, "error": None}

    @pytest.mark.parametrize("token", ["skip", "SKIP", "Skip", "cancel", "s", "n", "no", "q", "quit"])
    def test_skip_tokens_set_sentinel(self, monkeypatch, token):
        from tools.mcp_oauth import _USER_SKIPPED_SENTINEL
        result = self._empty_result()
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: token + "\n"))
        _paste_callback_reader(result)
        assert result["error"] == _USER_SKIPPED_SENTINEL
        assert result["auth_code"] is None

    def test_skip_message_printed(self, monkeypatch, capsys):
        result = self._empty_result()
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: "skip\n"))
        _paste_callback_reader(result)
        err = capsys.readouterr().err
        assert "OAuth skipped" in err
        assert "hermes mcp login" in err

    def test_skip_does_not_overwrite_http_winner(self, monkeypatch):
        """If HTTP listener already wrote a code, `skip` must not stomp it."""
        result = {"auth_code": "from_http", "state": "x", "error": None}
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: "skip\n"))
        _paste_callback_reader(result)
        assert result["auth_code"] == "from_http"
        assert result["error"] is None

    def test_skip_token_not_parsed_as_url(self, monkeypatch, capsys):
        """`skip` must NOT fall through to URL parsing (which would silently no-op)."""
        from tools.mcp_oauth import _USER_SKIPPED_SENTINEL
        result = self._empty_result()
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: "skip\n"))
        _paste_callback_reader(result)
        # Must take skip path, not the "did not contain code=" path
        assert result["error"] == _USER_SKIPPED_SENTINEL
        err = capsys.readouterr().err
        assert "did not contain" not in err


class TestWaitForCallbackSkipIntegration:
    """_wait_for_callback maps the skip sentinel to OAuthNonInteractiveError."""

    def test_skip_raises_non_interactive_error(self, monkeypatch):
        """Skip token must raise OAuthNonInteractiveError (mcp_tool handles as non-fatal)."""
        import tools.mcp_oauth as mod
        mod._oauth_port = _find_free_port()
        monkeypatch.setattr(mod, "_is_interactive", lambda: True)
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: "skip\n"))

        async def instant_sleep(_):
            pass
        with patch.object(mod.asyncio, "sleep", instant_sleep):
            with pytest.raises(OAuthNonInteractiveError, match="user_skipped"):
                asyncio.run(_wait_for_callback())

    def test_paste_prompt_mentions_skip(self, monkeypatch, capsys):
        """The interactive prompt must tell users about the skip option."""
        import tools.mcp_oauth as mod
        mod._oauth_port = _find_free_port()
        monkeypatch.setattr(mod, "_is_interactive", lambda: True)
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: "skip\n"))

        async def instant_sleep(_):
            pass
        with patch.object(mod.asyncio, "sleep", instant_sleep):
            with pytest.raises(OAuthNonInteractiveError):
                asyncio.run(_wait_for_callback())
        err = capsys.readouterr().err
        assert "skip" in err.lower()


def test_wait_for_callback_port_in_use_reports_clear_error():
    """A busy loopback callback port surfaces a clear 'already in use' error,
    not a misleading 'timed out'. Guards the stale-comment fix where the branch
    also wrongly claimed build_oauth_auth had started a server to poll."""
    import tools.mcp_oauth as mo

    with patch.object(mo, "_oauth_port", 54321), patch.object(
        mo, "HTTPServer", side_effect=OSError("address already in use")
    ):
        with pytest.raises(mo.OAuthNonInteractiveError) as excinfo:
            asyncio.run(mo._wait_for_callback())

    msg = str(excinfo.value)
    assert "54321" in msg
    assert "already in use" in msg
    assert "timed out" not in msg
