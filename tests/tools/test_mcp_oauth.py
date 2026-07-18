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
    _make_redirect_handler,
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
    """_make_redirect_handler must print an SSH tunnel hint on remote sessions."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_ssh_hint_shown_on_ssh_session(self, monkeypatch, capsys):
        import tools.mcp_oauth as mco
        monkeypatch.setattr(mco, "_is_interactive", lambda: True)
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.setattr(mco, "_can_open_browser", lambda: False)

        handler = _make_redirect_handler(49200)
        self._run(handler("https://example.com/auth?foo=bar"))

        err = capsys.readouterr().err
        assert "49200" in err
        assert "ssh -N -L" in err
        assert "Remote session detected" in err

    def test_ssh_hint_shown_via_ssh_tty(self, monkeypatch, capsys):
        import tools.mcp_oauth as mco
        monkeypatch.setattr(mco, "_is_interactive", lambda: True)
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.setenv("SSH_TTY", "/dev/pts/1")
        monkeypatch.setattr(mco, "_can_open_browser", lambda: False)

        handler = _make_redirect_handler(49201)
        self._run(handler("https://example.com/auth"))

        err = capsys.readouterr().err
        assert "49201" in err
        assert "ssh -N -L" in err

    def test_no_ssh_hint_on_local_session(self, monkeypatch, capsys):
        import tools.mcp_oauth as mco
        monkeypatch.setattr(mco, "_is_interactive", lambda: True)
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.setattr(mco, "_can_open_browser", lambda: True)
        monkeypatch.setattr("webbrowser.open", lambda url, **kw: True)

        handler = _make_redirect_handler(49202)
        self._run(handler("https://example.com/auth"))

        err = capsys.readouterr().err
        assert "ssh -N -L" not in err

    def test_no_ssh_hint_when_port_is_zero(self, monkeypatch, capsys):
        import tools.mcp_oauth as mco
        monkeypatch.setattr(mco, "_is_interactive", lambda: True)
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")
        monkeypatch.setattr(mco, "_can_open_browser", lambda: False)

        handler = _make_redirect_handler(0)
        self._run(handler("https://example.com/auth"))

        err = capsys.readouterr().err
        assert "ssh -N -L" not in err

    def test_configured_redirect_uri_shows_proxy_hint_not_tunnel(self, monkeypatch, capsys):
        """With a proxy redirect_uri, the SSH hint must not push the loopback tunnel.

        The Funnel/proxy callback reaches this machine on its own, so the
        ``ssh -N -L`` guidance would be actively misleading.
        """
        import tools.mcp_oauth as mco
        monkeypatch.setattr(mco, "_oauth_port", 49203)
        monkeypatch.setattr(mco, "_is_interactive", lambda: True)
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")
        monkeypatch.setattr(mco, "_can_open_browser", lambda: False)

        handler = _make_redirect_handler(
            49203, redirect_uri="https://oauth.example.ts.net/callback"
        )
        self._run(handler("https://example.com/auth"))

        err = capsys.readouterr().err
        assert "https://oauth.example.ts.net/callback" in err
        assert "no SSH tunnel needed" in err
        assert "ssh -N -L" not in err
        assert "127.0.0.1" not in err

    def test_configured_redirect_uri_no_hint_when_local(self, monkeypatch, capsys):
        """Off SSH, a configured redirect_uri prints no remote-session hint."""
        import tools.mcp_oauth as mco
        monkeypatch.setattr(mco, "_oauth_port", 49204)
        monkeypatch.setattr(mco, "_is_interactive", lambda: True)
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.setattr(mco, "_can_open_browser", lambda: True)
        monkeypatch.setattr("webbrowser.open", lambda url, **kw: True)

        handler = _make_redirect_handler(
            49204, redirect_uri="https://oauth.example.ts.net/callback"
        )
        self._run(handler("https://example.com/auth"))

        err = capsys.readouterr().err
        assert "Remote session detected" not in err
        assert "no SSH tunnel needed" not in err


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
# TOCTOU port reservation (#22161)
# ---------------------------------------------------------------------------

class TestCallbackPortReservation:
    """The socket picked at selection time stays bound until callback bind.

    _find_free_port() closed its probe socket before HTTPServer re-bound the
    port, leaving a race window where another process could steal it
    (#22161). _reserve_callback_port() keeps the bound socket parked in
    _reserved_sockets until _wait_for_callback adopts it.
    """

    def test_reserved_port_cannot_be_stolen(self):
        import socket as sock
        import tools.mcp_oauth as mod

        port = mod._reserve_callback_port()
        try:
            # The reservation holds the bind — a competing bind must fail.
            thief = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            with pytest.raises(OSError):
                thief.bind(("127.0.0.1", port))
            thief.close()
        finally:
            reserved = mod._reserved_sockets.pop(port, None)
            if reserved is not None:
                reserved.close()

    def test_configure_callback_port_reserves_ephemeral(self):
        import tools.mcp_oauth as mod

        cfg: dict = {}
        port = mod._configure_callback_port(cfg)
        try:
            assert cfg["_resolved_port"] == port
            assert port in mod._reserved_sockets
        finally:
            reserved = mod._reserved_sockets.pop(port, None)
            if reserved is not None:
                reserved.close()

    def test_pinned_port_is_not_reserved(self):
        import tools.mcp_oauth as mod

        cfg: dict = {"redirect_port": 49399}
        port = mod._configure_callback_port(cfg)
        assert port == 49399
        assert 49399 not in mod._reserved_sockets

    def test_reservation_pool_is_bounded(self):
        import tools.mcp_oauth as mod

        ports = [mod._reserve_callback_port() for _ in range(mod._MAX_RESERVED_SOCKETS + 3)]
        try:
            assert len(mod._reserved_sockets) <= mod._MAX_RESERVED_SOCKETS
            # newest reservations survive
            assert ports[-1] in mod._reserved_sockets
        finally:
            for p in list(mod._reserved_sockets):
                mod._reserved_sockets.pop(p).close()

    def test_wait_for_callback_adopts_reserved_socket(self, monkeypatch):
        """E2E: reserve → _wait_for_callback binds the SAME socket and the
        callback round-trips through it."""
        import asyncio
        import threading
        import urllib.request
        import tools.mcp_oauth as mod

        cfg: dict = {}
        port = mod._configure_callback_port(cfg)
        monkeypatch.setattr(mod, "_is_interactive", lambda: False)
        # Bypass the non-interactive guard — this test drives the flow directly.
        monkeypatch.setattr(mod, "_raise_if_non_interactive", lambda lead: None)

        async def drive():
            task = asyncio.create_task(mod._wait_for_callback())
            await asyncio.sleep(0.2)  # let the server adopt the socket

            def hit():
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/callback?code=abc123&state=xyz",
                    timeout=5,
                )

            t = threading.Thread(target=hit, daemon=True)
            t.start()
            return await asyncio.wait_for(task, timeout=10)

        code, state = asyncio.run(drive())
        assert code == "abc123"
        assert state == "xyz"
        # Reservation was consumed by adoption.
        assert port not in mod._reserved_sockets

    def test_concurrent_flows_keep_their_own_callback_ports(self, monkeypatch):
        """#34260: flow A's waiter listens on A's port even after flow B
        overwrites the legacy module-level global.

        This is the callback-side sibling of the #44588 redirect-handler fix:
        without a per-flow waiter, A's callback wait would bind B's port and
        A's redirect (pointing at A's port) would never be received.
        """
        import asyncio
        import threading
        import urllib.request
        import tools.mcp_oauth as mod

        monkeypatch.setattr(mod, "_is_interactive", lambda: False)
        monkeypatch.setattr(mod, "_raise_if_non_interactive", lambda lead: None)

        cfg_a: dict = {}
        port_a = mod._configure_callback_port(cfg_a)
        waiter_a = mod._make_callback_waiter(port_a)
        # Flow B configures afterwards — overwrites mod._oauth_port.
        cfg_b: dict = {}
        port_b = mod._configure_callback_port(cfg_b)
        assert mod._oauth_port == port_b != port_a

        async def drive():
            task = asyncio.create_task(waiter_a())
            await asyncio.sleep(0.2)

            def hit():
                # The redirect goes to flow A's port — where A's waiter
                # must be listening despite the clobbered global.
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port_a}/callback?code=flowA&state=sA",
                    timeout=5,
                )

            threading.Thread(target=hit, daemon=True).start()
            return await asyncio.wait_for(task, timeout=10)

        try:
            code, state = asyncio.run(drive())
        finally:
            leftover = mod._reserved_sockets.pop(port_b, None)
            if leftover is not None:
                leftover.close()
        assert code == "flowA"
        assert state == "sA"


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

    def test_suppress_interactive_oauth_disables_stdin_prompts(self, monkeypatch):
        import tools.mcp_oauth as mod

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)

        assert _is_interactive() is True
        with mod.suppress_interactive_oauth():
            assert _is_interactive() is False
        assert _is_interactive() is True

    def test_suppression_propagates_across_run_coroutine_threadsafe(self, monkeypatch):
        """#35927 core: suppression set on the discovery thread MUST reach the
        coroutine asyncio runs on a *different* (event-loop) thread — that is
        where the OAuth callback / _is_interactive() actually executes via
        run_coroutine_threadsafe. A threading.local would NOT propagate here
        (the original fix's defect); a ContextVar does."""
        import asyncio
        import threading
        import tools.mcp_oauth as mod

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        result = {}
        try:
            async def _probe_on_loop_thread():
                # runs on the loop thread, NOT the one that set suppression
                return (threading.current_thread() is not discovery_thread,
                        _is_interactive())

            discovery_thread = None

            def _discovery():
                nonlocal discovery_thread
                discovery_thread = threading.current_thread()
                with mod.suppress_interactive_oauth():
                    fut = asyncio.run_coroutine_threadsafe(
                        _probe_on_loop_thread(), loop
                    )
                    result["cross_thread"], result["interactive"] = fut.result(timeout=5)

            dt = threading.Thread(target=_discovery)
            dt.start()
            dt.join()
        finally:
            loop.call_soon_threadsafe(loop.stop)

        assert result["cross_thread"] is True, "probe must run on the loop thread"
        # The whole point: suppression must hold on the loop thread.
        assert result["interactive"] is False


class TestWaitForCallbackNoBlocking:
    """_wait_for_callback() must never call input() — it raises instead."""

    def test_raises_on_timeout_instead_of_input(self, monkeypatch):
        """Interactive session: when no auth code arrives, raises on timeout.

        Marked interactive so the fail-fast non-interactive guard (#57836)
        does not short-circuit — this test exercises the timeout path.
        """
        import tools.mcp_oauth as mod
        import asyncio

        mod._oauth_port = _find_free_port()
        monkeypatch.setattr(mod, "_is_interactive", lambda: True)
        # EOF on the paste reader so only the HTTP-listener timeout drives it.
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: ""))

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


class TestNonInteractiveFailFastAtCallbackBoundary:
    """#57836: a cached-but-unusable token (expired/revoked, refresh rejected)
    makes the MCP SDK fall through to the authorization-code flow even though
    build_oauth_auth's token-file guard passed. In a non-interactive context
    (systemd gateway, cron, background discovery) that flow must fail fast at
    the redirect/callback boundary — never launch a browser flow or bind a
    callback listener, and never block for the full timeout — so gateway
    startup is not gated on an unusable optional MCP server, and retries do not
    collide on the callback port ('Address already in use').
    """

    def test_wait_for_callback_rejects_before_binding_when_noninteractive(self, monkeypatch):
        """No listener bound and no poll loop entered when non-interactive."""
        import tools.mcp_oauth as mod
        import asyncio

        mod._oauth_port = _find_free_port()
        monkeypatch.setattr(mod, "_is_interactive", lambda: False)

        # Binding the callback listener or entering the poll loop is the bug.
        fake_server = MagicMock(side_effect=AssertionError("must not bind callback listener"))
        monkeypatch.setattr(mod, "HTTPServer", fake_server)

        async def no_sleep(_seconds):
            raise AssertionError("must not wait for the callback timeout")
        monkeypatch.setattr(mod.asyncio, "sleep", no_sleep)

        with pytest.raises(OAuthNonInteractiveError, match="interactive session"):
            asyncio.run(mod._wait_for_callback())
        fake_server.assert_not_called()

    def test_wait_for_callback_fail_fast_holds_even_with_cached_token_file(self, tmp_path, monkeypatch):
        """Guard does not depend on token-file existence.

        A stale token file on disk passes build_oauth_auth's guard, so the
        callback boundary is the only place that can reject the flow.
        """
        import tools.mcp_oauth as mod
        import asyncio

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        (d / "example.json").write_text(
            json.dumps({"access_token": "stale", "token_type": "Bearer"})
        )

        mod._oauth_port = _find_free_port()
        monkeypatch.setattr(mod, "_is_interactive", lambda: False)
        monkeypatch.setattr(
            mod, "HTTPServer", MagicMock(side_effect=AssertionError("must not bind"))
        )

        with pytest.raises(OAuthNonInteractiveError):
            asyncio.run(mod._wait_for_callback())

    def test_redirect_handler_rejects_and_does_not_open_browser(self, monkeypatch, capsys):
        """Non-interactive redirect must not print an auth URL or open a browser."""
        import tools.mcp_oauth as mod
        import asyncio

        monkeypatch.setattr(mod, "_is_interactive", lambda: False)
        monkeypatch.setattr(
            "webbrowser.open", MagicMock(side_effect=AssertionError("must not open browser"))
        )

        with pytest.raises(OAuthNonInteractiveError, match="browser authorization"):
            asyncio.run(mod._make_redirect_handler(49300)("https://idp.example.com/authorize?x=1"))

        err = capsys.readouterr().err
        assert "https://idp.example.com/authorize" not in err

    def test_boundary_errors_point_at_hermes_mcp_login(self, monkeypatch):
        """Both boundaries emit an actionable next step."""
        import tools.mcp_oauth as mod
        import asyncio

        monkeypatch.setattr(mod, "_is_interactive", lambda: False)
        with pytest.raises(OAuthNonInteractiveError, match="hermes mcp login"):
            asyncio.run(mod._make_redirect_handler(49301)("https://idp.example.com/authorize"))

        mod._oauth_port = _find_free_port()
        with pytest.raises(OAuthNonInteractiveError, match="hermes mcp login"):
            asyncio.run(mod._wait_for_callback())

    def test_guard_does_not_fire_on_interactive_redirect(self, monkeypatch, capsys):
        """Positive control: the fail-fast guard is scoped to the auth-code path.

        #57836 regression coverage asks that valid/refreshable OAuth keeps
        working non-interactively — a good token never reaches these handlers,
        so the guard must be inert once a real flow is in progress. Assert the
        interactive path still prints the URL and does not raise, proving the
        guard does not over-fire and swallow legitimate authorization.
        """
        import tools.mcp_oauth as mod
        import asyncio

        monkeypatch.setattr(mod, "_is_interactive", lambda: True)
        # Local (non-SSH) interactive session with no browser available, so the
        # handler falls through to the manual-URL print without opening a tab.
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.setattr(mod, "_can_open_browser", lambda: False)

        asyncio.run(mod._make_redirect_handler(49302)("https://idp.example.com/authorize?x=9"))

        err = capsys.readouterr().err
        assert "https://idp.example.com/authorize?x=9" in err


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


_PROXY_REDIRECT = "https://oauth.example.ts.net/callback"


def test_resolve_redirect_uri_prefers_configured_value():
    """An explicit redirect_uri in cfg overrides the localhost default."""
    from tools.mcp_oauth import _resolve_redirect_uri

    assert _resolve_redirect_uri({"redirect_uri": _PROXY_REDIRECT}, 1234) == _PROXY_REDIRECT


def test_resolve_redirect_uri_falls_back_to_localhost():
    """No redirect_uri → the loopback callback derived from the port."""
    from tools.mcp_oauth import _resolve_redirect_uri

    assert _resolve_redirect_uri({}, 1234) == "http://127.0.0.1:1234/callback"


def test_resolve_redirect_uri_empty_string_falls_back():
    """An empty redirect_uri is treated as unset (YAML ``redirect_uri:``)."""
    from tools.mcp_oauth import _resolve_redirect_uri

    assert _resolve_redirect_uri({"redirect_uri": ""}, 5678) == "http://127.0.0.1:5678/callback"


def test_resolve_redirect_uri_redirect_host_localhost():
    """``redirect_host: localhost`` swaps only the loopback hostname (WAF-safe)."""
    from tools.mcp_oauth import _resolve_redirect_uri

    assert (
        _resolve_redirect_uri({"redirect_host": "localhost"}, 1234)
        == "http://localhost:1234/callback"
    )


def test_resolve_redirect_uri_full_uri_wins_over_redirect_host():
    """An explicit redirect_uri takes precedence over redirect_host."""
    from tools.mcp_oauth import _resolve_redirect_uri

    cfg = {"redirect_uri": _PROXY_REDIRECT, "redirect_host": "localhost"}
    assert _resolve_redirect_uri(cfg, 1234) == _PROXY_REDIRECT


def test_resolve_redirect_uri_empty_redirect_host_falls_back():
    """An empty redirect_host is treated as unset (YAML ``redirect_host:``)."""
    from tools.mcp_oauth import _resolve_redirect_uri

    assert (
        _resolve_redirect_uri({"redirect_host": ""}, 9012)
        == "http://127.0.0.1:9012/callback"
    )


def test_build_client_metadata_uses_redirect_host():
    """redirect_host flows into the client metadata's redirect_uris."""
    pytest.importorskip("mcp")
    from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

    cfg = {"redirect_host": "localhost"}
    port = _configure_callback_port(cfg)
    md = _build_client_metadata(cfg)

    assert [str(u).rstrip("/") for u in md.redirect_uris] == [
        f"http://localhost:{port}/callback"
    ]


def test_build_client_metadata_uses_configured_redirect_uri():
    """A proxied redirect_uri (e.g. Tailscale Funnel) flows into the metadata.

    Without this the redirect_uri is pinned to ``http://127.0.0.1:<port>/callback``,
    which a public HTTPS proxy cannot reach.
    """
    pytest.importorskip("mcp")
    from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

    cfg = {"redirect_uri": _PROXY_REDIRECT}
    _configure_callback_port(cfg)
    md = _build_client_metadata(cfg)

    assert [str(u).rstrip("/") for u in md.redirect_uris] == [_PROXY_REDIRECT]


def test_build_client_metadata_redirect_uri_defaults_to_localhost():
    """Without redirect_uri, metadata keeps the loopback callback default."""
    pytest.importorskip("mcp")
    from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

    cfg: dict = {}
    port = _configure_callback_port(cfg)
    md = _build_client_metadata(cfg)

    assert [str(u).rstrip("/") for u in md.redirect_uris] == [
        f"http://127.0.0.1:{port}/callback"
    ]


def test_maybe_preregister_client_persists_configured_redirect_uri(tmp_path, monkeypatch):
    """Pre-registered client info records the configured redirect_uri verbatim.

    The redirect_uri on the stored client_info MUST match the one in the
    authorization request, or the provider rejects the callback.
    """
    pytest.importorskip("mcp")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth import (
        HermesTokenStorage,
        _build_client_metadata,
        _configure_callback_port,
        _maybe_preregister_client,
    )

    cfg = {"client_id": "preset-client", "redirect_uri": _PROXY_REDIRECT}
    _configure_callback_port(cfg)
    storage = HermesTokenStorage("proxy-srv")
    _maybe_preregister_client(storage, cfg, _build_client_metadata(cfg))

    written = json.loads(storage._client_info_path().read_text())
    assert [u.rstrip("/") for u in written["redirect_uris"]] == [_PROXY_REDIRECT]


def test_maybe_preregister_client_redirect_uri_defaults_to_localhost(tmp_path, monkeypatch):
    """Without redirect_uri, pre-registration falls back to the loopback callback."""
    pytest.importorskip("mcp")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth import (
        HermesTokenStorage,
        _build_client_metadata,
        _configure_callback_port,
        _maybe_preregister_client,
    )

    cfg = {"client_id": "preset-client"}
    port = _configure_callback_port(cfg)
    storage = HermesTokenStorage("loopback-srv")
    _maybe_preregister_client(storage, cfg, _build_client_metadata(cfg))

    written = json.loads(storage._client_info_path().read_text())
    assert [u.rstrip("/") for u in written["redirect_uris"]] == [
        f"http://127.0.0.1:{port}/callback"
    ]


def test_configure_callback_port_reuses_cached_client_redirect_port(tmp_path, monkeypatch):
    """Cached client registrations must keep using their registered port."""
    from tools.mcp_oauth import _configure_callback_port

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("summ")
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "summ.client.json").write_text(json.dumps({
        "client_id": "client-123",
        "redirect_uris": ["http://127.0.0.1:57727/callback"],
    }))

    cfg = {"redirect_port": 0}
    port = _configure_callback_port(cfg, storage)

    assert port == 57727
    assert cfg["_resolved_port"] == 57727


def test_configure_callback_reuses_cached_https_redirect_uri(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth import (
        HermesTokenStorage,
        _build_client_metadata,
        _configure_callback_port,
    )

    storage = HermesTokenStorage("hosted")
    storage._client_info_path().parent.mkdir(parents=True)
    storage._client_info_path().write_text(json.dumps({
        "client_id": "client-123",
        "redirect_uris": ["https://agent.example/api/mcp/oauth/callback/hosted"],
    }))

    cfg: dict = {}
    _configure_callback_port(cfg, storage)
    metadata = _build_client_metadata(cfg)

    assert str(metadata.redirect_uris[0]) == (
        "https://agent.example/api/mcp/oauth/callback/hosted"
    )


def test_configure_callback_port_explicit_overrides_cached_client_port(tmp_path, monkeypatch):
    """Explicit config wins over any cached registration."""
    from tools.mcp_oauth import _configure_callback_port

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("summ")
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "summ.client.json").write_text(json.dumps({
        "client_id": "client-123",
        "redirect_uris": ["http://127.0.0.1:57727/callback"],
    }))

    cfg = {"redirect_port": 54321}
    port = _configure_callback_port(cfg, storage)

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


def test_build_oauth_auth_wires_configured_redirect_uri_into_handler(monkeypatch, capsys):
    """The configured redirect_uri is bound into the redirect_handler closure so
    the remote-session hint stays accurate for proxied callbacks."""
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
            server_name="proxy",
            server_url="https://mcp.example.com/mcp",
            oauth_config={"redirect_uri": _PROXY_REDIRECT},
        )

    # Behavior check: on a remote session, the bound handler prints the proxy
    # callback hint (not the loopback SSH-tunnel guidance).
    monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")
    monkeypatch.setattr(mcp_oauth, "_is_interactive", lambda: True)
    monkeypatch.setattr(mcp_oauth, "_can_open_browser", lambda: False)
    asyncio.get_event_loop().run_until_complete(
        captured["redirect_handler"]("https://example.com/auth")
    )
    err = capsys.readouterr().err
    assert _PROXY_REDIRECT in err
    assert "no SSH tunnel needed" in err
    assert "ssh -N -L" not in err


def test_build_oauth_auth_handler_redirect_uri_none_when_unset(monkeypatch, capsys):
    """Without a configured redirect_uri, the bound handler falls back to the
    loopback SSH-tunnel hint."""
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
            server_name="loopback",
            server_url="https://mcp.example.com/mcp",
            oauth_config={"redirect_port": 49299},
        )

    monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")
    monkeypatch.setattr(mcp_oauth, "_is_interactive", lambda: True)
    monkeypatch.setattr(mcp_oauth, "_can_open_browser", lambda: False)
    asyncio.get_event_loop().run_until_complete(
        captured["redirect_handler"]("https://example.com/auth")
    )
    err = capsys.readouterr().err
    assert "ssh -N -L" in err
    assert "no SSH tunnel needed" not in err


def test_build_client_metadata_redirect_uri_without_path_is_normalized():
    """pydantic AnyUrl appends a trailing slash to a bare-hostname redirect_uri.

    Both _build_client_metadata and _maybe_preregister_client run the value
    through AnyUrl, so they normalize identically and stay consistent — this
    pins that behavior so a future pydantic change is caught.
    """
    pytest.importorskip("mcp")
    from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

    cfg = {"redirect_uri": "https://oauth.example.ts.net"}
    _configure_callback_port(cfg)
    md = _build_client_metadata(cfg)

    assert str(md.redirect_uris[0]) == "https://oauth.example.ts.net/"


def test_maybe_preregister_client_skips_when_no_client_id(tmp_path, monkeypatch):
    """No client_id → pre-registration is a no-op even with a configured redirect_uri."""
    pytest.importorskip("mcp")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth import (
        HermesTokenStorage,
        _build_client_metadata,
        _configure_callback_port,
        _maybe_preregister_client,
    )

    cfg = {"redirect_uri": _PROXY_REDIRECT}  # no client_id
    _configure_callback_port(cfg)
    storage = HermesTokenStorage("no-client-id-srv")
    _maybe_preregister_client(storage, cfg, _build_client_metadata(cfg))

    assert not storage._client_info_path().exists()


def test_maybe_preregister_client_redirect_uri_with_secret(tmp_path, monkeypatch):
    """redirect_uri + client_secret: callback stored verbatim, auth method confidential."""
    pytest.importorskip("mcp")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth import (
        HermesTokenStorage,
        _build_client_metadata,
        _configure_callback_port,
        _maybe_preregister_client,
    )

    cfg = {
        "client_id": "my-client",
        "client_secret": "shhh",
        "redirect_uri": _PROXY_REDIRECT,
    }
    _configure_callback_port(cfg)
    storage = HermesTokenStorage("secret-proxy-srv")
    _maybe_preregister_client(storage, cfg, _build_client_metadata(cfg))

    written = json.loads(storage._client_info_path().read_text())
    assert [u.rstrip("/") for u in written["redirect_uris"]] == [_PROXY_REDIRECT]
    assert written["client_secret"] == "shhh"
    assert written["token_endpoint_auth_method"] == "client_secret_post"



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

    def test_paste_prompt_NOT_shown_when_interactivity_suppressed(self, monkeypatch, capsys):
        """Background MCP discovery must not race the CLI/TUI stdin reader."""
        import tools.mcp_oauth as mod

        mod._oauth_port = _find_free_port()
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        monkeypatch.setattr(mod.sys, "stdin", mock_stdin)

        async def instant_sleep(_):
            pass

        with patch.object(mod.asyncio, "sleep", instant_sleep):
            with mod.suppress_interactive_oauth():
                with pytest.raises(OAuthNonInteractiveError):
                    asyncio.run(_wait_for_callback())
        err = capsys.readouterr().err
        assert "paste the redirect URL" not in err
        mock_stdin.readline.assert_not_called()


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


# ---------------------------------------------------------------------------
# poison_client_registration (GH#36767)
# ---------------------------------------------------------------------------

class TestPoisonClientRegistration:
    def test_poison_backs_up_and_removes_client_and_meta(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("srv")
        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        (d / "srv.json").write_text('{"access_token": "keep-me"}')
        (d / "srv.client.json").write_text('{"client_id": "dead"}')
        (d / "srv.meta.json").write_text('{"token_endpoint": "https://idp/token"}')

        removed = storage.poison_client_registration()

        assert removed is True
        # Client + metadata gone, forcing re-registration on the next flow.
        assert not (d / "srv.client.json").exists()
        assert not (d / "srv.meta.json").exists()
        # Backup of the client file kept for recovery.
        assert (d / "srv.client.json.bak").read_text() == '{"client_id": "dead"}'
        # Tokens are intentionally preserved.
        assert (d / "srv.json").read_text() == '{"access_token": "keep-me"}'

    def test_poison_noop_when_no_client_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("srv")
        assert storage.poison_client_registration() is False


def test_wait_for_callback_port_in_use_reports_clear_error(monkeypatch):
    """A busy loopback callback port surfaces a clear 'already in use' error,
    not a misleading 'timed out'. Guards the stale-comment fix where the branch
    also wrongly claimed build_oauth_auth had started a server to poll."""
    import tools.mcp_oauth as mo

    monkeypatch.setattr(mo, "_is_interactive", lambda: True)
    with patch.object(mo, "_oauth_port", 54321), patch.object(
        mo, "HTTPServer", side_effect=OSError("address already in use")
    ):
        with pytest.raises(mo.OAuthNonInteractiveError) as excinfo:
            asyncio.run(mo._wait_for_callback())

    msg = str(excinfo.value)
    assert "54321" in msg
    assert "already in use" in msg
    assert "timed out" not in msg
