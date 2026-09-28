"""Tests for tools/mcp_oauth.py — OAuth 2.1 PKCE support for MCP servers."""

import json
import stat
import sys
import time
from io import BytesIO
from unittest.mock import patch, MagicMock
from urllib.parse import quote

import pytest

import asyncio

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK 1.26.0+ required for OAuth support",
)

from tools.mcp_oauth import (
    HermesTokenStorage,
    OAuthNonInteractiveError,
    build_oauth_auth,
    remove_oauth_tokens,
    _cached_redirect,
    _can_open_browser,
    _is_interactive,
    _make_callback_handler,
    _paste_callback_reader,
)


def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for_callback():
    """Await the per-flow waiter on the legacy module-level port (the removed shim)."""
    import tools.mcp_oauth as mod
    return await mod._make_callback_waiter(mod._oauth_port)()


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


def _hit_callback_when_ready(url: str, timeout: float = 15.0) -> None:
    """Drive the loopback callback as soon as the waiter's server answers.

    Polls instead of sleeping a fixed interval: the reserved socket is bound
    but NOT listening until ``_wait_for_callback`` adopts it, so attempts
    before adoption fail fast with a connection error.
    """
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except OSError:
            time.sleep(0.01)
    raise AssertionError(f"callback listener never came up: {url}")


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

    def test_client_info_with_secret_uses_client_secret_post(self, tmp_path, monkeypatch):
        from mcp.shared.auth import OAuthClientInformationFull

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("supabase")
        client_info = OAuthClientInformationFull.model_validate({
            "client_id": "client-id",
            "client_secret": "secret",
            "redirect_uris": ["http://127.0.0.1:12345/callback"],
        })

        asyncio.run(storage.set_client_info(client_info))
        loaded = asyncio.run(storage.get_client_info())

        assert loaded is not None
        assert loaded.token_endpoint_auth_method == "client_secret_post"
        client_path = tmp_path / "mcp-tokens" / "supabase.client.json"
        assert json.loads(client_path.read_text())["token_endpoint_auth_method"] == "client_secret_post"

    def test_client_info_with_secret_and_none_method_is_coerced(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        token_dir = tmp_path / "mcp-tokens"
        token_dir.mkdir(parents=True)
        client_path = token_dir / "supabase.client.json"
        client_path.write_text(json.dumps({
            "client_id": "client-id",
            "client_secret": "secret",
            "redirect_uris": ["http://127.0.0.1:12345/callback"],
            "token_endpoint_auth_method": "none",
        }))

        loaded = asyncio.run(HermesTokenStorage("supabase").get_client_info())

        assert loaded is not None
        assert loaded.token_endpoint_auth_method == "client_secret_post"
        assert json.loads(client_path.read_text())["token_endpoint_auth_method"] == "client_secret_post"


    def test_corrupt_tokens_returns_none(self, tmp_path, monkeypatch):
        import asyncio
        from mcp.shared.auth import OAuthMetadata
        from tools.mcp_oauth_device import DeviceOAuthMetadata

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("bad-server")
        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        (d / "bad-server.json").write_text("NOT VALID JSON{{{")
        assert asyncio.run(storage.get_tokens()) is None
        for raw in ('NOT VALID JSON{{{', '[]', 'null', '"cached-secret"', '42', 'true', '{}'):
            path = d / "bad-server.meta.json"
            path.write_text(raw)
            assert storage.load_oauth_metadata() is None
            assert path.read_text() == raw

        metadata = {"issuer": "https://example.com", "token_endpoint": "https://example.com/token",
                    "response_types_supported": ["code"], "authorization_endpoint": "https://example.com/auth"}
        for device in (False, True):
            if device:
                metadata.pop("authorization_endpoint")
                metadata["device_authorization_endpoint"] = "https://example.com/device"
            path = d / "bad-server.meta.json"
            path.write_text(json.dumps(metadata))
            loaded = storage.load_oauth_metadata()
            assert type(loaded) is (DeviceOAuthMetadata if device else OAuthMetadata)
            assert str(loaded.token_endpoint) == metadata["token_endpoint"]
            assert json.loads(path.read_text()) == metadata

    def test_corrupt_tokens_warning_never_echoes_the_token_material(self, tmp_path, monkeypatch, caplog):
        """A pydantic ValidationError's str() includes the raw input; the corrupt-store warning must
        name the failing fields only (#102308)."""
        import asyncio
        import logging

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("bad-server")
        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        secret = "sk-live-QQQQQQQQ"  # short enough that pydantic's input echo does not elide it
        # access_token must be a str: a one-element list fails validation on THAT field, and pydantic's
        # message echoes the failing field's input — i.e. the token.
        (d / "bad-server.json").write_text(json.dumps({"access_token": [secret], "token_type": "Bearer"}))

        with caplog.at_level(logging.WARNING, logger="tools.mcp_oauth"):
            assert asyncio.run(storage.get_tokens()) is None
        assert any("Corrupt" in r.message for r in caplog.records)
        assert secret not in caplog.text


# ---------------------------------------------------------------------------
# build_oauth_auth
# ---------------------------------------------------------------------------

class TestBuildOAuthAuth:
    def test_returns_none_without_sdk(self, monkeypatch):
        import tools.mcp_oauth as mod
        monkeypatch.setattr(mod, "_OAUTH_AVAILABLE", False)
        result = build_oauth_auth("test", "https://example.com")
        assert result is None


    def test_scope_passed_through(self, tmp_path, monkeypatch):
        pytest.importorskip("mcp.client.auth")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_interactive_stdin(monkeypatch)
        provider = build_oauth_auth("scoped", "https://example.com/mcp", {
            "scope": "read write admin",
        })
        assert provider is not None
        assert provider.context.client_metadata.scope == "read write admin"


    @pytest.mark.asyncio
    async def test_token_response_accepts_201_created(self, tmp_path, monkeypatch):
        import httpx

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_interactive_stdin(monkeypatch)
        provider = build_oauth_auth("supabase", "https://mcp.supabase.com/mcp")
        assert provider is not None
        response = httpx.Response(201, json={
            "access_token": "access-token",
            "token_type": "Bearer",
            "refresh_token": "refresh-token",
        })

        await provider._handle_token_response(response)

        tokens = provider.context.current_tokens
        assert tokens is not None
        assert tokens.access_token == "access-token"
        token_path = tmp_path / "mcp-tokens" / "supabase.json"
        assert token_path.exists()
        assert json.loads(token_path.read_text())["access_token"] == "access-token"


    @pytest.mark.asyncio
    async def test_failed_token_exchange_carries_a_bounded_redacted_excerpt(self, tmp_path, monkeypatch):
        """A non-2xx body names the cause (WAF "Request blocked" vs ``invalid_grant``) without HTML,
        beyond 200 characters or credential-shaped spans (#115329)."""
        import httpx
        from mcp.client.auth.oauth2 import OAuthTokenError

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_interactive_stdin(monkeypatch)
        provider = build_oauth_auth("supabase", "https://mcp.supabase.com/mcp")
        waf = ("<HTML><HEAD><TITLE>ERROR</TITLE></HEAD><BODY><H1>403 ERROR</H1>\n  Request blocked.\n"
               "Bearer leaked-bearer-token <PRE>" + "x" * 400 + "</PRE></BODY></HTML>")

        with pytest.raises(OAuthTokenError) as exc_info:
            await provider._handle_token_response(httpx.Response(403, content=waf.encode()))

        message = str(exc_info.value)
        assert "Request blocked." in message
        assert "[REDACTED]" in message
        assert "<" not in message and "leaked-bearer-token" not in message
        assert len(message) <= len("Token exchange failed (403): ") + 200
        assert provider.context.current_tokens is None





# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_can_open_browser_false_in_ssh(self, monkeypatch):
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")
        assert _can_open_browser() is False

    def test_can_open_browser_true_with_display(self, monkeypatch):
        # No ``os.name`` pin: on Linux this exercises the DISPLAY branch for
        # real, and on macOS/Windows the function early-returns True anyway —
        # the assertion holds on every host without faking one.
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        assert _can_open_browser() is True




# ---------------------------------------------------------------------------
# Path traversal protection
# ---------------------------------------------------------------------------

class TestPathTraversal:
    """Verify server_name is sanitized to prevent path traversal."""

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

    def _fake_get(self, HandlerClass, path):
        handler = HandlerClass.__new__(HandlerClass)
        handler.path = path
        handler.wfile = BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.do_GET()

    def test_handler_writes_to_own_result(self):
        HandlerClass, result = _make_callback_handler()
        assert result["auth_code"] is None

        self._fake_get(HandlerClass, "/callback?code=test123&state=mystate")

        assert result["auth_code"] == "test123"
        assert result["state"] == "mystate"

    def test_handler_captures_error(self):
        HandlerClass, result = _make_callback_handler()

        self._fake_get(HandlerClass, "/callback?error=access_denied")

        assert result["auth_code"] is None
        assert result["error"] == "access_denied"


class TestCallbackHandlerErrorEscaping:
    """Regression: a hostile ``error`` parameter must be HTML-escaped before
    being reflected into the callback response body (reflected XSS)."""

    def test_hostile_error_is_escaped_in_response_body(self):
        HandlerClass, result = _make_callback_handler()

        handler = HandlerClass.__new__(HandlerClass)
        handler.path = "/callback?error=" + quote("<script>alert(1)</script>")
        handler.wfile = BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.do_GET()

        body = handler.wfile.getvalue().decode("utf-8")
        assert "<script>" not in body
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
        # The raw (unescaped) value is still captured for programmatic use.
        assert result["error"] == "<script>alert(1)</script>"


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

    def test_pinned_port_is_not_reserved(self):
        import tools.mcp_oauth as mod

        cfg: dict = {"redirect_port": 49399}
        port = mod._configure_callback_port(cfg)
        assert port == 49399
        assert cfg["_resolved_port"] == 49399
        assert 49399 not in mod._reserved_sockets

    @pytest.mark.usefixtures("require_mcp_2_sdk")  # asserts the 2.0-only AuthorizationCodeResult.code
    def test_wait_for_callback_adopts_reserved_socket(self, monkeypatch):
        """E2E: reserve → _wait_for_callback binds the SAME socket and the
        callback round-trips through it."""
        import asyncio
        import threading
        import tools.mcp_oauth as mod

        # cimd: false keeps this on the ephemeral branch. A CIMD-eligible
        # config would take a pinned port instead, and this test would pass
        # while never exercising _reserve_callback_port at all.
        cfg: dict = {"cimd": False}
        port = mod._configure_callback_port(cfg)
        monkeypatch.setattr(mod, "_is_interactive", lambda: False)
        # Bypass the non-interactive guard — this test drives the flow directly.
        monkeypatch.setattr(mod, "_raise_if_non_interactive", lambda lead: None)

        async def drive():
            task = asyncio.create_task(_wait_for_callback())
            threading.Thread(
                target=_hit_callback_when_ready,
                args=(f"http://127.0.0.1:{port}/callback?code=abc123&state=xyz",),
                daemon=True,
            ).start()
            return await asyncio.wait_for(task, timeout=20)

        # mcp 2.0's callback_handler contract returns an
        # AuthorizationCodeResult, not the legacy (code, state) tuple.
        result = asyncio.run(drive())
        assert result.code == "abc123"
        assert result.state == "xyz"
        # Reservation was consumed by adoption.
        assert port not in mod._reserved_sockets

    @pytest.mark.usefixtures("require_mcp_2_sdk")  # asserts the 2.0-only AuthorizationCodeResult.code
    def test_concurrent_flows_keep_their_own_callback_ports(self, monkeypatch):
        """#34260: flow A's waiter listens on A's port even after flow B
        overwrites the legacy module-level global.

        This is the callback-side sibling of the #44588 redirect-handler fix:
        without a per-flow waiter, A's callback wait would bind B's port and
        A's redirect (pointing at A's port) would never be received.
        """
        import asyncio
        import threading
        import tools.mcp_oauth as mod

        monkeypatch.setattr(mod, "_is_interactive", lambda: False)
        monkeypatch.setattr(mod, "_raise_if_non_interactive", lambda lead: None)

        # cimd: false keeps both flows on ephemeral ports, which is where the
        # #34260 clobbering happens; the pinned range has its own coverage in
        # tests/tools/test_mcp_cimd.py.
        cfg_a: dict = {"cimd": False}
        port_a = mod._configure_callback_port(cfg_a)
        waiter_a = mod._make_callback_waiter(port_a)
        # Flow B configures afterwards — overwrites mod._oauth_port.
        cfg_b: dict = {"cimd": False}
        port_b = mod._configure_callback_port(cfg_b)
        assert mod._oauth_port == port_b != port_a

        async def drive():
            task = asyncio.create_task(waiter_a())
            # The redirect goes to flow A's port — where A's waiter must be
            # listening despite the clobbered global.
            threading.Thread(
                target=_hit_callback_when_ready,
                args=(f"http://127.0.0.1:{port_a}/callback?code=flowA&state=sA",),
                daemon=True,
            ).start()
            return await asyncio.wait_for(task, timeout=20)

        try:
            result = asyncio.run(drive())
        finally:
            leftover = mod._reserved_sockets.pop(port_b, None)
            if leftover is not None:
                leftover.close()
        assert result.code == "flowA"
        assert result.state == "sA"

    @staticmethod
    def _seed_client_info(tmp_path, payload):
        """Write *payload* verbatim to the real ``mcp-tokens/srv.client.json`` under a temp home."""
        storage = HermesTokenStorage("srv", hermes_home=tmp_path)
        path = storage._client_info_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return storage

    @pytest.mark.parametrize("bad_uri", [
        "http://127.0.0.1:abc/callback",      # .port raises: non-numeric
        "http://127.0.0.1:99999/callback",    # .port raises: out of range
        "http://[bad/callback",               # urlparse itself raises: bad IPv6 bracket
    ])
    def test_cached_redirect_skips_malformed_entries(self, tmp_path, bad_uri):
        """DCR-supplied redirect_uris persist to client.json. urlparse() alone does not
        validate ports — .port is lazy and raises ValueError on access — so the try/except
        around urlparse never fires. A poisoned entry must be skipped like every other
        malformed one, not crash the whole OAuth flow (#112568)."""
        storage = self._seed_client_info(tmp_path, {
            "client_id": "client-a",
            "redirect_uris": [bad_uri, "http://127.0.0.1:1455/callback", "https://proxy.example.com/cb"]})
        assert _cached_redirect(storage) == ("https://proxy.example.com/cb", 1455)

    @pytest.mark.parametrize("payload", [
        ["not", "a", "dict"],                    # non-dict client.json: .get would AttributeError
        {"redirect_uris": 123},                  # non-iterable redirect_uris: for would TypeError
        {"redirect_uris": {"a": 1}},             # dict redirect_uris: iterate keys, nothing matches
        {"redirect_uris": None},                 # explicit null
        {"client_id": "c"},                      # missing key entirely
    ])
    def test_cached_redirect_tolerates_misshaped_client_info(self, tmp_path, payload):
        """_read_json returns whatever the file holds — the crash class isn't limited to
        bad URIs inside a well-formed list. Any misshaped payload must degrade to
        (None, None), not propagate AttributeError/TypeError through the OAuth flow (#112568)."""
        storage = self._seed_client_info(tmp_path, payload)
        assert _cached_redirect(storage) == (None, None)


    @pytest.mark.parametrize("payload", [
        {"client_id": "c", "redirect_uris": ["http://127.0.0.1:abc/callback"]},  # the issue's repro
        ["x"],                                                                    # non-dict client.json
    ])
    def test_malformed_client_info_flow_reserves_fresh_ephemeral_port(self, tmp_path, payload):
        """Flow-level: the login path calls _configure_callback_port(cfg, storage) and the SDK
        then calls storage.get_client_info(). A poisoned client.json must fall through to a
        freshly reserved ephemeral port and read as "no registration", so the flow re-registers
        instead of crashing on every attempt until the file is removed by hand (#112568)."""
        import tools.mcp_oauth as mod

        storage = self._seed_client_info(tmp_path, payload)
        cfg: dict = {"cimd": False}  # keep the fresh-port branch, as the sibling tests do
        port = mod._configure_callback_port(cfg, storage)
        try:
            assert port == cfg["_resolved_port"] > 0
            assert port in mod._reserved_sockets  # only a truly fresh pick is parked
            assert asyncio.run(storage.get_client_info()) is None
        finally:
            reserved = mod._reserved_sockets.pop(port, None)
            if reserved is not None:
                reserved.close()


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


# ---------------------------------------------------------------------------
# Client-change token invalidation (port of cline/cline#12983)
# ---------------------------------------------------------------------------

class TestInvalidateTokensOnClientChange:
    """Editing oauth.client_id/client_secret must discard tokens minted
    under the previous client identity (they can only fail with
    invalid_client), while an unchanged identity preserves them."""

    def _seed(self, tmp_path, monkeypatch, client_id="client-a",
              client_secret=None):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("chg-server")
        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True, exist_ok=True)
        info = {"client_id": client_id, "redirect_uris": ["http://localhost:1455/callback"]}
        if client_secret:
            info["client_secret"] = client_secret
        (d / "chg-server.client.json").write_text(json.dumps(info))
        (d / "chg-server.json").write_text(json.dumps({
            "access_token": "old-token", "token_type": "Bearer",
        }))
        (d / "chg-server.meta.json").write_text(json.dumps({
            "issuer": "https://idp.example",
            "authorization_endpoint": "https://idp.example/auth",
            "token_endpoint": "https://idp.example/token",
        }))
        return storage, d

    def test_changed_client_id_drops_tokens(self, tmp_path, monkeypatch):
        from tools.mcp_oauth import _invalidate_tokens_on_client_change
        storage, d = self._seed(tmp_path, monkeypatch)
        _invalidate_tokens_on_client_change(storage, "client-b", None)
        assert not (d / "chg-server.json").exists()
        assert not (d / "chg-server.meta.json").exists()
        # client.json is left for _maybe_preregister_client to overwrite
        assert (d / "chg-server.client.json").exists()

    def test_changed_secret_drops_tokens(self, tmp_path, monkeypatch):
        from tools.mcp_oauth import _invalidate_tokens_on_client_change
        storage, d = self._seed(tmp_path, monkeypatch,
                                client_secret="old-secret")
        _invalidate_tokens_on_client_change(storage, "client-a", "new-secret")
        assert not (d / "chg-server.json").exists()

    def test_same_client_preserves_tokens(self, tmp_path, monkeypatch):
        from tools.mcp_oauth import _invalidate_tokens_on_client_change
        storage, d = self._seed(tmp_path, monkeypatch,
                                client_secret="sec")
        _invalidate_tokens_on_client_change(storage, "client-a", "sec")
        assert (d / "chg-server.json").exists()
        assert (d / "chg-server.meta.json").exists()

    def test_no_prior_client_info_is_noop(self, tmp_path, monkeypatch):
        from tools.mcp_oauth import _invalidate_tokens_on_client_change
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("fresh-server")
        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True, exist_ok=True)
        (d / "fresh-server.json").write_text(json.dumps({
            "access_token": "tok", "token_type": "Bearer",
        }))
        _invalidate_tokens_on_client_change(storage, "client-x", None)
        # No recorded client identity -> nothing provably stale.
        assert (d / "fresh-server.json").exists()

    def test_preregister_flow_invalidates_end_to_end(self, tmp_path, monkeypatch):
        """_maybe_preregister_client wires the check in before overwriting
        client.json — the full config-edit flow drops stale tokens."""
        pytest.importorskip("mcp")
        from tools.mcp_oauth import (
            _build_client_metadata, _maybe_preregister_client,
        )
        storage, d = self._seed(tmp_path, monkeypatch)
        cfg = {"client_id": "client-b", "_resolved_port": 1455}
        meta = _build_client_metadata(dict(cfg))
        _maybe_preregister_client(storage, cfg, meta)
        assert not (d / "chg-server.json").exists(), (
            "tokens minted under client-a must not survive switch to client-b"
        )
        info = json.loads((d / "chg-server.client.json").read_text())
        assert info["client_id"] == "client-b"



# ---------------------------------------------------------------------------
# Non-interactive / startup-safety tests
# ---------------------------------------------------------------------------

class TestIsInteractive:
    """_is_interactive() detects headless/daemon/container environments."""

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
            asyncio.run(_wait_for_callback())
        fake_server.assert_not_called()

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

_PROXY_REDIRECT = "https://oauth.example.ts.net/callback"


@pytest.mark.parametrize("cfg, expected_auth", [
    ({"cimd": False}, "none"),                       # public client
    ({"client_secret": "shh"}, "client_secret_post"),  # confidential client
])
def test_build_client_metadata_token_endpoint_auth(cfg, expected_auth):
    pytest.importorskip("mcp")
    from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

    _configure_callback_port(cfg)
    md = _build_client_metadata(cfg)
    assert md.token_endpoint_auth_method == expected_auth
    assert "authorization_code" in md.grant_types
    assert "refresh_token" in md.grant_types


@pytest.mark.parametrize("cfg, expected", [
    ({"redirect_uri": _PROXY_REDIRECT}, _PROXY_REDIRECT),
    ({}, "http://127.0.0.1:1234/callback"),
    # ``redirect_host: localhost`` swaps only the loopback hostname (WAF-safe)
    ({"redirect_host": "localhost"}, "http://localhost:1234/callback"),
])
def test_resolve_redirect_uri(cfg, expected):
    from tools.mcp_oauth import _resolve_redirect_uri

    assert _resolve_redirect_uri(cfg, 1234) == expected


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
         patch.object(mcp_oauth, "HermesOAuthClientProvider", _FakeProvider), \
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

    def test_parses_pasted_callback(self, monkeypatch):
        result = self._empty_result()
        pasted = "http://127.0.0.1:37949/callback?code=abc&state=xyz\n"
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: pasted))
        _paste_callback_reader(result)
        assert result["auth_code"] == "abc"
        assert result["state"] == "xyz"
        assert result["error"] is None


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


    def test_paste_prompt_NOT_shown_when_interactivity_suppressed(self, monkeypatch):
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
        mock_stdin.readline.assert_not_called()


class TestPasteCallbackSkipToken:
    """User can type `skip` (or similar) at the paste prompt to bail out."""

    def _empty_result(self):
        return {"auth_code": None, "state": None, "error": None}

    @pytest.mark.parametrize("token", ["skip", "QUIT"])
    def test_skip_tokens_set_sentinel(self, monkeypatch, token):
        from tools.mcp_oauth import _USER_SKIPPED_SENTINEL
        result = self._empty_result()
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: token + "\n"))
        _paste_callback_reader(result)
        assert result["error"] == _USER_SKIPPED_SENTINEL
        assert result["auth_code"] is None

    def test_skip_does_not_overwrite_http_winner(self, monkeypatch):
        """If HTTP listener already wrote a code, `skip` must not stomp it."""
        result = {"auth_code": "from_http", "state": "x", "error": None}
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: "skip\n"))
        _paste_callback_reader(result)
        assert result["auth_code"] == "from_http"
        assert result["error"] is None


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




def test_cancelled_waiter_releases_pinned_port_for_retry(monkeypatch):
    """A flow cancelled mid-wait (handshake timeout) must leave its pinned/cached port bindable: the
    retry reuses the same port and previously died with EADDRINUSE because the listener thread parked
    in select() kept the closed socket alive (#113771)."""
    import socket
    import tools.mcp_oauth as mo

    monkeypatch.setattr(mo, "_is_interactive", lambda: False)
    monkeypatch.setattr(mo, "_raise_if_non_interactive", lambda lead: None)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    waiter = mo._make_callback_waiter(port, timeout=30)

    async def drive():
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.3)  # listener bound and parked in its poll
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with socket.socket() as again:  # what the retry's _start_callback_server does
            again.bind(("127.0.0.1", port))

    asyncio.run(drive())


# ---------------------------------------------------------------------------
# Figma remote MCP DCR allowlist workarounds
# ---------------------------------------------------------------------------


def test_figma_provider_defaults_set_allowlisted_client_name():
    from tools.mcp_oauth import (
        apply_oauth_provider_defaults,
        _FIGMA_DCR_CLIENT_NAME,
        _FIGMA_DEFAULT_SCOPE,
    )

    cfg = apply_oauth_provider_defaults(
        {},
        server_name="figma",
        server_url="https://mcp.figma.com/mcp",
    )
    assert cfg["client_name"] == _FIGMA_DCR_CLIENT_NAME
    assert cfg["scope"] == _FIGMA_DEFAULT_SCOPE


def test_humanize_non_registration_403_passthrough():
    from tools.mcp_oauth import humanize_oauth_registration_error

    assert (
        humanize_oauth_registration_error(
            "linear",
            RuntimeError("HTTP 403: insufficient_scope"),
            server_url="https://mcp.linear.app/mcp",
        )
        is None
    )
