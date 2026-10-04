"""Contracts for the Feishu / Lark user grant (``user_access_token``) — #11540.

The token endpoint, the Open API calls and the loopback callback are all exercised against real
local HTTP servers driven by real ``httpx``: the only thing redirected is the host table
(``OPEN_BASE_URLS``), so status codes, form encoding, JSON bodies and query strings are the real
ones. Mocking the transport here would hide exactly the integration bugs this flow can have.
"""

import threading
import time
from urllib.parse import parse_qs, urlparse

import pytest

from tests.tools.feishu_user_helpers import (
    FakeFeishu, configure_app, free_port, point_open_host, store_grant, token_payload)
from tools import feishu_user_auth


@pytest.fixture
def feishu_app(monkeypatch):
    configure_app(monkeypatch)


@pytest.fixture
def fake_open_host(monkeypatch):
    def _install(server, domain="feishu"):
        point_open_host(monkeypatch, server, domain)
    return _install


# ---- authorize URL ---------------------------------------------------------------


def test_authorize_url_uses_the_domains_accounts_host_and_pkce_s256():
    """A Lark grant must consent on the Lark host; PKCE is declared, and the nonce round-trips."""
    url = feishu_user_auth.build_authorize_url(
        client_id="cli_app123", redirect_uri="http://127.0.0.1:43829/feishu/callback",
        scope="offline_access search:message", state="nonce-abc", code_challenge="chal", domain="lark")
    parsed = urlparse(url)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    assert f"{parsed.scheme}://{parsed.netloc}" == feishu_user_auth.ACCOUNTS_BASE_URLS["lark"]
    assert parsed.path == feishu_user_auth.AUTHORIZE_PATH
    assert params["response_type"] == "code"
    assert params["code_challenge_method"] == "S256"
    assert params["code_challenge"] == "chal"
    assert params["state"] == "nonce-abc"
    assert params["redirect_uri"] == "http://127.0.0.1:43829/feishu/callback"


def test_adapter_qr_onboarding_and_user_oauth_share_one_host_table():
    """Two Feishu flows, one pair of hosts: a Lark tenant must not consent on a Feishu host."""
    from plugins.platforms.feishu import adapter

    assert adapter._ONBOARD_ACCOUNTS_URLS is feishu_user_auth.ACCOUNTS_BASE_URLS
    assert adapter._ONBOARD_OPEN_URLS is feishu_user_auth.OPEN_BASE_URLS


def test_default_scopes_request_offline_access_so_the_grant_outlives_the_access_token():
    """Without ``offline_access`` Feishu returns no refresh token and the grant dies in two hours."""
    assert "offline_access" in feishu_user_auth.scope_string().split()
    assert "search:message" in feishu_user_auth.scope_string().split()
    # An explicit scope wins, is de-duplicated, and keeps its order.
    assert feishu_user_auth.scope_string("b  a b") == "b a"


def test_loopback_redirect_uri_must_name_an_explicit_port():
    """The URI is pasted into the app console's allow-list, so an OS-assigned port cannot work."""
    assert feishu_user_auth.validate_redirect_uri("http://127.0.0.1:43829/feishu/callback") == (
        "127.0.0.1", 43829, "/feishu/callback")
    for bad in ("https://127.0.0.1:43829/cb", "http://example.com:80/cb", "http://127.0.0.1/cb"):
        with pytest.raises(Exception):
            feishu_user_auth.validate_redirect_uri(bad)


def test_app_credentials_point_at_setup_when_the_bot_app_is_not_configured(monkeypatch):
    monkeypatch.delenv("FEISHU_APP_ID", raising=False)
    monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)
    with pytest.raises(Exception) as excinfo:
        feishu_user_auth.app_credentials()
    assert "hermes setup" in str(excinfo.value)


# ---- login (end to end over the real loopback listener) -------------------------


def test_login_end_to_end_stores_a_usable_grant(feishu_app, fake_open_host, monkeypatch):
    """Consent → code → exchange → store → resolve, with a real callback server and real httpx."""
    port = free_port()
    redirect_uri = f"http://127.0.0.1:{port}/feishu/callback"
    captured = {}
    real_build = feishu_user_auth.build_authorize_url

    def _spy(**kwargs):
        captured.update(kwargs)
        return real_build(**kwargs)

    monkeypatch.setattr(feishu_user_auth, "build_authorize_url", _spy)

    with FakeFeishu([(200, token_payload())]) as server:
        fake_open_host(server)
        result = {}

        def _run():
            result["state"] = feishu_user_auth.login(
                redirect_uri=redirect_uri, open_browser=False, timeout_seconds=20.0)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while "state" not in captured and time.monotonic() < deadline:
            time.sleep(0.02)
        assert "state" in captured, "login never built an authorize URL"

        import httpx
        # The browser's redirect, replayed against the listener login() already owns.
        for _ in range(50):
            try:
                httpx.get(redirect_uri, params={"code": "auth-code-1", "state": captured["state"]},
                          timeout=5.0)
                break
            except httpx.ConnectError:
                time.sleep(0.05)
        thread.join(timeout=15)
        assert not thread.is_alive(), "login did not finish after the callback"

        exchange = server.form(0)

    assert exchange["grant_type"] == "authorization_code"
    assert exchange["code"] == "auth-code-1"
    assert exchange["client_id"] == "cli_app123"
    # Feishu's token endpoint is confidential-client only: the secret rides along even with PKCE.
    assert exchange["client_secret"] == "secret456"
    assert exchange["redirect_uri"] == redirect_uri
    # PKCE: the verifier is sent on the exchange and its S256 digest was sent on the authorize URL.
    import base64
    import hashlib
    digest = hashlib.sha256(exchange["code_verifier"].encode("utf-8")).digest()
    assert base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=") == captured["code_challenge"]

    assert feishu_user_auth.has_user_token() is True
    stored = feishu_user_auth.load_state()
    assert stored["access_token"] == "u-access-1"
    assert stored["refresh_token"] == "u-refresh-1"
    assert feishu_user_auth.auth_status()["logged_in"] is True
    # A fresh, unexpired token resolves without touching the network at all.
    resolved = feishu_user_auth.resolve_user_access_token()
    assert resolved["access_token"] == "u-access-1"


def test_login_rejects_a_callback_whose_state_does_not_match(feishu_app, monkeypatch):
    """CSRF guard: a redirect Hermes did not initiate must never reach the token endpoint."""
    port = free_port()
    redirect_uri = f"http://127.0.0.1:{port}/feishu/callback"
    captured = {}
    real_build = feishu_user_auth.build_authorize_url
    monkeypatch.setattr(
        feishu_user_auth, "build_authorize_url",
        lambda **kw: (captured.update(kw), real_build(**kw))[1])

    def _never(*_args, **_kwargs):
        raise AssertionError("token endpoint must not be called on a state mismatch")

    monkeypatch.setattr(feishu_user_auth, "exchange_code", _never)

    outcome = {}

    def _run():
        try:
            feishu_user_auth.login(redirect_uri=redirect_uri, open_browser=False, timeout_seconds=20.0)
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while "state" not in captured and time.monotonic() < deadline:
        time.sleep(0.02)
    import httpx
    for _ in range(50):
        try:
            httpx.get(redirect_uri, params={"code": "c", "state": "not-the-nonce"}, timeout=5.0)
            break
        except httpx.ConnectError:
            time.sleep(0.05)
    thread.join(timeout=15)
    assert "state mismatch" in str(outcome.get("error", ""))


# ---- refresh rotation ------------------------------------------------------------


def _store_expired_grant(refresh_token="u-refresh-1"):
    store_grant(access_token="u-access-old", refresh_token=refresh_token,
                expires_at="2020-01-01T00:00:00+00:00")


def test_expiring_grant_refreshes_and_the_rotated_refresh_token_is_persisted(
        feishu_app, fake_open_host):
    """Feishu kills the old refresh token the instant a new pair is minted.

    So the rotated pair must be on disk by the time the resolver returns: a second resolver that
    reads the store afterwards has to see ``u-refresh-2``, never the token Feishu already revoked.
    """
    _store_expired_grant()
    with FakeFeishu([(200, token_payload(access_token="u-access-2", refresh_token="u-refresh-2"))]) as server:
        fake_open_host(server)
        resolved = feishu_user_auth.resolve_user_access_token()
        refresh_form = server.form(0)

    assert refresh_form["grant_type"] == "refresh_token"
    assert refresh_form["refresh_token"] == "u-refresh-1"
    assert refresh_form["client_secret"] == "secret456"
    assert resolved["access_token"] == "u-access-2"
    stored = feishu_user_auth.load_state()
    assert stored["refresh_token"] == "u-refresh-2"
    assert stored["access_token"] == "u-access-2"


def test_a_refresh_response_without_a_refresh_token_keeps_the_stored_one(feishu_app, fake_open_host):
    """Blanking it on an omitted field would silently downgrade the grant to single-use."""
    _store_expired_grant()
    payload = token_payload(access_token="u-access-2")
    payload.pop("refresh_token")
    with FakeFeishu([(200, payload)]) as server:
        fake_open_host(server)
        feishu_user_auth.resolve_user_access_token()
    assert feishu_user_auth.load_state()["refresh_token"] == "u-refresh-1"


def test_a_revoked_grant_is_quarantined_so_the_next_call_fails_without_a_round_trip(
        feishu_app, fake_open_host):
    """One 400 from the token endpoint, then no further HTTP: the tokens are gone, not retried."""
    _store_expired_grant()
    with FakeFeishu([(400, {"error": "invalid_grant",
                             "error_description": "refresh token expired"})]) as server:
        fake_open_host(server)
        with pytest.raises(Exception) as first:
            feishu_user_auth.resolve_user_access_token()
        assert "refresh token expired" in str(first.value)
        assert len(server.requests) == 1

        with pytest.raises(Exception):
            feishu_user_auth.resolve_user_access_token()
        assert len(server.requests) == 1, "a quarantined grant must not be retried over the network"

    stored = feishu_user_auth.load_state()
    assert not stored.get("refresh_token")
    assert stored["last_auth_error"]["relogin_required"] is True
    assert feishu_user_auth.has_user_token() is False


def test_logout_forgets_the_grant_and_reports_whether_there_was_one(feishu_app):
    _store_expired_grant()
    assert feishu_user_auth.clear_state() is True
    assert feishu_user_auth.load_state() is None
    assert feishu_user_auth.clear_state() is False
