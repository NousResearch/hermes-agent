"""Tests for the MCP OAuth manager (tools/mcp_oauth_manager.py).

The manager consolidates the eight scattered MCP-OAuth call sites into a
single object with disk-mtime watch, dedup'd 401 handling, and a provider
cache. See `tools/mcp_oauth_manager.py` for design rationale.
"""
import json
import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK 1.26.0+ required for OAuth support",
)


def test_manager_isolates_same_named_servers_by_profile_home(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import MCPOAuthManager

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    for home, access_token in ((profile_a, "TOKEN_A"), (profile_b, "TOKEN_B")):
        token = set_hermes_home_override(home)
        try:
            storage = HermesTokenStorage("shared")
            storage._tokens_path().parent.mkdir(parents=True, exist_ok=True)
            storage._tokens_path().write_text(
                f'{{"access_token":"{access_token}","token_type":"Bearer","expires_in":3600}}'
            , encoding="utf-8")
        finally:
            reset_hermes_home_override(token)

    manager = MCPOAuthManager()
    providers = []
    for home in (profile_a, profile_b):
        token = set_hermes_home_override(home)
        try:
            provider = manager.get_or_build_provider("shared", "https://mcp.example/mcp", {})
            asyncio.run(provider._initialize())
            providers.append(provider)
        finally:
            reset_hermes_home_override(token)

    assert providers[0] is not providers[1]
    assert providers[0].context.current_tokens.access_token == "TOKEN_A"
    assert providers[1].context.current_tokens.access_token == "TOKEN_B"


def test_manager_restore_entry_preserves_newer_concurrent_entry(tmp_path, monkeypatch):
    from tools.mcp_oauth_manager import MCPOAuthManager

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    manager = MCPOAuthManager()
    old_provider = manager.get_or_build_provider("shared", "https://old.example", {})
    old_entry = manager.remove("shared")
    new_provider = manager.get_or_build_provider("shared", "https://new.example", {})

    manager.restore_entry("shared", old_entry)

    assert manager.get_or_build_provider("shared", "https://new.example", {}) is new_provider
    assert new_provider is not old_provider

def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)




@pytest.mark.asyncio
async def test_first_observation_401_still_allows_in_place_refresh(tmp_path, monkeypatch):
    """A 401 on the first disk observation seeds the baseline but must still
    report refreshable when the SDK can refresh in place (GH#39551)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, _ProviderEntry

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(json.dumps({"access_token": "OLD"}), encoding="utf-8")

    class _Ctx:
        current_tokens = object()  # tokens already in memory (in-process sign-in)

        def can_refresh_token(self):
            return True

    class _Provider:
        context = _Ctx()
        _initialized = True

    mgr = MCPOAuthManager()
    provider = _Provider()
    mgr._entries[mgr._key("srv")] = _ProviderEntry(
        server_url="https://example.com/mcp", oauth_config=None, provider=provider,
    )
    assert await mgr.handle_401("srv", failed_access_token="OLD") is True
    # Baseline seeding must not have torn down the live provider.
    assert provider._initialized is True
    assert mgr._entries[mgr._key("srv")].last_mtime_ns != 0


@pytest.mark.asyncio
async def test_external_login_after_absent_tokens_file_forces_reload(tmp_path, monkeypatch):
    """Provider built before login (no file, no in-memory tokens) must reload
    once another process writes the tokens file (GH#39551 review)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, _ProviderEntry

    class _Ctx:
        current_tokens = None

    class _Provider:
        context = _Ctx()
        _initialized = True

    mgr = MCPOAuthManager()
    provider = _Provider()
    mgr._entries[mgr._key("srv")] = _ProviderEntry(
        server_url="https://example.com/mcp", oauth_config=None, provider=provider,
    )
    assert await mgr.invalidate_if_disk_changed("srv") is False  # file absent
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(json.dumps({"access_token": "NEW"}), encoding="utf-8")
    assert await mgr.invalidate_if_disk_changed("srv") is True
    assert provider._initialized is False


@pytest.mark.asyncio
async def test_handle_401_dedup_survives_even_if_task_reference_dropped(tmp_path, monkeypatch):
    """Concurrent 401s share one handler task and all callers resolve.

    Regression guard: if the manager ever stops holding a strong reference
    to the `_do_handle` task, this test can intermittently hang when the
    task is GC'd between the ``await`` checkpoints inside ``_do_handle``.
    Running it in CI with ``gc.collect()`` mid-flight (below) exercises
    that window.
    """
    import asyncio
    import gc

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, _ProviderEntry

    mgr = MCPOAuthManager()

    class _DummyProvider:
        context = None

    mgr._entries[mgr._key("srv")] = _ProviderEntry(
        server_url="https://example.com/mcp",
        oauth_config=None,
        provider=_DummyProvider(),
    )

    # Fan out N concurrent callers sharing the same failed token so all
    # collapse onto a single deduped handler future.
    async def _caller():
        return await mgr.handle_401("srv", failed_access_token="TOK")

    tasks = [asyncio.create_task(_caller()) for _ in range(8)]
    # Give the event loop one tick to schedule _do_handle, then force GC.
    await asyncio.sleep(0)
    gc.collect()

    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
    assert results == [False] * 8
    # Let the shared _do_handle task's discard done-callback (call_soon) run.
    await asyncio.sleep(0)
    assert len(mgr._inflight_tasks) == 0


# ---------------------------------------------------------------------------
# invalid_client auto-heal (GH#36767) — _maybe_flag_poisoned_client
# ---------------------------------------------------------------------------

import asyncio
from types import SimpleNamespace


def _fake_response(status, url, body):
    """A minimal stand-in for the httpx.Response the SDK feeds our bridge."""
    resp = MagicMock()
    resp.status_code = status
    resp.request = SimpleNamespace(url=url)

    async def _aread():
        return body

    resp.aread = _aread
    return resp


def _provider_with_token_endpoint(tmp_path, oauth_config, token_endpoint, monkeypatch):
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()
    # Provider construction fails fast in a non-interactive environment with no
    # cached tokens (mcp_oauth_manager.py guard). The hermetic test env has no
    # TTY, so present an interactive stdin to reach the code under test.
    _set_interactive_stdin(monkeypatch)
    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider(
        "srv", "https://mcp.example.com", {"redirect_port": 54321, **oauth_config}
    )
    provider.context.oauth_metadata = SimpleNamespace(token_endpoint=token_endpoint)
    provider._initialized = True
    return provider




def test_invalid_client_metadata_does_not_trip(tmp_path, monkeypatch):
    """RFC 7591 `invalid_client_metadata` must NOT be mistaken for invalid_client."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "live"}', encoding="utf-8")
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    resp = _fake_response(
        400, "https://idp.example.com/oauth/token", b'{"error":"invalid_client_metadata"}'
    )

    asyncio.run(provider._maybe_flag_poisoned_client(resp))

    assert (d / "srv.client.json").exists()
    assert provider._initialized is True


class _FakeMeta:
    """Metadata stub usable by both detection and the post-flow persist hook."""

    def __init__(self, token_endpoint):
        self.token_endpoint = token_endpoint

    def model_dump(self, **kwargs):
        return {"token_endpoint": self.token_endpoint}


def test_bridge_forwards_requests_and_poisons_on_token_endpoint_400(
    tmp_path, monkeypatch
):
    """Drive the REAL async_auth_flow bridge to prove the inserted detection
    hook does not break the bidirectional asend() forwarding contract — the
    genuinely fragile part. A patched SDK base generator stands in for the
    real OAuth flow so we control exactly which response the bridge sees.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token_ep = "https://idp.example.com/oauth/token"
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "dead"}', encoding="utf-8")

    forwarded = []

    async def fake_base_flow(self, request):
        # Mimic the SDK: yield the request, receive the response, then finish.
        async with self.context.lock:
            forwarded.append(("out", request))
            response = yield request
            forwarded.append(("in", response))

    from mcp.client.auth.oauth2 import OAuthClientProvider
    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", fake_base_flow)

    provider = _provider_with_token_endpoint(tmp_path, {}, token_ep, monkeypatch)
    provider.context.oauth_metadata = _FakeMeta(token_ep)

    sentinel_request = object()
    poison_resp = _fake_response(400, token_ep, b'{"error":"invalid_client"}')

    async def drive():
        gen = provider.async_auth_flow(sentinel_request)
        out0 = await gen.__anext__()
        assert out0 is sentinel_request  # request forwarded unchanged
        try:
            await gen.asend(poison_resp)
        except StopAsyncIteration:
            pass

    asyncio.run(drive())

    # The poison response reached the inner generator (forwarding intact)...
    assert ("in", poison_resp) in forwarded
    # ...and the detection hook fired.
    assert not (d / "srv.client.json").exists()
    assert provider._initialized is False
    assert provider.context.client_info is None
@pytest.mark.asyncio
async def test_manager_provider_token_exchange_includes_dcr_secret(tmp_path, monkeypatch):
    """The manager provider path applies the same Supabase DCR secret fix."""
    from urllib.parse import parse_qs

    from mcp.shared.auth import OAuthClientInformationFull
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    reset_manager_for_tests()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("supabase", "https://mcp.supabase.com/mcp", None)
    assert provider is not None
    redirect_uris = provider.context.client_metadata.redirect_uris
    assert redirect_uris is not None
    provider.context.client_info = OAuthClientInformationFull.model_validate({
        "client_id": "client-id",
        "client_secret": "secret",
        "redirect_uris": [str(redirect_uris[0])],
        "token_endpoint_auth_method": "none",
    })

    request = await provider._exchange_token_authorization_code("auth-code", "verifier")
    body = parse_qs(request.content.decode())

    assert body["client_secret"] == ["secret"]
    assert provider.context.client_info is not None
    assert provider.context.client_info.token_endpoint_auth_method == "client_secret_post"


@pytest.mark.asyncio
async def test_manager_malformed_201_token_response_does_not_expose_body(
    tmp_path, monkeypatch
):
    from mcp.client.auth.oauth2 import OAuthTokenError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )

    with pytest.raises(OAuthTokenError, match="^Invalid token response$") as exc_info:
        await provider._handle_token_response(
            _fake_response(
                201,
                "https://idp.example.com/oauth/token",
                b'{"access_token": {"secret": "access-secret"}}',
            )
        )

    assert "access-secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_manager_token_read_error_does_not_expose_body(tmp_path, monkeypatch):
    import httpx
    from mcp.client.auth.oauth2 import OAuthTokenError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )

    class _ReadErrorResponse:
        status_code = 201

        async def aread(self):
            raise httpx.ReadError("access-secret refresh-secret")

    with pytest.raises(OAuthTokenError, match="^Invalid token response$") as exc_info:
        await provider._handle_token_response(_ReadErrorResponse())

    assert "access-secret" not in str(exc_info.value)
    assert "refresh-secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_manager_malformed_201_refresh_response_clears_tokens(
    tmp_path, monkeypatch, caplog
):
    import logging

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    provider.context.current_tokens = object()

    response = _fake_response(
        201,
        "https://idp.example.com/oauth/token",
        b'{"refresh_token": "refresh-secret"}',
    )
    with caplog.at_level(logging.WARNING, logger="tools.mcp_oauth_manager"):
        result = await provider._handle_refresh_response(response)

    assert result is False
    assert provider.context.current_tokens is None
    assert "refresh-secret" not in caplog.text


@pytest.mark.asyncio
async def test_manager_refresh_read_error_clears_tokens(tmp_path, monkeypatch):
    import httpx

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    provider.context.current_tokens = object()

    class _ReadErrorResponse:
        status_code = 201

        async def aread(self):
            raise httpx.ReadError("body read failed")

    result = await provider._handle_refresh_response(_ReadErrorResponse())

    assert result is False
    assert provider.context.current_tokens is None


@pytest.mark.asyncio
async def test_refresh_response_without_refresh_token_keeps_stored_one(tmp_path, monkeypatch):
    """RFC 6749 §6: an AS that does not rotate omits refresh_token; the prior one must survive in
    the live provider AND on disk, or the server dies at the next expiry (#62333)."""
    import json
    from mcp.shared.auth import OAuthToken

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    provider.context.current_tokens = OAuthToken(
        access_token="at-1", token_type="Bearer", expires_in=3600, refresh_token="rt-keep", scope="read"
    )
    provider.context.client_info = SimpleNamespace(client_id="cid")

    body = b'{"access_token": "at-2", "token_type": "Bearer", "expires_in": 3600}'
    assert await provider._handle_refresh_response(
        _fake_response(200, "https://idp.example.com/oauth/token", body)
    )

    on_disk = json.loads((tmp_path / "mcp-tokens" / "srv.json").read_text(encoding="utf-8"))
    assert provider.context.current_tokens.access_token == "at-2"
    assert provider.context.current_tokens.refresh_token == "rt-keep" == on_disk["refresh_token"]
    assert provider.context.current_tokens.scope == "read" == on_disk["scope"]
    assert provider.context.can_refresh_token()


@pytest.mark.asyncio
async def test_refresh_response_with_new_refresh_token_rotates(tmp_path, monkeypatch):
    """A rotating AS's new refresh_token replaces the stored one (carry-forward fills gaps only)."""
    import json
    from mcp.shared.auth import OAuthToken

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    provider.context.current_tokens = OAuthToken(
        access_token="at-1", token_type="Bearer", expires_in=3600, refresh_token="rt-old"
    )

    body = b'{"access_token": "at-2", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "rt-new"}'
    assert await provider._handle_refresh_response(
        _fake_response(200, "https://idp.example.com/oauth/token", body)
    )

    on_disk = json.loads((tmp_path / "mcp-tokens" / "srv.json").read_text(encoding="utf-8"))
    assert provider.context.current_tokens.refresh_token == "rt-new" == on_disk["refresh_token"]


# ---------------------------------------------------------------------------
# Cross-process refresh-token rotation (single-use refresh tokens)
#
# Two Hermes backends routinely share one HERMES_HOME (desktop `serve` +
# `gateway run`). With a provider that rotates refresh tokens, the loser of the
# race POSTs a token the winner already consumed and gets 400 — while a valid
# replacement sits on disk. Clearing state there forces an interactive browser
# reauth that a cron/background context cannot satisfy.
# ---------------------------------------------------------------------------


def _token(access, refresh, expires_in=3600):
    from mcp.shared.auth import OAuthToken

    return OAuthToken(
        access_token=access,
        token_type="Bearer",
        expires_in=expires_in,
        refresh_token=refresh,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 429, 503])
async def test_refresh_failure_recovers_token_rotated_by_peer(tmp_path, monkeypatch, status):
    """A peer rotated the refresh token: recover from disk instead of clearing."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )

    # We hold R1 in memory and are about to fail with it.
    provider.context.current_tokens = _token("A1", "R1")
    # The peer process already persisted its replacement.
    await provider.context.storage.set_tokens(_token("A2", "R2"))

    resp = _fake_response(
        status, "https://idp.example.com/oauth/token", b'{"error":"invalid_grant"}'
    )
    result = await provider._handle_refresh_response(resp)

    assert result is True, "a rotated-token race must be recoverable"
    assert provider.context.current_tokens.access_token == "A2"
    assert provider.context.current_tokens.refresh_token == "R2"


@pytest.mark.asyncio
async def test_refresh_400_rejects_disk_token_without_refresh_token(
    tmp_path, monkeypatch
):
    """A disk token with no refresh token is a dead end, not a recovery.

    Its access token may still be inside its TTL, so the naive "is it
    different and currently valid?" test says yes — but adopting it only
    defers the reauth to expiry, with no way to refresh in between. Recovery
    must require a refresh token to recover *onto*.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )

    provider.context.current_tokens = _token("A1", "R1")
    # Different access token, still valid, but nothing to refresh with later.
    await provider.context.storage.set_tokens(_token("A2", None))

    resp = _fake_response(
        400, "https://idp.example.com/oauth/token", b'{"error":"invalid_grant"}'
    )
    result = await provider._handle_refresh_response(resp)

    assert result is False, "a token with no refresh token must not be adopted"
    assert provider.context.current_tokens is None





@pytest.mark.asyncio
async def test_refresh_400_still_clears_when_disk_is_same_token(tmp_path, monkeypatch):
    """A genuine invalid_grant clears only memory and still permits SDK browser reauth."""
    from urllib.parse import parse_qs, urlsplit
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://idp.example.com/oauth/token"
    provider = _fenced_provider(tmp_path, monkeypatch, endpoint)
    storage = provider.context.storage
    storage.bind_issuer("https://idp.example.com")
    await storage.set_tokens(_token("A1", "R1"))
    await storage.set_client_info(provider.context.client_info)
    disk_before = storage._tokens_path().read_bytes()
    browser_calls = []
    state = []

    async def redirect(url):
        browser_calls.append("redirect")
        state.append(parse_qs(urlsplit(url).query)["state"][0])

    async def callback():
        browser_calls.append("callback")
        return SimpleNamespace(code="auth-code", state=state[0], iss="https://idp.example.com")

    provider.context.redirect_handler = redirect
    provider.context.callback_handler = callback
    grants = []

    def responder(request):
        if request.method == "POST":
            grant = parse_qs(request.content.decode())["grant_type"][0]
            grants.append(grant)
            if grant == "refresh_token":
                return httpx.Response(400, request=request, json={"error": "invalid_grant"})
            return httpx.Response(200, request=request, json=_token("A2", "R2").model_dump(mode="json"))
        if "oauth-protected-resource" in str(request.url):
            return httpx.Response(200, request=request, json={
                "resource": "https://mcp.example.com", "authorization_servers": ["https://idp.example.com"]
            })
        if "oauth-authorization-server" in str(request.url):
            return httpx.Response(200, request=request, json=provider.context.oauth_metadata.model_dump(mode="json"))
        if not request.headers.get("Authorization"):
            assert provider.context.current_tokens is None
            assert storage._tokens_path().read_bytes() == disk_before
            return httpx.Response(401, request=request)
        return httpx.Response(200, request=request)

    await _drive_flow(provider, responder)
    assert grants == ["refresh_token", "authorization_code"]
    assert browser_calls == ["redirect", "callback"]
    assert provider.context.current_tokens.refresh_token == "R2"


@pytest.mark.asyncio
async def test_refresh_400_does_not_recover_expired_disk_token(tmp_path, monkeypatch):
    """A *different* but already-expired disk token is not a recovery."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )

    provider.context.current_tokens = _token("A1", "R1")
    await provider.context.storage.set_tokens(_token("A2", "R2", expires_in=-60))

    resp = _fake_response(
        400, "https://idp.example.com/oauth/token", b'{"error":"invalid_grant"}'
    )
    result = await provider._handle_refresh_response(resp)

    assert result is False
    assert provider.context.current_tokens is None


@pytest.mark.asyncio
async def test_refresh_400_does_not_recover_tokenless_disk_entry(
    tmp_path, monkeypatch
):
    """A disk entry without an access token is not a recovery."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )

    provider.context.current_tokens = _token("A1", "R1")
    # Rotated refresh token, but the access token is empty — recovering here
    # would ship an Authorization header with no credential.
    await provider.context.storage.set_tokens(_token("", "R2"))

    resp = _fake_response(
        400, "https://idp.example.com/oauth/token", b'{"error":"invalid_grant"}'
    )
    result = await provider._handle_refresh_response(resp)

    assert result is False
    assert provider.context.current_tokens is None


@pytest.mark.asyncio
async def test_refresh_400_recovery_never_logs_token_material(
    tmp_path, monkeypatch, caplog
):
    """The recovery path must not leak secrets into logs."""
    import logging

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )

    provider.context.current_tokens = _token("access-secret", "refresh-secret")
    await provider.context.storage.set_tokens(
        _token("rotated-access-secret", "rotated-refresh-secret")
    )

    resp = _fake_response(
        400, "https://idp.example.com/oauth/token", b'{"error":"invalid_grant"}'
    )
    with caplog.at_level(logging.DEBUG):
        result = await provider._handle_refresh_response(resp)

    assert result is True
    assert "refresh-secret" not in caplog.text
    assert "rotated-refresh-secret" not in caplog.text
    assert "rotated-access-secret" not in caplog.text


# ---------------------------------------------------------------------------
# Refresh fence: one refresh generation is consumed by exactly one holder
# ---------------------------------------------------------------------------


def _fenced_provider(tmp_path, monkeypatch, endpoint):
    """A real provider holding an EXPIRED (A1, R1) pair, ready to refresh.

    The SDK only refreshes when ``can_refresh_token()`` sees client_info, and
    ``_store_tokens`` reads ``oauth_metadata.issuer``: both need real models.
    """
    from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata

    provider = _provider_with_token_endpoint(tmp_path, {}, endpoint, monkeypatch)
    provider.context.oauth_metadata = OAuthMetadata(
        issuer="https://idp.example.com",
        authorization_endpoint="https://idp.example.com/authorize",
        token_endpoint=endpoint,
    )
    provider.context.client_info = OAuthClientInformationFull.model_validate(
        {"client_id": "client-id", "redirect_uris": ["http://localhost/cb"]}
    )
    provider.context.current_tokens = _token("A1", "R1")
    provider.context.token_expiry_time = time.time() - 10
    return provider


async def _drive_flow(provider, responder):
    """Pump the auth flow the way httpx does: one asend(response) per yielded request.

    Yields to the event loop before answering so a concurrent flow gets to
    contend for the fence while this one is "on the wire".
    """
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()

    gen = provider.async_auth_flow(httpx.Request("GET", "https://mcp.example.com/mcp"))
    out = await gen.asend(None)
    while True:
        await asyncio.sleep(0)
        try:
            out = await gen.asend(responder(out))
        except StopAsyncIteration:
            return


@pytest.mark.asyncio
@pytest.mark.parametrize("status,retry_after,delay", [
    (429, "3", 3),
    (429, "900", 900),
    (503, "date", 900),
    (500, None, 1),
    (503, "malformed", 1),
    (429, "nan", 1),
    (429, "inf", 1),
    (429, "-inf", 1),
])
async def test_transient_refresh_preserves_grant_and_defers_post(
    tmp_path, monkeypatch, caplog, status, retry_after, delay
):
    """Temporary outages must not trigger anonymous requests, reauth, or early refreshes."""
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime
    from agent import retry_utils
    from tools import mcp_oauth, mcp_oauth_provider
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_errors import _classify_mcp_failure

    httpx = sdk_httpx()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://idp.example.com/oauth/token"
    provider = _fenced_provider(tmp_path, monkeypatch, endpoint)
    storage = provider.context.storage
    provider.context.current_tokens = _token("access-secret", "refresh-secret")
    provider.context.client_info.client_secret = "client-secret"
    provider.context.client_info.token_endpoint_auth_method = "client_secret_post"
    storage.bind_issuer("https://idp.example.com")
    await storage.set_tokens(provider.context.current_tokens)
    await storage.set_client_info(provider.context.client_info)
    paths = (storage._tokens_path(), storage._client_info_path())
    disk_before = [path.read_bytes() for path in paths]
    tokens_before = provider.context.current_tokens.model_dump()
    client_before = provider.context.client_info.model_dump()
    clock = [100.0]
    monkeypatch.setattr(mcp_oauth_provider, "monotonic", lambda: clock[0], raising=False)
    fixed_now = datetime(2026, 10, 4, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(retry_utils, "datetime", FixedDateTime)
    if retry_after == "date":
        retry_after = format_datetime(fixed_now + timedelta(seconds=delay), usegmt=True)
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    browser_calls = []

    async def unexpected_browser(*args):
        browser_calls.append(args)
        raise AssertionError("Transient refresh must not start browser authorization")

    provider.context.redirect_handler = unexpected_browser
    provider.context.callback_handler = unexpected_browser
    request = httpx.Request("GET", "https://mcp.example.com/mcp")
    flow = provider.async_auth_flow(request)
    try:
        refresh = await flow.asend(None)
        assert refresh.method == "POST" and str(refresh.url) == endpoint
        with pytest.raises(ConnectionError) as failed:
            await flow.asend(httpx.Response(
                status, headers=headers, request=refresh, content=b"response-body-secret"
            ))
    finally:
        await flow.aclose()

    assert _classify_mcp_failure(failed.value) == "transient"
    assert provider.context.current_tokens.model_dump() == tokens_before
    assert provider.context.client_info.model_dump() == client_before
    assert [path.read_bytes() for path in paths] == disk_before
    assert browser_calls == []
    for secret in ("access-secret", "refresh-secret", "client-secret", "response-body-secret"):
        assert secret not in str(failed.value) + caplog.text

    # Exercise the actual file lock, not just the provider's descriptor bookkeeping.
    fd = await mcp_oauth.acquire_refresh_fence(storage._tokens_path(), timeout=0)
    try:
        for elapsed in (0, delay - 0.01):
            clock[0] = 100 + elapsed
            with pytest.raises(ConnectionError) as deferred:
                await _drive_flow(provider, lambda req: pytest.fail("Refresh sent before deadline"))
            assert _classify_mcp_failure(deferred.value) == "transient"
            async with provider.context.lock:
                assert provider.context.current_tokens.model_dump() == tokens_before
    finally:
        mcp_oauth.release_refresh_fence(fd)
    assert [path.read_bytes() for path in paths] == disk_before

    clock[0] = 100 + delay
    sent = []

    def success(req):
        sent.append(req)
        body = _token("A2", "R2").model_dump(mode="json", exclude_none=True)
        return httpx.Response(200, request=req, json=body if req.method == "POST" else {})

    await _drive_flow(provider, success)
    assert [req.method for req in sent] == ["POST", "GET"]
    assert sent[-1].headers["Authorization"] == "Bearer A2"
    assert provider.context.current_tokens.refresh_token == "R2"
    assert (await storage.get_tokens()).refresh_token == "R2"
    assert provider.context.client_info.model_dump() == client_before
    assert paths[1].read_bytes() == disk_before[1]
    assert browser_calls == []
    fd = await mcp_oauth.acquire_refresh_fence(storage._tokens_path(), timeout=0)
    mcp_oauth.release_refresh_fence(fd)


@pytest.mark.asyncio
async def test_rejected_expired_peer_preserves_expiry_through_refresh_cooldown(
    tmp_path, monkeypatch
):
    """Rejecting an expired peer must never renew the held token's absolute TTL."""
    from tools import mcp_oauth, mcp_oauth_provider
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://idp.example.com/oauth/token"
    template = _fenced_provider(tmp_path, monkeypatch, endpoint)
    # The manager reloads disk before each flow, masking a resurrected in-memory TTL.
    provider = mcp_oauth.build_oauth_auth(
        "srv", "https://mcp.example.com", {"redirect_port": 54321}
    )
    provider.context.oauth_metadata = template.context.oauth_metadata
    provider.context.client_info = template.context.client_info
    provider.context.current_tokens = template.context.current_tokens
    provider.context.token_expiry_time = template.context.token_expiry_time
    provider._initialized = True
    old_tokens = provider.context.current_tokens
    old_expiry = provider.context.token_expiry_time
    storage = provider.context.storage
    storage.bind_issuer("https://idp.example.com")
    await storage.set_tokens(old_tokens)
    clock = [100.0]
    monkeypatch.setattr(mcp_oauth_provider, "monotonic", lambda: clock[0], raising=False)

    async def unexpected_browser(*args):
        pytest.fail("Rejected expired peer must not start browser authorization")

    provider.context.redirect_handler = unexpected_browser
    provider.context.callback_handler = unexpected_browser
    flow = provider.async_auth_flow(httpx.Request("GET", "https://mcp.example.com/mcp"))
    try:
        refresh = await flow.asend(None)
        assert refresh.method == "POST" and str(refresh.url) == endpoint
        with pytest.raises(ConnectionError, match="temporarily unavailable"):
            await flow.asend(httpx.Response(
                503, request=refresh, headers={"Retry-After": "60"}
            ))
    finally:
        await flow.aclose()

    assert provider.context.current_tokens is old_tokens
    assert provider.context.token_expiry_time == old_expiry
    assert not provider.context.is_token_valid()
    # A peer rotates during cooldown, but its access token is already expired.
    await storage.set_tokens(_token("A2", "R2", expires_in=-60))
    disk_before = storage._tokens_path().read_bytes()
    for elapsed in (1, 59):
        clock[0] = 100.0 + elapsed
        flow = provider.async_auth_flow(httpx.Request("GET", "https://mcp.example.com/mcp"))
        try:
            with pytest.raises(ConnectionError, match="retry deadline"):
                await flow.asend(None)
        finally:
            await flow.aclose()
        assert provider.context.current_tokens is old_tokens
        assert provider.context.token_expiry_time == old_expiry
        assert not provider.context.is_token_valid()
        assert storage._tokens_path().read_bytes() == disk_before
        assert provider._hermes_fence is None


@pytest.mark.asyncio
async def test_concurrent_refresh_presents_single_use_token_exactly_once(tmp_path, monkeypatch):
    """Two providers on one token store: R1 is POSTed once, both end on the rotated pair."""
    from urllib.parse import parse_qs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://idp.example.com/oauth/token"
    a = _fenced_provider(tmp_path, monkeypatch, endpoint)
    b = _fenced_provider(tmp_path, monkeypatch, endpoint)
    assert a is not b
    await a.context.storage.set_tokens(_token("A1", "R1"))

    presented = []

    def responder(request):
        if request.method != "POST":
            return _fake_response(200, str(request.url), b"{}")
        refresh = parse_qs(request.content.decode())["refresh_token"][0]
        presented.append(refresh)
        if presented == ["R1"]:
            body = json.dumps(_token("A2", "R2").model_dump(mode="json", exclude_none=True)).encode()
            return _fake_response(200, endpoint, body)
        # A single-use provider rejects any second presentation.
        return _fake_response(400, endpoint, b'{"error":"invalid_grant"}')

    await asyncio.gather(_drive_flow(a, responder), _drive_flow(b, responder))

    assert presented == ["R1"], presented
    assert (a.context.current_tokens.access_token, a.context.current_tokens.refresh_token) == ("A2", "R2")
    assert (b.context.current_tokens.access_token, b.context.current_tokens.refresh_token) == ("A2", "R2")
    assert a._hermes_fence is None and b._hermes_fence is None


@pytest.mark.asyncio
async def test_refresh_fails_closed_while_a_peer_holds_the_fence(tmp_path, monkeypatch):
    """A fence held elsewhere past the deadline aborts the refresh: no POST, tokens kept."""
    import functools

    from tools import mcp_oauth

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://idp.example.com/oauth/token"
    provider = _fenced_provider(tmp_path, monkeypatch, endpoint)
    await provider.context.storage.set_tokens(_token("A1", "R1"))
    monkeypatch.setattr(
        mcp_oauth, "acquire_refresh_fence", functools.partial(mcp_oauth.acquire_refresh_fence, timeout=0.2)
    )

    sent = []

    peer_fd = await mcp_oauth.acquire_refresh_fence(provider.context.storage._tokens_path())
    try:
        with pytest.raises(mcp_oauth.RefreshFenceTimeout):
            await _drive_flow(provider, sent.append)
    finally:
        mcp_oauth.release_refresh_fence(peer_fd)

    assert sent == []
    assert provider.context.current_tokens.refresh_token == "R1"
    assert (await provider.context.storage.get_tokens()).refresh_token == "R1"
    assert provider._hermes_fence is None


@pytest.mark.asyncio
async def test_refresh_adopts_expired_peer_pair_and_posts_its_refresh_token(tmp_path, monkeypatch):
    """A peer rotated to (A2, R2) but A2 already expired: we must POST R2, never R1.

    The adopt path installs the rotated pair even without a live access
    token, because the POST we are about to build needs the new grant.
    """
    from urllib.parse import parse_qs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://idp.example.com/oauth/token"
    provider = _fenced_provider(tmp_path, monkeypatch, endpoint)
    await provider.context.storage.set_tokens(_token("A2", "R2", expires_in=0))

    presented = []

    def responder(request):
        if request.method != "POST":
            return _fake_response(200, str(request.url), b"{}")
        presented.append(parse_qs(request.content.decode())["refresh_token"][0])
        body = json.dumps(_token("A3", "R3").model_dump(mode="json", exclude_none=True)).encode()
        return _fake_response(200, endpoint, body)

    await _drive_flow(provider, responder)

    assert presented == ["R2"], presented
    assert provider.context.current_tokens.refresh_token == "R3"


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_pending", [False, True])
async def test_refresh_adopts_peer_pair_without_expiry_and_skips_the_post(tmp_path, monkeypatch, retry_pending):
    """A peer rotated to (A2, R2) with no ``expires_in`` (RFC 6749 optional): that pair is live.

    Treating a missing expiry as expired would POST R2 needlessly and burn a
    generation on a single-use provider.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://idp.example.com/oauth/token"
    provider = _fenced_provider(tmp_path, monkeypatch, endpoint)
    await provider.context.storage.set_tokens(_token("A2", "R2", expires_in=None))
    from tools import mcp_oauth, mcp_oauth_provider
    monkeypatch.setattr(mcp_oauth_provider, "monotonic", lambda: 100.0)
    provider._hermes_refresh_retry_at = 1000.0 if retry_pending else 0.0

    posted = []

    def responder(request):
        if request.method == "POST":
            posted.append(request)
        return _fake_response(200, str(request.url), b"{}")

    fd = (await mcp_oauth.acquire_refresh_fence(provider.context.storage._tokens_path())
          if retry_pending else None)
    try:
        await _drive_flow(provider, responder)
    finally:
        if fd is not None:
            mcp_oauth.release_refresh_fence(fd)

    assert posted == [], "a live peer pair must be adopted without presenting a refresh token"
    assert (provider.context.current_tokens.access_token, provider.context.current_tokens.refresh_token) == ("A2", "R2")


@pytest.mark.asyncio
async def test_refresh_restarts_flow_when_disk_pair_is_from_another_issuer(tmp_path, monkeypatch):
    """A disk pair bound to a different issuer loses its refresh token on adoption.

    With nothing left to refresh, _refresh_token must restart the SDK flow
    (401 -> full auth) instead of building a POST from the foreign grant or
    raising OAuthTokenError, and it must not keep the fence.
    """
    from tools.mcp_oauth_provider import _RefreshCompletedByPeer

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://idp.example.com/oauth/token"
    provider = _fenced_provider(tmp_path, monkeypatch, endpoint)
    storage = provider.context.storage
    storage.bind_issuer("https://other-idp.example.com")
    await storage.set_tokens(_token("A2", "R2"))

    with pytest.raises(_RefreshCompletedByPeer):
        await provider._refresh_token()

    assert not provider.context.current_tokens.refresh_token, "foreign refresh token must be stripped"
    assert (await storage.get_tokens()).refresh_token is None, "strip must reach disk"
    assert provider._hermes_fence is None


@pytest.mark.asyncio
async def test_refresh_fence_surfaces_non_contention_lock_errors_immediately(tmp_path, monkeypatch):
    """A lock syscall failing for a reason other than contention must not spin to the deadline."""
    import errno

    from tools import mcp_oauth

    if mcp_oauth.fcntl is None:
        pytest.skip("flock-based fence only")

    def broken_flock(fd, op):
        if op & mcp_oauth.fcntl.LOCK_UN:
            return
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(mcp_oauth.fcntl, "flock", broken_flock)
    started = time.monotonic()
    with pytest.raises(mcp_oauth.RefreshFenceTimeout):
        await mcp_oauth.acquire_refresh_fence(tmp_path / "srv.json", timeout=5.0)
    assert time.monotonic() - started < 1.0, "must fail fast, not wait out the deadline"


@pytest.mark.asyncio
async def test_refresh_400_recovery_rejects_disk_pair_from_another_issuer(tmp_path, monkeypatch):
    """A 400 must not be "recovered" with a disk pair bound to a different issuer.

    The enforcer strips that pair's refresh token on install; a stripped pair
    is not a recovery, so the session is cleared as on any dead grant and the
    foreign refresh token never survives on disk.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://idp.example.com/oauth/token"
    provider = _fenced_provider(tmp_path, monkeypatch, endpoint)
    storage = provider.context.storage
    storage.bind_issuer("https://other-idp.example")
    await storage.set_tokens(_token("A2", "R2"))

    recovered = await provider._handle_refresh_response(
        _fake_response(400, endpoint, b'{"error":"invalid_grant"}')
    )

    assert recovered is False
    assert provider.context.current_tokens is None
    assert (await storage.get_tokens()).refresh_token is None, "foreign refresh token must not survive on disk"
