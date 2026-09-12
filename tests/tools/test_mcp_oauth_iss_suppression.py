"""Regression tests for RFC 9207 ``iss`` suppression on advertise-but-omit issuers.

Figma (``issuer: https://api.figma.com``) and Cloudflare (``issuer:
https://mcp.cloudflare.com``) advertise
``authorization_response_iss_parameter_supported: true`` in their authorization
server metadata but never send ``iss`` in the authorization callback. The MCP
SDK 2.0.0 validator (``mcp/client/auth/oauth2.py:425``) correctly rejects that
per RFC 9207 — which makes the connector dead on arrival through no fault of
the client (#99984 for Cloudflare; same class for Figma).

``HermesMCPOAuthProvider.async_auth_flow`` suppresses the advertised flag for
these known-broken issuers around each iteration of the generator bridge
(metadata can be discovered lazily mid-flow, so a one-shot suppression at flow
start would miss first logins) and restores it afterwards so persisted metadata
stays accurate.

These tests drive the REAL SDK flow (no mocks of the SDK's own logic): 401 →
PRM discovery → ASM discovery → authorization branch, with ``iss=None`` in the
callback — the exact sequence a ``hermes mcp login figma`` hits.
"""
from __future__ import annotations

import asyncio

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")

_FIGMA_ISSUER = "https://api.figma.com"


async def _build_provider(tmp_path, monkeypatch, *, callback_iss):
    """Provider wired with stub redirect/callback handlers, mirroring the
    bidirectional-bridge tests' harness (real storage, real SDK provider)."""
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    from mcp.shared.auth import AuthorizationCodeResult, OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None, "SDK OAuth types must be available"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    # Expired token + client info: forces the 401 → full-authorization branch
    # (skips registration; the callback supplies the authorization code).
    await storage.set_tokens(
        OAuthToken(access_token="dead", token_type="Bearer", expires_in=-3600, refresh_token="dead_refresh")
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )
    client_metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        client_name="Hermes Agent",
    )

    async def _noop_redirect(url: str) -> None:
        pass

    _captured_state = {"state": None}

    async def _callback() -> AuthorizationCodeResult:
        # The RFC 9207 violation the real Figma/Cloudflare servers commit: no iss.
        # state echoes what the SDK generated only if the flow asked for one; the
        # SDK compares result.state against its own `state` — return None so the
        # comparison path is skipped deterministically via compare_digest(None…)
        # raising early is NOT acceptable; instead we re-read the generated state
        # from the request the redirect handler saw.
        return AuthorizationCodeResult(code="auth_code_123", state=_captured_state["state"], iss=callback_iss)

    async def _redirect_capture_state(url: str) -> None:
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(url).query)
        _captured_state["state"] = qs.get("state", [None])[0]

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=_redirect_capture_state,
        callback_handler=_callback,
    )
    return provider, httpx


async def _drive_to_authorization_branch(provider, httpx, *, issuer: str, iss_advertised: bool):
    """Drive refresh-fail → 401 → PRM discovery → ASM discovery; return the flow
    positioned right after the ASM response was consumed (inside the authorization
    branch — the next asend return value is the token-exchange request)."""
    req = httpx.Request("POST", "https://example.com/mcp")
    flow = provider.async_auth_flow(req)

    # Expired token + refresh_token: the SDK yields a refresh POST first.
    refresh_req = await flow.__anext__()
    assert refresh_req is not None
    # Refresh fails (401) → SDK re-initializes and yields the MCP request itself.
    mcp_req = await flow.asend(httpx.Response(401, request=refresh_req))
    assert mcp_req is not None
    # MCP request gets a 401 with WWW-Authenticate → OAuth branch: PRM discovery GET.
    prm_req = await flow.asend(httpx.Response(401, request=mcp_req, headers={
        "WWW-Authenticate": 'Bearer resource_metadata="https://example.com/.well-known/oauth-protected-resource"',
    }))
    # PRM points at the issuer under test as the authorization server (mirrors
    # reality: mcp.figma.com's PRM lists api.figma.com, whose ASM issuer matches).
    prm_resp = httpx.Response(
        200, request=prm_req,
        json={"resource": "https://example.com", "authorization_servers": [issuer]},
    )
    asm_req = await flow.asend(prm_resp)
    asm_resp = httpx.Response(200, request=asm_req, json={
        "issuer": issuer,
        "authorization_endpoint": "https://example.com/authorize",
        "token_endpoint": "https://example.com/token",
        "authorization_response_iss_parameter_supported": iss_advertised,
    })
    # Consuming the ASM response runs the authorization branch (redirect →
    # callback with no iss → validate) and yields the token-exchange POST.
    return flow, asm_resp


@pytest.mark.asyncio
async def test_figma_style_issuer_login_survives_without_iss(tmp_path, monkeypatch):
    """Figma advertises iss support but omits iss — login must complete instead of
    dying with 'Authorization response missing iss parameter advertised by the
    authorization server'."""
    provider, httpx = await _build_provider(tmp_path, monkeypatch, callback_iss=None)
    flow, asm_resp = await _drive_to_authorization_branch(
        provider, httpx, issuer=_FIGMA_ISSUER, iss_advertised=True)

    # Unsuppressed, the SDK raises OAuthFlowError here (callback had no iss).
    # Suppressed, the flow proceeds and yields the token-exchange POST.
    token_req = await asyncio.wait_for(flow.asend(asm_resp), timeout=5)
    assert token_req is not None and token_req.url.host == "example.com"
    # Persisted metadata must keep the honest advertised flag.
    md = provider.context.oauth_metadata
    assert md is not None and md.authorization_response_iss_parameter_supported is True


@pytest.mark.asyncio
async def test_prefix_spoofed_issuer_is_not_suppressed(tmp_path, monkeypatch):
    """An attacker-controlled issuer sharing the prefix (e.g.
    https://api.figma.com.attacker.example) must NOT be suppressed — matching is
    exact scheme+host, so only the genuinely broken issuers skip the iss check."""
    from mcp.client.auth.exceptions import OAuthFlowError

    provider, httpx = await _build_provider(tmp_path, monkeypatch, callback_iss=None)
    flow, asm_resp = await _drive_to_authorization_branch(
        provider, httpx, issuer="https://api.figma.com.attacker.example", iss_advertised=True)
    with pytest.raises(OAuthFlowError):
        await asyncio.wait_for(flow.asend(asm_resp), timeout=5)


@pytest.mark.asyncio
async def test_unknown_issuer_advertising_iss_still_fails_closed(tmp_path, monkeypatch):
    """An issuer NOT in the known-broken set that advertises iss support but omits
    it must still fail closed (SDK contract preserved for honest servers)."""
    from mcp.client.auth.exceptions import OAuthFlowError

    provider, httpx = await _build_provider(tmp_path, monkeypatch, callback_iss=None)
    flow, asm_resp = await _drive_to_authorization_branch(
        provider, httpx, issuer="https://honest.example.com", iss_advertised=True)
    with pytest.raises(OAuthFlowError):
        await asyncio.wait_for(flow.asend(asm_resp), timeout=5)
