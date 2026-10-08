"""Regression tests for ``oauth.trust_prm_resource`` (#135227).

A multi-tenant MCP serves each tenant from ``https://<tenant>.example/mcp`` but publishes
protected-resource metadata naming a canonical application resource
(``https://app.example/mcp``). The SDK's ``_validate_resource_match`` rejects that PRM
before authorization even starts, while the tenant's authorization server rejects the
canonical URL as the RFC 8707 ``resource`` parameter (``invalid_target``) and demands the
tenant one — no configured URL satisfies both sides without the per-server opt-in.

Fail-closed contract under test:
- Without the flag, a mismatched PRM stays a hard ``OAuthFlowError`` (default unchanged).
- With the flag, the metadata is accepted — and the RFC 8707 ``resource`` parameter still
  derives from the *configured* server URL, so the token audience never silently follows
  the advertised canonical resource.
- A matching PRM is accepted with or without the flag (no regression on the normal path).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")

from mcp.client.auth.exceptions import OAuthFlowError
from mcp.shared.auth import OAuthClientMetadata, ProtectedResourceMetadata
from pydantic import AnyUrl

from tools.mcp_oauth import HermesTokenStorage, humanize_oauth_registration_error
from tools.mcp_oauth_manager import HermesMCPOAuthProvider, reset_manager_for_tests

SERVER_URL = "https://tenant.example/mcp"
# The shape from #135227: canonical application resource on a different host.
CANONICAL_PRM = ProtectedResourceMetadata.model_validate({
    "resource": "https://app.example/mcp",
    "authorization_servers": ["https://app.example"],
    "scopes_supported": ["mcp"],
})
MATCHING_PRM = ProtectedResourceMetadata.model_validate({
    "resource": "https://tenant.example/mcp",
    "authorization_servers": ["https://tenant.example"],
})


def _provider(tmp_path, monkeypatch, *, trust: bool) -> HermesMCPOAuthProvider:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    return HermesMCPOAuthProvider(
        server_name="srv",
        server_url=SERVER_URL,
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            client_name="Hermes Agent",
        ),
        storage=HermesTokenStorage("srv"),
        trust_prm_resource=trust,
    )


@pytest.mark.asyncio
async def test_default_rejects_mismatched_prm(tmp_path, monkeypatch):
    """Fail-closed default: without the opt-in the canonical-resource PRM is refused."""
    provider = _provider(tmp_path, monkeypatch, trust=False)
    with pytest.raises(OAuthFlowError, match="does not match expected"):
        await provider._validate_resource_match(CANONICAL_PRM)


@pytest.mark.asyncio
async def test_opt_in_accepts_mismatched_prm(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch, trust=True)
    await provider._validate_resource_match(CANONICAL_PRM)


@pytest.mark.asyncio
async def test_opt_in_keeps_configured_url_as_resource_parameter(tmp_path, monkeypatch):
    """Load-bearing invariant: opting in accepts the metadata, but the RFC 8707 ``resource``
    parameter still derives from the configured server URL — the SDK only swaps in the
    advertised resource when it is a true parent of the configured one."""
    provider = _provider(tmp_path, monkeypatch, trust=True)
    provider.context.protected_resource_metadata = CANONICAL_PRM
    assert str(provider.context.get_resource_url()) == SERVER_URL


@pytest.mark.asyncio
async def test_matching_prm_accepted_with_and_without_opt_in(tmp_path, monkeypatch):
    for trust in (False, True):
        provider = _provider(tmp_path, monkeypatch, trust=trust)
        await provider._validate_resource_match(MATCHING_PRM)


def test_build_provider_kwargs_reads_the_flag(monkeypatch, tmp_path):
    from tools import mcp_oauth as mo
    from tools.mcp_oauth_provider import build_provider_kwargs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mo._configure_callback_port = lambda cfg, storage: (
        cfg.__setitem__("_resolved_port", 12345),
        12345,
    )[1]
    mo._maybe_preregister_client = lambda *a, **k: None
    kwargs = build_provider_kwargs(
        {"trust_prm_resource": True}, HermesTokenStorage("srv"), ssh_proxy_hint=False
    )
    assert kwargs["trust_prm_resource"] is True
    kwargs = build_provider_kwargs({}, HermesTokenStorage("srv"), ssh_proxy_hint=False)
    assert kwargs["trust_prm_resource"] is False


def test_humanize_error_names_the_opt_in():
    msg = (
        "Protected resource https://app.example/mcp does not match expected "
        "https://tenant.example/mcp"
    )
    humanized = humanize_oauth_registration_error("srv", OAuthFlowError(msg))
    assert humanized is not None and "trust_prm_resource" in humanized
