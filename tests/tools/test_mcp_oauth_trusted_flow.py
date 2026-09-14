"""Issuer compatibility through the unmodified SDK state machine and HTTP transport."""
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2")


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [None, "discovery", "issuer", "trust", "trusted_alias"])
@pytest.mark.parametrize("invalid_oauth_candidate", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("netsuite", [False, True])
async def test_registered_client_binding_survives_only_same_trusted_relation(
    tmp_path, monkeypatch, changed, legacy, netsuite, invalid_oauth_candidate,
):
    import httpx2 as httpx
    from mcp.client.auth import OAuthFlowError
    from mcp.shared.auth import AuthorizationCodeResult, OAuthClientMetadata
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import HermesMCPOAuthProvider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    server = "https://resource.example/mcp"
    discovery = "https://tenant.example/"
    issuer = "https://issuer.example"
    if netsuite:
        from tools.mcp_oauth import apply_oauth_provider_defaults
        server = "https://123456-sb1.suitetalk.api.netsuite.com/services/mcp/v1/suiteapp/all"
        discovery = "https://123456-sb1.suitetalk.api.netsuite.com/"
        issuer = apply_oauth_provider_defaults({}, server_url=server)["trusted_issuers"][0]
    original_issuer = issuer
    registered = []
    authorized = []
    state = None
    storage = HermesTokenStorage("relation")

    async def redirect(url):
        nonlocal state
        query = parse_qs(urlsplit(url).query)
        authorized.append(query["client_id"][0])
        state = query["state"][0]

    async def callback():
        return AuthorizationCodeResult(code="test-code", state=state)

    def transport(request):
        url = str(request.url)
        if url == server:
            if request.headers.get("Authorization") == f"Bearer token-{len(authorized)}" and authorized:
                return httpx.Response(200)
            return httpx.Response(401)
        if "oauth-protected-resource" in url:
            return httpx.Response(200, json={"resource": server, "authorization_servers": [discovery]})
        if "oauth-authorization-server" in url and invalid_oauth_candidate:
            return httpx.Response(200, json={"issuer": issuer})
        if "oauth-authorization-server" in url or "openid-configuration" in url:
            return httpx.Response(200, json={
                "issuer": issuer, "authorization_endpoint": "https://issuer.example/authorize",
                "token_endpoint": "https://issuer.example/token",
                "registration_endpoint": "https://issuer.example/register",
                "response_types_supported": ["code"],
            })
        if url.endswith("/register"):
            registered.append(f"client-{len(registered)}")
            return httpx.Response(201, json={"client_id": registered[-1], "token_endpoint_auth_method": "none"})
        if url.endswith("/token"):
            return httpx.Response(200, json={"access_token": f"token-{len(authorized)}", "token_type": "Bearer"})
        raise AssertionError(f"Unexpected HTTP request: {request.method} {url}")

    def provider(trust):
        if legacy:
            from mcp.client.auth.oauth2 import OAuthClientProvider
            from tools.mcp_oauth_provider import HermesProviderMixin
            cls = type("LegacyProvider", (HermesProviderMixin, OAuthClientProvider), {})
            return cls(
                server_url=server, storage=storage,
                client_metadata=OAuthClientMetadata(redirect_uris=["http://127.0.0.1:12345/callback"]),
                redirect_handler=redirect, callback_handler=callback, trusted_issuers=trust,
            )
        return HermesMCPOAuthProvider(
            server_name="relation", server_url=server, storage=storage,
            client_metadata=OAuthClientMetadata(redirect_uris=["http://127.0.0.1:12345/callback"]),
            redirect_handler=redirect, callback_handler=callback, trusted_issuers=trust,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        assert (await client.get(server, auth=provider((issuer, "https://second-trusted.example")))).status_code == 200
        original = await storage.get_client_info()
        # Force a new process-equivalent auth cycle, keeping the actual persisted registration.
        storage._tokens_path().unlink()
        if changed == "discovery":
            discovery = "https://other-tenant.example/"
        elif changed == "issuer":
            issuer = "https://unlisted.example"
        elif changed == "trusted_alias":
            issuer = "https://second-trusted.example"
        trust = () if changed == "trust" else (original_issuer, "https://second-trusted.example")
        if changed in ("issuer", "trust", "trusted_alias"):
            with pytest.raises(OAuthFlowError, match="issuer mismatch"):
                await client.get(server, auth=provider(trust))
            assert registered == [original.client_id]
            assert authorized == [original.client_id]
        else:
            assert (await client.get(server, auth=provider(trust))).status_code == 200
            if changed == "discovery":
                assert authorized[-1] != original.client_id
            else:
                assert authorized == [original.client_id, original.client_id]
                assert registered == [original.client_id]
                assert (await storage.get_client_info()).client_id == original.client_id


@pytest.mark.asyncio
@pytest.mark.parametrize("trusted", [True, False])
async def test_cold_missing_metadata_validates_before_persist_or_refresh(tmp_path, monkeypatch, trusted):
    import httpx2 as httpx
    from mcp.client.auth import OAuthFlowError
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import HermesMCPOAuthProvider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    server = "https://resource.example/mcp"
    discovery = "https://tenant.example/"
    issuer = "https://issuer.example"
    storage = HermesTokenStorage("cold")
    await storage.set_client_info(OAuthClientInformationFull(client_id="existing", issuer=discovery))
    await storage.set_tokens(OAuthToken(access_token="expired", token_type="Bearer", expires_in=0, refresh_token="refresh"))
    posts = []

    def transport(request):
        url = str(request.url)
        if "oauth-protected-resource" in url:
            return httpx.Response(200, json={"resource": server, "authorization_servers": [discovery]})
        if "oauth-authorization-server" in url:
            return httpx.Response(200, json={"issuer": issuer, "authorization_endpoint": "https://issuer.example/authorize", "token_endpoint": "https://issuer.example/token", "response_types_supported": ["code"]})
        if request.method == "POST":
            posts.append(url)
            return httpx.Response(200, json={"access_token": "fresh", "token_type": "Bearer"})
        if url == server:
            return httpx.Response(200)
        raise AssertionError(url)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(transport), **kwargs))
    provider = HermesMCPOAuthProvider(
        server_name="cold", server_url=server, storage=storage,
        client_metadata=OAuthClientMetadata(redirect_uris=None), trusted_issuers=(issuer,) if trusted else (),
    )
    async with httpx.AsyncClient() as client:
        if trusted:
            assert (await client.get(server, auth=provider)).status_code == 200
            assert posts == ["https://issuer.example/token"]
            assert str(storage.load_oauth_metadata().issuer) == issuer
        else:
            with pytest.raises(OAuthFlowError, match="issuer mismatch"):
                await client.get(server, auth=provider)
            assert posts == []
            assert provider.context.oauth_metadata is None
            assert storage.load_oauth_metadata() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_authority", [False, True])
@pytest.mark.parametrize("asm_failure", ["missing", "invalid", "unavailable"])
async def test_cold_refresh_requires_verified_bound_endpoint(
    tmp_path, monkeypatch, changed_authority, asm_failure,
):
    import httpx2 as httpx
    from mcp.client.auth import OAuthFlowError
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import HermesMCPOAuthProvider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    server = "https://resource.example/mcp"
    old = "https://old-as.example/"
    discovery = "https://new-as.example/" if changed_authority else old
    storage = HermesTokenStorage("unverified")
    await storage.set_client_info(OAuthClientInformationFull(
        client_id="old-client", client_secret="old-secret", issuer=old,
    ))
    await storage.set_tokens(OAuthToken(
        access_token="expired", token_type="Bearer", expires_in=0, refresh_token="old-refresh",
    ))
    requests = []

    def transport(request):
        requests.append(request)
        url = str(request.url)
        if "oauth-protected-resource" in url:
            return httpx.Response(200, json={"resource": server, "authorization_servers": [discovery]})
        if request.method == "POST":
            return httpx.Response(200, json={"access_token": "fresh", "token_type": "Bearer"})
        if url == server:
            return httpx.Response(200)
        if asm_failure == "unavailable":
            raise httpx.ConnectError("offline", request=request)
        if asm_failure == "invalid":
            return httpx.Response(200, json={"issuer": discovery})
        return httpx.Response(404)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(transport), **kwargs,
    ))
    provider = HermesMCPOAuthProvider(
        server_name="unverified", server_url=server, storage=storage,
        client_metadata=OAuthClientMetadata(redirect_uris=None),
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(OAuthFlowError, match="refusing cached credentials|verified token endpoint"):
            await client.get(server, auth=provider)
    assert all(request.method != "POST" for request in requests)
    assert provider.context.oauth_metadata is None
    assert storage.load_oauth_metadata() is None
    if changed_authority:
        assert all("oauth-protected-resource" in str(request.url) for request in requests)
