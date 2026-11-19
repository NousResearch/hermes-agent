"""Exercise the native SDK login and publication races without real credentials."""

from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

from mcp.shared.auth import AuthorizationCodeResult, OAuthClientMetadata, OAuthMetadata, OAuthToken
from tools.mcp_oauth import HermesTokenStorage
from tools.mcp_oauth_login import _promote, _read_state
from tools.mcp_oauth_manager import HermesMCPOAuthProvider


RESOURCE = "https://resource.example/mcp"
ISSUER = "https://issuer.example"


def metadata(issuer=ISSUER):
    return OAuthMetadata(issuer=issuer, authorization_endpoint=issuer + "/authorize",
                         token_endpoint=issuer + "/token", registration_endpoint=issuer + "/register",
                         response_types_supported=["code"], grant_types_supported=["authorization_code"],
                         code_challenge_methods_supported=["S256"])


def provider(home):
    storage = HermesTokenStorage("orbit", hermes_home=home)
    redirects = []

    async def redirect(url):
        redirects.append(url)

    async def callback():
        state = parse_qs(urlsplit(redirects[-1]).query)["state"][0]
        return AuthorizationCodeResult(code="fixture-code", state=state, iss=ISSUER)

    client = OAuthClientMetadata(redirect_uris=["http://localhost:8765/callback"],
                                grant_types=["authorization_code"], response_types=["code"],
                                token_endpoint_auth_method="none")
    result = HermesMCPOAuthProvider(server_url=RESOURCE, client_metadata=client, storage=storage,
                                   redirect_handler=redirect, callback_handler=callback, server_name="orbit")
    return result, storage, redirects


def response(request):
    import httpx2

    url = str(request.url)
    if request.method == "POST" and url == ISSUER + "/register":
        return httpx2.Response(201, request=request, json={
            "client_id": "fixture-client", "redirect_uris": ["http://localhost:8765/callback"],
            "grant_types": ["authorization_code"], "response_types": ["code"],
            "token_endpoint_auth_method": "none"})
    if request.method == "POST" and url == ISSUER + "/token":
        return httpx2.Response(200, request=request, json={
            "access_token": "fixture-access", "refresh_token": "fixture-refresh", "token_type": "Bearer",
            "expires_in": 3600})
    if "oauth-protected-resource" in url:
        return httpx2.Response(200, request=request, json={"resource": RESOURCE, "authorization_servers": [ISSUER]})
    if "oauth-authorization-server" in url:
        return httpx2.Response(200, request=request, json=metadata().model_dump(mode="json", exclude_none=True))
    assert url == RESOURCE
    return httpx2.Response(200 if "authorization" in request.headers else 401, request=request)


async def replacement(home, storage, token_only=False):
    from mcp.shared.auth import OAuthClientInformationFull

    candidate = HermesTokenStorage("orbit", hermes_home=home)
    if token_only:
        await candidate.set_client_info(await storage.get_client_info())
        candidate.save_oauth_metadata(storage.load_oauth_metadata() or metadata())
    else:
        await candidate.set_client_info(OAuthClientInformationFull(client_id="replacement-client"))
        candidate.save_oauth_metadata(metadata("https://replacement.example"))
    await candidate.set_tokens(OAuthToken(access_token="replacement-access", refresh_token="replacement-refresh",
                                         token_type="Bearer", expires_in=3600))
    await _promote(storage, candidate, _read_state(storage))
    return _read_state(storage)


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_metadata", [False, True])
async def test_native_discovery_registration_callback_token_and_final_persistence(tmp_path, cached_metadata):
    import httpx2

    auth, storage, redirects = provider(tmp_path)
    if cached_metadata:
        storage.save_oauth_metadata(metadata("https://old.example"))
    flow = auth.async_auth_flow(httpx2.Request("POST", RESOURCE))
    request = await anext(flow)
    methods = []
    try:
        while True:
            methods.append((request.method, str(request.url)))
            request = await flow.asend(response(request))
    except StopAsyncIteration:
        pass
    finally:
        await flow.aclose()
    assert ("POST", ISSUER + "/register") in methods
    assert ("POST", ISSUER + "/token") in methods
    assert len(redirects) == 1
    assert (await storage.get_tokens()).access_token == "fixture-access"
    assert (await storage.get_client_info()).client_id == "fixture-client"
    assert storage.load_oauth_metadata() == metadata()


@pytest.mark.asyncio
async def test_identical_registration_does_not_admit_a_late_token_or_invalid_client_write(tmp_path):
    import httpx2

    auth, storage, _ = provider(tmp_path / "live")
    flow = auth.async_auth_flow(httpx2.Request("POST", RESOURCE))
    request = await anext(flow)
    try:
        while str(request.url) != ISSUER + "/token":
            request = await flow.asend(response(request))
        await auth._persist_oauth_metadata_if_changed()
        before = await replacement(tmp_path / "candidate", storage, token_only=True)
        with pytest.raises(RuntimeError, match="grant changed"):
            await auth._invalidate_rejected_registration(storage)
        with pytest.raises(RuntimeError, match="grant changed"):
            await flow.asend(response(request))
        assert _read_state(storage) == before
    finally:
        await flow.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_endpoint", ["register", "token"])
async def test_late_native_response_cannot_overwrite_published_grant(tmp_path, pause_endpoint):
    import httpx2

    auth, storage, _ = provider(tmp_path / "live")
    flow = auth.async_auth_flow(httpx2.Request("POST", RESOURCE))
    request = await anext(flow)
    try:
        while str(request.url) != ISSUER + "/" + pause_endpoint:
            request = await flow.asend(response(request))
        before = await replacement(tmp_path / "candidate", storage)
        with pytest.raises(RuntimeError, match="grant changed"):
            await flow.asend(response(request))
        assert _read_state(storage) == before
    finally:
        await flow.aclose()
