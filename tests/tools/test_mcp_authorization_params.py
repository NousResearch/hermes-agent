"""``oauth.authorization_params``: extra query parameters on the browser authorization URL.

Some servers need an authorization-URL parameter that MCP discovery does not advertise: ArtifactBridge
asks for an agent grant with ``agent=1`` (a server or cron install connects as an agent of the person,
who confirms it on the consent screen), Dropbox issues a refresh token only with
``token_access_type=offline``. The SDK builds the URL itself, so the parameters are added in the
redirect handler, never overriding a key the SDK already set.
"""
from __future__ import annotations

from urllib.parse import parse_qsl, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

from tests.tools.test_mcp_google_offline_access import GOOGLE, OTHER_AS, RESOURCE, _standin


async def _run(tmp_path, monkeypatch, *, issuer, authorization_params=None):
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage, _authorization_code_result
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    seen = {}

    async def redirect(url):
        seen["authorize_url"] = url
        seen["state"] = dict(parse_qsl(urlsplit(url).query))["state"]

    async def callback():
        return _authorization_code_result("code-1", seen["state"])

    metadata = OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:1/cb")], client_name="Hermes Agent")
    kwargs = {} if authorization_params is None else {"authorization_params": authorization_params}
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv", server_url=RESOURCE, storage=HermesTokenStorage("srv"), client_metadata=metadata,
        redirect_handler=redirect, callback_handler=callback, **kwargs)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(_standin(
            httpx, issuer=issuer, authorization_servers=[issuer]))) as client:
        response = await client.get(RESOURCE)
    seen["provider"] = provider
    return response, parse_qsl(urlsplit(seen["authorize_url"]).query), seen


@pytest.mark.asyncio
async def test_configured_params_reach_the_authorization_url_once(tmp_path, monkeypatch):
    response, pairs, seen = await _run(tmp_path, monkeypatch, issuer=OTHER_AS, authorization_params={"agent": "1"})
    assert response.status_code == 200
    assert [v for k, v in pairs if k == "agent"] == ["1"]
    # A non-Google issuer gets no Google parameters because of the configured ones.
    assert not any(k in ("access_type", "prompt") for k, _ in pairs)
    # The exchange still completed and the tokens were saved.
    assert (tmp_path / "mcp-tokens" / "srv.json").exists()
    # Wraps once across re-authorization.
    provider = seen["provider"]
    wrapped = provider.context.redirect_handler
    provider._extend_authorization_url()
    assert provider.context.redirect_handler is wrapped


@pytest.mark.asyncio
async def test_configured_params_never_override_sdk_keys(tmp_path, monkeypatch):
    hostile = {"client_id": "evil", "redirect_uri": "https://evil.example/cb", "state": "x",
               "code_challenge": "x", "response_type": "token", "agent": "1"}
    response, pairs, _ = await _run(tmp_path, monkeypatch, issuer=OTHER_AS, authorization_params=hostile)
    assert response.status_code == 200
    query = dict(pairs)
    assert query["client_id"] != "evil"
    assert query["redirect_uri"] == "http://127.0.0.1:1/cb"
    assert query["state"] != "x" and query["code_challenge"] != "x"
    assert query["response_type"] == "code"
    assert query["agent"] == "1"
    assert len(pairs) == len(query), "no key may appear twice"


@pytest.mark.asyncio
async def test_without_config_the_url_is_unchanged(tmp_path, monkeypatch):
    response, pairs, _ = await _run(tmp_path, monkeypatch, issuer=OTHER_AS)
    assert response.status_code == 200
    assert "agent" not in dict(pairs)
    response, pairs, _ = await _run(tmp_path / "empty", monkeypatch, issuer=OTHER_AS, authorization_params={})
    assert "agent" not in dict(pairs)


@pytest.mark.asyncio
async def test_google_offline_access_still_applies_beside_configured_params(tmp_path, monkeypatch):
    response, pairs, _ = await _run(tmp_path, monkeypatch, issuer=GOOGLE,
                                    authorization_params={"login_hint": "a@example.com", "access_type": "online"})
    assert response.status_code == 200
    query = dict(pairs)
    # Google's own handling wins over a configured value for the same key.
    assert query["access_type"] == "offline" and query["prompt"] == "consent"
    assert query["login_hint"] == "a@example.com"


def test_parse_authorization_params():
    from tools.mcp_oauth_provider import authorization_params

    assert authorization_params({}) == {}
    assert authorization_params({"authorization_params": None}) == {}
    assert authorization_params({"authorization_params": {"agent": "1"}}) == {"agent": "1"}
    assert authorization_params({"authorization_params": {"agent": True}}) == {"agent": "1"}
    assert authorization_params({"authorization_params": {"agent": False}}) == {"agent": "0"}
    assert authorization_params({"authorization_params": {"n": 2}}) == {"n": "2"}
    # Invalid blocks are dropped whole, never sent half-parsed.
    assert authorization_params({"authorization_params": "agent=1"}) == {}
    assert authorization_params({"authorization_params": {"agent": ["1"]}}) == {}
    assert authorization_params({"authorization_params": {"": "1", "agent": "1"}}) == {}


def test_build_provider_kwargs_passes_authorization_params(tmp_path, monkeypatch):
    from tools.mcp_oauth_provider import build_provider_kwargs, prepare_oauth_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg, storage = prepare_oauth_config("srv", RESOURCE, {"authorization_params": {"agent": "1"}, "redirect_port": 0})
    kwargs = build_provider_kwargs(cfg, storage, ssh_proxy_hint=False)
    assert kwargs["authorization_params"] == {"agent": "1"}
    cfg, storage = prepare_oauth_config("srv2", RESOURCE, {"redirect_port": 0})
    assert build_provider_kwargs(cfg, storage, ssh_proxy_hint=False)["authorization_params"] == {}
