"""OAuth edge cases measured against live servers on 2026-10-26 (Semgrep MCP).

Two independent defects parked a catalog MCP that had never been authorized end to end:

1. ``hermes mcp login semgrep --flow device`` → ``Client registration failed (HTTP 400)``: the
   device flow posted ``response_types: []`` to the Dynamic Client Registration endpoint.
2. ``--flow browser`` → ``Authorization server metadata issuer mismatch: https://login.semgrep.dev
   != https://login.semgrep.dev/``: the protected-resource document advertises
   ``authorization_servers: ["https://login.semgrep.dev/"]`` and pydantic's ``str(AnyHttpUrl)``
   keeps the root slash, while the authorization server's own document publishes the slashless
   ``issuer``. RFC 8414 §3.3 compares exactly, so the SDK rejected the pair.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from tools import mcp_oauth_device as device
from tools.mcp_oauth_provider import HermesProviderMixin, normalize_advertised_auth_server_url


@pytest.mark.parametrize(
    "advertised, expected",
    [
        # pydantic's rendering of a host-only URL keeps a root slash; the AS document does not.
        ("https://login.semgrep.dev/", "https://login.semgrep.dev"),
        ("https://login.semgrep.dev", "https://login.semgrep.dev"),
        # A path-scoped authorization server must keep its path (Strava-style, #116233).
        ("https://auth.example.com/tenant/", "https://auth.example.com/tenant/"),
        ("https://auth.example.com/tenant", "https://auth.example.com/tenant"),
        # Anything carrying userinfo, a query or a fragment stays as advertised: the exact-match
        # rule keeps rejecting unusual identifiers.
        ("https://auth.example.com/?a=1", "https://auth.example.com/?a=1"),
        ("https://auth.example.com/#f", "https://auth.example.com/#f"),
        ("https://user@auth.example.com/", "https://user@auth.example.com/"),
        (None, None),
        ("", ""),
    ],
)
def test_normalize_advertised_auth_server_url(advertised, expected):
    assert normalize_advertised_auth_server_url(advertised) == expected


class _FakeResponse:
    status_code = 201

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self.posts: list[tuple[str, dict]] = []
        self._payload = payload

    async def post(self, url, json=None):  # mirrors httpx.AsyncClient.post
        self.posts.append((str(url), json))
        return _FakeResponse(self._payload)


class _FakeContext:
    def __init__(self, client_metadata, registration_endpoint, auth_server_url):
        self.client_metadata = client_metadata
        self.oauth_metadata = types.SimpleNamespace(registration_endpoint=registration_endpoint,
                                                    issuer=auth_server_url)
        self.auth_server_url = auth_server_url
        self.client_info = None


class _FakeProvider:
    def __init__(self, context):
        self.context = context

    def _coerce_client_secret_post(self):
        return None


def _registration_payload():
    return {
        "client_id": "client_01TEST",
        "redirect_uris": ["http://127.0.0.1:41917/callback"],
        "grant_types": ["authorization_code", "refresh_token", "urn:ietf:params:oauth:grant-type:device_code"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }


def test_device_registration_keeps_response_types(monkeypatch):
    """DCR must not send an empty ``response_types`` (Semgrep rejects it with HTTP 400)."""
    from mcp.shared.auth import OAuthClientMetadata

    client_metadata = OAuthClientMetadata.model_validate({
        "client_name": "Hermes Agent",
        "redirect_uris": ["http://127.0.0.1:41917/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    })
    context = _FakeContext(client_metadata, "https://login.semgrep.dev/oauth2/register",
                           "https://login.semgrep.dev")
    provider = _FakeProvider(context)
    client = _FakeClient(_registration_payload())

    asyncio.run(device._register(client, provider, {}))

    assert len(client.posts) == 1
    url, body = client.posts[0]
    assert url == "https://login.semgrep.dev/oauth2/register"
    assert body["response_types"] == ["code"], "empty response_types is rejected by strict DCR endpoints"
    assert body["grant_types"] == [
        "authorization_code",
        "refresh_token",
        "urn:ietf:params:oauth:grant-type:device_code",
    ]
    assert body["redirect_uris"] == ["http://127.0.0.1:41917/callback"]


def test_device_metadata_accepts_slashless_issuer(monkeypatch):
    """The device metadata check must treat ``https://host/`` and ``https://host`` as one issuer."""
    from mcp.client.auth.exceptions import OAuthFlowError

    class _Resp:
        status_code = 200
        url = "https://login.semgrep.dev/.well-known/oauth-authorization-server"

        def json(self):
            return {
                "issuer": "https://login.semgrep.dev",
                "authorization_endpoint": "https://login.semgrep.dev/oauth2/authorize",
                "token_endpoint": "https://login.semgrep.dev/oauth2/token",
                "registration_endpoint": "https://login.semgrep.dev/oauth2/register",
                "device_authorization_endpoint": "https://login.semgrep.dev/oauth2/device_authorization",
                "grant_types_supported": ["authorization_code", "refresh_token",
                                          "urn:ietf:params:oauth:grant-type:device_code"],
            }

    class _Client:
        async def get(self, url):
            return _Resp()

    metadata = asyncio.run(
        device._device_metadata(_Client(), "https://mcp.semgrep.ai/mcp", "https://login.semgrep.dev/")
    )
    assert str(metadata.issuer).rstrip("/") == "https://login.semgrep.dev"
    assert metadata.device_authorization_endpoint
    # A genuinely different issuer still fails closed.
    class _Other(_Client):
        async def get(self, url):
            original = _Resp().json()  # snapshot before overriding, or the override recurses
            resp = _Resp()
            resp.json = lambda: {**original, "issuer": "https://evil.example.com"}
            return resp

    with pytest.raises(OAuthFlowError):
        asyncio.run(
            device._device_metadata(_Other(), "https://mcp.semgrep.ai/mcp", "https://login.semgrep.dev/")
        )


class _BaseProvider:
    """Stand-in for the SDK provider: exposes a context whose own rule sends `resource`."""

    def __init__(self, **kwargs):
        self.context = types.SimpleNamespace(
            protected_resource_metadata=object(),
            should_include_resource_param=lambda protocol_version=None: True,
        )


class _Provider(HermesProviderMixin, _BaseProvider):
    pass


def test_resource_param_opt_out_reaches_the_context():
    """``oauth.resource_param: false`` must silence the *context* rule the SDK actually consults.

    Semgrep advertises protected-resource metadata (so the SDK sends RFC 8707 `resource`) yet answers
    ``invalid_target`` to every value (measured 2026-10-26); the opt-out has to land on the context,
    because ``OAuthClientProvider`` builds the authorization URL from ``context.should_include_resource_param``.
    """
    opted_out = _Provider(resource_param=False)
    assert opted_out.context.should_include_resource_param() is False
    assert opted_out.should_include_resource_param() is False

    default = _Provider()
    assert default.context.should_include_resource_param() is True
