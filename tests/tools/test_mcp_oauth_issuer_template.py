"""Tests for ``metadata_issuer_template_matches`` (Microsoft Entra multi-tenant issuer templates).

Entra's multi-tenant endpoints publish an OIDC discovery document whose ``issuer`` keeps an
unsubstituted ``{tenantid}`` placeholder, which the SDK's exact-string check (RFC 8414 §3.3)
rejects. The shim accepts that one shape and must keep rejecting everything else.
"""
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.mcp_oauth_provider import metadata_issuer_template_matches  # noqa: E402

ENTRA = "https://login.microsoftonline.com"
COMMON = f"{ENTRA}/common/v2.0"
COMMON_DOC = f"{ENTRA}/common/v2.0/.well-known/openid-configuration"
TEMPLATED_ISSUER = f"{ENTRA}/{{tenantid}}/v2.0"


def _response(url, status_code=200):
    return SimpleNamespace(url=url, status_code=status_code)


def _metadata(issuer):
    return SimpleNamespace(issuer=issuer)


def _sdk_metadata(issuer):
    """Issuer as the SDK actually presents it.

    ``OAuthMetadata`` stores the issuer as a pydantic ``AnyHttpUrl``, and ``str()`` of that
    percent-encodes the placeholder braces (``{tenantid}`` -> ``%7Btenantid%7D``). Tests that
    build the issuer as a plain string never see that form and would pass against a matcher
    that is broken in production, so the real Entra shapes go through the model.
    """
    from mcp.shared.auth import OAuthMetadata
    return OAuthMetadata.model_validate({
        "issuer": issuer,
        "authorization_endpoint": f"{ENTRA}/common/oauth2/v2.0/authorize",
        "token_endpoint": f"{ENTRA}/common/oauth2/v2.0/token",
        "response_types_supported": ["code"],
    })


def test_sdk_percent_encodes_placeholder_braces():
    """Guards the assumption the matcher is built on."""
    assert str(_sdk_metadata(TEMPLATED_ISSUER).issuer) == f"{ENTRA}/%7Btenantid%7D/v2.0"


def test_accepts_entra_common_via_sdk_model():
    """The real end-to-end shape: issuer as the SDK hands it to the matcher."""
    assert metadata_issuer_template_matches(
        _sdk_metadata(TEMPLATED_ISSUER), COMMON, _response(COMMON_DOC))


def test_accepts_organizations_via_sdk_model():
    assert metadata_issuer_template_matches(
        _sdk_metadata(TEMPLATED_ISSUER),
        f"{ENTRA}/organizations/v2.0",
        _response(f"{ENTRA}/organizations/v2.0/.well-known/openid-configuration"))


def test_rejects_foreign_origin_via_sdk_model():
    assert not metadata_issuer_template_matches(
        _sdk_metadata("https://evil.example.com/{tenantid}/v2.0"), COMMON, _response(COMMON_DOC))


def test_rejects_encoded_slash_in_placeholder():
    """A percent-encoded slash must not smuggle extra path structure into one segment."""
    assert not metadata_issuer_template_matches(
        _metadata(f"{ENTRA}/%7Bx/y%7D/v2.0"), COMMON, _response(COMMON_DOC))


def test_accepts_entra_common_tenantid_placeholder():
    """The real Business Central / Entra pair, served from the derived discovery URL."""
    assert metadata_issuer_template_matches(
        _metadata(TEMPLATED_ISSUER), COMMON, _response(COMMON_DOC))


def test_accepts_organizations_endpoint():
    assert metadata_issuer_template_matches(
        _metadata(TEMPLATED_ISSUER),
        f"{ENTRA}/organizations/v2.0",
        _response(f"{ENTRA}/organizations/v2.0/.well-known/openid-configuration"))


def test_accepts_rfc8414_style_discovery_url():
    assert metadata_issuer_template_matches(
        _metadata(TEMPLATED_ISSUER), COMMON,
        _response(f"{ENTRA}/.well-known/oauth-authorization-server/common/v2.0"))


@pytest.mark.parametrize("issuer", [
    "https://evil.example.com/{tenantid}/v2.0",          # different host
    "http://login.microsoftonline.com/{tenantid}/v2.0",  # downgraded scheme
    "https://login.microsoftonline.com.evil.com/{tenantid}/v2.0",  # suffix host
])
def test_rejects_foreign_origin(issuer):
    """A placeholder must never let the issuer move to another origin."""
    assert not metadata_issuer_template_matches(
        _metadata(issuer), COMMON, _response(COMMON_DOC))


def test_rejects_two_templated_segments():
    """Only the single tenant discriminator may be templated."""
    assert not metadata_issuer_template_matches(
        _metadata(f"{ENTRA}/{{tenantid}}/{{version}}"), COMMON, _response(COMMON_DOC))


def test_rejects_segment_count_mismatch():
    assert not metadata_issuer_template_matches(
        _metadata(f"{ENTRA}/{{tenantid}}/v2.0/extra"), COMMON, _response(COMMON_DOC))


def test_rejects_non_placeholder_difference():
    """A plain differing segment is a genuine mismatch, not a template."""
    assert not metadata_issuer_template_matches(
        _metadata(f"{ENTRA}/consumers/v2.0"), COMMON, _response(COMMON_DOC))


def test_rejects_partial_segment_placeholder():
    """The placeholder must be a whole segment, not a substring of one."""
    assert not metadata_issuer_template_matches(
        _metadata(f"{ENTRA}/tenant-{{id}}/v2.0"), COMMON, _response(COMMON_DOC))


def test_rejects_empty_placeholder():
    assert not metadata_issuer_template_matches(
        _metadata(f"{ENTRA}/{{}}/v2.0"), COMMON, _response(COMMON_DOC))


def test_rejects_redirected_document():
    """response.url is the final URL after redirects; only the derived location qualifies."""
    assert not metadata_issuer_template_matches(
        _metadata(TEMPLATED_ISSUER), COMMON,
        _response("https://evil.example.com/.well-known/openid-configuration"))


def test_rejects_undeclared_discovery_path():
    assert not metadata_issuer_template_matches(
        _metadata(TEMPLATED_ISSUER), COMMON, _response(f"{ENTRA}/common/v2.0/metadata.json"))


def test_rejects_non_200():
    assert not metadata_issuer_template_matches(
        _metadata(TEMPLATED_ISSUER), COMMON, _response(COMMON_DOC, status_code=404))


def test_rejects_missing_auth_server_url():
    assert not metadata_issuer_template_matches(_metadata(TEMPLATED_ISSUER), None, _response(COMMON_DOC))
    assert not metadata_issuer_template_matches(_metadata(TEMPLATED_ISSUER), "", _response(COMMON_DOC))


def test_rejects_root_path_authorization_server():
    """A root-path server has no segment that could legitimately be templated."""
    assert not metadata_issuer_template_matches(
        _metadata(f"{ENTRA}/{{tenantid}}"), ENTRA,
        _response(f"{ENTRA}/.well-known/openid-configuration"))


def test_rejects_dot_segments():
    assert not metadata_issuer_template_matches(
        _metadata(TEMPLATED_ISSUER), f"{ENTRA}/common/../common/v2.0",
        _response(COMMON_DOC))


@pytest.mark.parametrize("auth_server_url", [
    f"https://user:pw@login.microsoftonline.com/common/v2.0",
    f"{ENTRA}/common/v2.0?x=1",
    f"{ENTRA}/common/v2.0#frag",
])
def test_rejects_userinfo_query_fragment(auth_server_url):
    assert not metadata_issuer_template_matches(
        _metadata(TEMPLATED_ISSUER), auth_server_url, _response(COMMON_DOC))


def test_exact_match_is_not_this_shims_job():
    """An exact match needs no template; the SDK's own check already accepts it."""
    assert not metadata_issuer_template_matches(
        _metadata(COMMON), COMMON, _response(COMMON_DOC))


class _FakeResponse:
    """Minimal httpx-like response the shim can read and re-instantiate."""

    def __init__(self, status_code, *, url="", body=b"", request=None):
        self.status_code = status_code
        self.url = url
        self._body = body
        self.request = request

    async def aread(self):
        return self._body


@pytest.mark.asyncio
async def test_shim_accepts_entra_oidc_suffix_discovery_document():
    """End-to-end through the shim itself.

    The matcher being right is not enough: the shim first gates on the REQUEST path, and
    Entra serves its document at the OIDC suffix layout ``/common/v2.0/.well-known/
    openid-configuration``. A prefix-only gate returns the response untouched and the SDK
    still raises the mismatch, which is exactly the bug this guards.
    """
    import json

    from tools.mcp_oauth_provider import HermesProviderMixin

    body = json.dumps({
        "issuer": TEMPLATED_ISSUER,
        "authorization_endpoint": f"{ENTRA}/common/oauth2/v2.0/authorize",
        "token_endpoint": f"{ENTRA}/common/oauth2/v2.0/token",
        "response_types_supported": ["code"],
    }).encode()

    provider = HermesProviderMixin.__new__(HermesProviderMixin)
    provider.context = SimpleNamespace(auth_server_url=COMMON, oauth_metadata=None)
    provider._hermes_logger = logging.getLogger("test")

    response = _FakeResponse(200, url=COMMON_DOC, body=body,
                             request=SimpleNamespace(url=COMMON_DOC))
    out = await provider._hermes_accept_origin_issued_metadata(response)

    assert out.status_code == 204, "shim did not consume the document; SDK will reject it"
    assert provider.context.oauth_metadata is not None
    assert str(provider.context.oauth_metadata.issuer) == f"{ENTRA}/%7Btenantid%7D/v2.0"


@pytest.mark.asyncio
async def test_shim_ignores_unrelated_suffix_path_on_another_host():
    """The OIDC suffix form is accepted only at the ADVERTISED server's own path.

    Guards the tightening that keeps `test_mcp_oauth_issuer_origin`'s "unrelated-prefix"
    case intact: a `/proxy/.well-known/...` that is not the advertised server's path must
    still be returned untouched and never read.
    """
    from tools.mcp_oauth_provider import HermesProviderMixin

    class _NeverRead(_FakeResponse):
        async def aread(self):
            raise AssertionError("unrelated suffix path must not be consumed")

    provider = HermesProviderMixin.__new__(HermesProviderMixin)
    provider.context = SimpleNamespace(auth_server_url=COMMON, oauth_metadata=None)
    provider._hermes_logger = logging.getLogger("test")

    url = "https://as.example/proxy/.well-known/oauth-authorization-server"
    response = _NeverRead(200, url=url, request=SimpleNamespace(url=url))
    assert await provider._hermes_accept_origin_issued_metadata(response) is response
    assert provider.context.oauth_metadata is None


@pytest.mark.asyncio
async def test_shim_ignores_non_discovery_responses():
    """A resource response must be returned untouched and never read (SSE would hang)."""
    from tools.mcp_oauth_provider import HermesProviderMixin

    class _NeverRead(_FakeResponse):
        async def aread(self):
            raise AssertionError("shim must not read a non-discovery response")

    provider = HermesProviderMixin.__new__(HermesProviderMixin)
    provider.context = SimpleNamespace(auth_server_url=COMMON, oauth_metadata=None)
    provider._hermes_logger = logging.getLogger("test")

    stream_url = "https://mcp.businesscentral.dynamics.com/v2/mcp"
    response = _NeverRead(200, url=stream_url, request=SimpleNamespace(url=stream_url))
    assert await provider._hermes_accept_origin_issued_metadata(response) is response
    assert provider.context.oauth_metadata is None
