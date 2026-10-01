"""Regression for #111135: Figma advertises RFC 9207 ``iss`` support and then omits ``iss`` from the
redirect, so the SDK rejected every valid authorization code. Only that issuer is tolerated."""

import asyncio
from types import SimpleNamespace

import pytest

# mcp.client.auth.utils also exists in the 1.x line — only the iss validator
# below is 2.0-only, so a module presence check lets a stale SDK through and
# the unguarded imports fail at collection time instead of skipping.
_mcp_auth_utils = pytest.importorskip(
    "mcp.client.auth.utils", reason="mcp 2.x SDK not installed"
)
if not hasattr(_mcp_auth_utils, "validate_authorization_response_iss"):
    pytest.skip(
        "mcp 2.x SDK not installed (older mcp distribution present)",
        allow_module_level=True,
    )

from mcp.client.auth.oauth2 import OAuthClientProvider  # noqa: E402
from mcp.client.auth.utils import validate_authorization_response_iss  # noqa: E402
from mcp.shared.auth import AuthorizationCodeResult, OAuthMetadata  # noqa: E402

from tools.mcp_oauth_provider import HermesProviderMixin  # noqa: E402


class _Provider(HermesProviderMixin, OAuthClientProvider):
    pass


def _provider_for(issuer: str, *, iss_in_redirect: str | None) -> _Provider:
    provider = _Provider.__new__(_Provider)
    meta = OAuthMetadata(
        issuer=issuer,
        authorization_endpoint=f"{issuer}/oauth",
        token_endpoint=f"{issuer}/token",
        authorization_response_iss_parameter_supported=True,
    )

    async def callback():
        return AuthorizationCodeResult(code="c0de", state="st", iss=iss_in_redirect)

    provider.context = SimpleNamespace(oauth_metadata=meta, callback_handler=callback,
                                       client_info=SimpleNamespace(grant_types=["authorization_code"]))
    provider._hermes_oauth_flow = "browser"
    return provider


def _redirect_passes_sdk_check(provider: _Provider) -> bool:
    provider._tolerate_missing_iss_for_known_server()
    result = asyncio.run(provider.context.callback_handler())
    try:
        validate_authorization_response_iss(result.iss, provider.context.oauth_metadata)
    except Exception:
        return False
    return True


@pytest.mark.parametrize("issuer, iss, expected", [
    ("https://api.figma.com", None, True),        # the advertised-but-omitted case Figma ships
    ("https://api.figma.com", "https://evil.example", False),  # a wrong iss is still rejected
    ("https://auth.example.com", None, False),   # every other server keeps the strict RFC 9207 rule
])
def test_missing_iss_is_tolerated_for_figma_only(issuer, iss, expected):
    assert _redirect_passes_sdk_check(_provider_for(issuer, iss_in_redirect=iss)) is expected
