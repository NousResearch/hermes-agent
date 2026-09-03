"""Narrow compatibility coverage for malformed provider OAuth metadata."""

import json

import pytest


def _metadata(issuer: str):
    from mcp.shared.auth import OAuthMetadata

    return OAuthMetadata(
        issuer=issuer,
        authorization_endpoint=f"{issuer.rstrip('/')}/authorize",
        token_endpoint=f"{issuer.rstrip('/')}/token",
        response_types_supported=["code"],
    )


def test_trailing_slash_compat_is_strictly_byte_preserving():
    from tools.mcp_oauth_manager import _differs_only_by_one_trailing_slash

    assert _differs_only_by_one_trailing_slash(
        "https://secure.indeed.com",
        "https://secure.indeed.com/",
    )
    assert not _differs_only_by_one_trailing_slash(
        "https://SECURE.indeed.com",
        "https://secure.indeed.com/",
    )
    assert not _differs_only_by_one_trailing_slash(
        "https://secure.indeed.com/oauth",
        "https://secure.indeed.com/",
    )
    assert not _differs_only_by_one_trailing_slash(
        "https://secure.indeed.com?tenant=x",
        "https://secure.indeed.com/",
    )


def test_sdk_validator_remains_strict_without_provider_opt_in():
    from mcp.client.auth import oauth2
    from mcp.client.auth.oauth2 import OAuthFlowError

    import tools.mcp_oauth_manager  # noqa: F401 - installs scoped wrapper

    with pytest.raises(OAuthFlowError, match="issuer mismatch"):
        oauth2.validate_metadata_issuer(
            _metadata("https://secure.indeed.com"),
            "https://secure.indeed.com/",
        )


def test_sdk_validator_accepts_only_the_scoped_trailing_slash_case():
    from mcp.client.auth import oauth2
    from mcp.client.auth.oauth2 import OAuthFlowError
    from tools.mcp_oauth_manager import _issuer_trailing_slash_compat_enabled

    token = _issuer_trailing_slash_compat_enabled.set(True)
    try:
        oauth2.validate_metadata_issuer(
            _metadata("https://secure.indeed.com"),
            "https://secure.indeed.com/",
        )
        with pytest.raises(OAuthFlowError, match="issuer mismatch"):
            oauth2.validate_metadata_issuer(
                _metadata("https://attacker.example"),
                "https://secure.indeed.com/",
            )
    finally:
        _issuer_trailing_slash_compat_enabled.reset(token)


def _registration_request_body():
    from mcp.client.auth import oauth2
    from mcp.shared.auth import OAuthClientMetadata

    metadata = _metadata("https://secure.indeed.com")
    metadata.registration_endpoint = "https://secure.indeed.com/register"
    client = OAuthClientMetadata(
        client_name="Hermes Agent",
        redirect_uris=["http://127.0.0.1:27892/callback"],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        scope="job_seeker.jobs.search offline_access",
    )
    request = oauth2.create_client_registration_request(
        metadata,
        client,
        "https://secure.indeed.com",
    )
    return json.loads(request.content), client


def test_registration_scope_is_preserved_without_provider_opt_in():
    import tools.mcp_oauth_manager  # noqa: F401 - installs scoped wrapper

    body, client = _registration_request_body()

    assert body["scope"] == "job_seeker.jobs.search offline_access"
    assert client.scope == "job_seeker.jobs.search offline_access"


def test_registration_scope_omission_is_scoped_and_keeps_authorization_scope():
    from tools.mcp_oauth_manager import _registration_omit_scope_enabled

    token = _registration_omit_scope_enabled.set(True)
    try:
        body, client = _registration_request_body()
    finally:
        _registration_omit_scope_enabled.reset(token)

    assert "scope" not in body
    assert client.scope == "job_seeker.jobs.search offline_access"
