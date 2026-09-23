"""Root-slash issuer equivalence in the MCP OAuth browser flow.

Semgrep's MCP connector advertises `authorization_servers: ["https://login.semgrep.dev/"]` in its
protected-resource document while its own authorization-server document says
`"issuer": "https://login.semgrep.dev"`. The SDK's `validate_metadata_issuer` compares those as
simple strings (RFC 3986 §6.2.1) and rejects the pair, so the browser flow could not complete at
all — while Hermes' own device-flow path normalizes exactly this difference
(`tools/mcp_oauth_device.py`, documented there for Google). These tests pin that the browser flow
now agrees, AND that nothing wider was relaxed.
"""

from __future__ import annotations

from tools.mcp_oauth_provider import (
    metadata_issued_by_origin,
    metadata_issuer_root_slash_equivalent,
)


class _Metadata:
    def __init__(self, issuer: str):
        self.issuer = issuer


class _Response:
    def __init__(self, url: str, status_code: int = 200):
        self.url = url
        self.status_code = status_code


DERIVED = "https://login.semgrep.dev/.well-known/oauth-authorization-server"


def test_the_semgrep_pair_is_accepted():
    """Advertised identifier has the root slash; the document's issuer does not."""
    accepted = metadata_issuer_root_slash_equivalent(
        _Metadata("https://login.semgrep.dev"),      # document issuer, no slash
        "https://login.semgrep.dev/",               # advertised identifier, slash
        _Response(DERIVED),
    )
    assert accepted is True


def test_the_reverse_direction_is_also_accepted():
    """The equivalence must not depend on which side carries the slash."""
    accepted = metadata_issuer_root_slash_equivalent(
        _Metadata("https://as.example/"),
        "https://as.example",
        _Response("https://as.example/.well-known/oauth-authorization-server"),
    )
    assert accepted is True


def test_a_different_host_is_still_rejected():
    assert metadata_issuer_root_slash_equivalent(
        _Metadata("https://evil.example"),
        "https://login.semgrep.dev/",
        _Response(DERIVED),
    ) is False


def test_a_different_port_is_still_rejected():
    assert metadata_issuer_root_slash_equivalent(
        _Metadata("https://login.semgrep.dev:8443"),
        "https://login.semgrep.dev/",
        _Response(DERIVED),
    ) is False


def test_a_different_scheme_is_still_rejected():
    assert metadata_issuer_root_slash_equivalent(
        _Metadata("http://login.semgrep.dev"),
        "https://login.semgrep.dev/",
        _Response(DERIVED),
    ) is False


def test_a_sub_path_issuer_is_still_rejected():
    """Only a ROOT slash may differ — a path is a different authorization server."""
    assert metadata_issuer_root_slash_equivalent(
        _Metadata("https://login.semgrep.dev/tenant"),
        "https://login.semgrep.dev/",
        _Response(DERIVED),
    ) is False


def test_a_path_scoped_advertised_server_is_not_this_predicate_s_business():
    """Path-scoped servers are covered by metadata_issued_by_origin on their own terms."""
    assert metadata_issuer_root_slash_equivalent(
        _Metadata("https://example.com"),
        "https://example.com/oauth/mcp",
        _Response("https://example.com/.well-known/oauth-authorization-server/oauth/mcp"),
    ) is False


def test_a_redirected_document_is_rejected():
    """The document must come from the derived well-known URL, so a redirect target never matches."""
    assert metadata_issuer_root_slash_equivalent(
        _Metadata("https://login.semgrep.dev"),
        "https://login.semgrep.dev/",
        _Response("https://attacker.example/.well-known/oauth-authorization-server"),
    ) is False


def test_the_oidc_fallback_document_is_rejected():
    """Only the RFC 8414 §3.1 derived URL counts, never the OIDC fallback."""
    assert metadata_issuer_root_slash_equivalent(
        _Metadata("https://login.semgrep.dev"),
        "https://login.semgrep.dev/",
        _Response("https://login.semgrep.dev/.well-known/openid-configuration"),
    ) is False


def test_a_non_200_response_is_rejected():
    assert metadata_issuer_root_slash_equivalent(
        _Metadata("https://login.semgrep.dev"),
        "https://login.semgrep.dev/",
        _Response(DERIVED, status_code=404),
    ) is False


def test_credentials_or_query_in_the_advertised_identifier_are_rejected():
    for bad in ("https://user:pw@login.semgrep.dev/",
                "https://login.semgrep.dev/?x=1",
                "https://login.semgrep.dev/#frag"):
        assert metadata_issuer_root_slash_equivalent(
            _Metadata("https://login.semgrep.dev"), bad, _Response(DERIVED),
        ) is False, bad


def test_no_advertised_identifier_is_rejected():
    assert metadata_issuer_root_slash_equivalent(
        _Metadata("https://login.semgrep.dev"), None, _Response(DERIVED),
    ) is False


def test_the_origin_predicate_is_unchanged_by_this_change():
    """The pre-existing shim keeps its own contract; the two are independent."""
    # path-scoped: accepted by the origin predicate, not by the new one
    assert metadata_issued_by_origin(
        _Metadata("https://example.com"),
        "https://example.com/oauth/mcp",
        _Response("https://example.com/.well-known/oauth-authorization-server/oauth/mcp"),
    ) is True
    # root-scoped: not its business
    assert metadata_issued_by_origin(
        _Metadata("https://login.semgrep.dev"),
        "https://login.semgrep.dev/",
        _Response(DERIVED),
    ) is False
