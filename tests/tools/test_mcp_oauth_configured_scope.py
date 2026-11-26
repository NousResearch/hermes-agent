"""A configured ``oauth.scope`` is authoritative over the SDK's scope selection (#142477).

The MCP SDK's 401 branch reassigns ``client_metadata.scope`` from server metadata (WWW-Authenticate
hint, else PRM / AS ``scopes_supported``, plus ``offline_access``) before dynamic client registration
and the authorization URL read it. A server whose metadata advertises more scopes than its
registration accepts then refuses every login, and ``oauth.scope`` reads as broken.
"""
import json

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK required for OAuth support")

SELECTED = "mcp:read mcp:write admin offline_access"  # what SDK discovery assigns
CONFIGURED = "mcp:read offline_access"


def _metadata(cfg):
    from tools.mcp_oauth import _build_client_metadata
    return _build_client_metadata({"_resolved_port": 43123, **cfg})


def test_configured_scope_reaches_registration_and_authorization():
    """Invariant: after the SDK's Step-3 reassignment, the DCR body and the scope the
    authorization URL is built from both carry the configured value."""
    from mcp.client.auth.utils import create_client_registration_request
    from mcp.shared.auth import OAuthMetadata

    metadata = _metadata({"scope": CONFIGURED})
    metadata.scope = SELECTED  # SDK oauth2.py Step 3 / step-up union
    assert metadata.scope == CONFIGURED

    asm = OAuthMetadata.model_validate({
        "issuer": "https://as.example", "authorization_endpoint": "https://as.example/authorize",
        "token_endpoint": "https://as.example/token", "registration_endpoint": "https://as.example/register"})
    request = create_client_registration_request(asm, metadata, "https://as.example")
    assert json.loads(request.content)["scope"] == CONFIGURED
    metadata.client_name = "renamed"  # every other field stays writable
    assert metadata.client_name == "renamed"


def test_unset_scope_leaves_sdk_selection_untouched():
    metadata = _metadata({})
    metadata.scope = SELECTED
    assert metadata.scope == SELECTED
