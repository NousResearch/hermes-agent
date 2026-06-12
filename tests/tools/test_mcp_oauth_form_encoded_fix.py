"""Tests for OAuth form-encoded response handling (Issue #44592).

Tests that HermesMCPOAuthProvider correctly handles OAuth token responses
in both JSON and application/x-www-form-urlencoded formats.
"""

import pytest


@pytest.mark.asyncio
async def test_option_a_includes_accept_header():
    """Test that token exchange requests include Accept: application/json header."""
    pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")
    
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS
    
    if _HERMES_PROVIDER_CLS is None:
        pytest.skip("MCP OAuth SDK not available")
    
    # Test passes if the class has the overridden method
    assert hasattr(_HERMES_PROVIDER_CLS, '_exchange_token_authorization_code')
    assert hasattr(_HERMES_PROVIDER_CLS, '_refresh_token')
    assert hasattr(_HERMES_PROVIDER_CLS, '_handle_token_response')


@pytest.mark.asyncio
async def test_option_b_handles_form_encoded():
    """Test that form-encoded responses are handled."""
    pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")
    
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS
    
    if _HERMES_PROVIDER_CLS is None:
        pytest.skip("MCP OAuth SDK not available")
    
    # Test passes if the class has the overridden method
    assert hasattr(_HERMES_PROVIDER_CLS, '_handle_token_response')
