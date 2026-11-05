"""Configured ``oauth.scope`` outranks the SDK's discovery-based scope selection (#142477).

The MCP SDK's 401-discovery branch (mcp 2.x oauth2.py Step 3) assigns
``context.client_metadata.scope`` from the WWW-Authenticate hint, else PRM/AS
``scopes_supported``, with no case for a scope the client already set. When the
authorization server advertises more scopes than its clients may present, every
authorization fails with ``invalid_scope`` and ``oauth.scope`` is dead weight.
Hermes re-asserts the configured scope immediately before the SDK builds the
authorization URL.
"""
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK 1.26.0+ required for OAuth support",
)

SELECTED = "web-origins openid profile ext groups"  # what SDK discovery would pick
CONFIGURED = "openid profile offline_access"


@pytest.mark.asyncio
async def test_configured_scope_survives_the_sdks_step3_overwrite():
    """Invariant: the URL the SDK builds carries the configured scope, not the discovered one."""
    from tools.mcp_oauth_manager import HermesMCPOAuthProvider

    provider = HermesMCPOAuthProvider(
        server_url="https://mcp.example/mcp", server_name="s",
        client_metadata=MagicMock(), storage=MagicMock(),
        redirect_handler=MagicMock(), callback_handler=MagicMock())
    seen: list[str] = []

    async def fake_grant():  # stands in for the SDK grant: reads what the URL builder would send
        seen.append(provider.context.client_metadata.scope)  # SDK URL builder reads this
        return ("code", "verifier")

    provider._perform_authorization_code_grant = fake_grant  # bind wraps this stand-in
    provider._hermes_bind_configured_scope({"scope": CONFIGURED})
    # SDK Step 3 (401-discovery branch) overwrites the scope immediately before the grant runs.
    provider.context.client_metadata.scope = SELECTED
    await provider._perform_authorization_code_grant()
    assert seen == [CONFIGURED]


def test_unset_scope_leaves_sdk_selection_untouched():
    from tools.mcp_oauth_manager import HermesMCPOAuthProvider

    provider = HermesMCPOAuthProvider(
        server_url="https://mcp.example/mcp", server_name="s",
        client_metadata=MagicMock(), storage=MagicMock(),
        redirect_handler=MagicMock(), callback_handler=MagicMock())
    provider._hermes_bind_configured_scope({})
    assert provider._hermes_configured_scope == ""
    assert not getattr(provider, "_hermes_scope_pin_installed", False)
    provider.context.client_metadata.scope = SELECTED
    provider._pin_configured_scope()
    assert provider.context.client_metadata.scope == SELECTED


def test_rebind_never_stacks_wrappers():
    from tools.mcp_oauth_manager import HermesMCPOAuthProvider

    provider = HermesMCPOAuthProvider(
        server_url="https://mcp.example/mcp", server_name="s",
        client_metadata=MagicMock(), storage=MagicMock(),
        redirect_handler=MagicMock(), callback_handler=MagicMock())
    provider._hermes_bind_configured_scope({"scope": CONFIGURED})
    first = provider._perform_authorization_code_grant
    provider._hermes_bind_configured_scope({"scope": "other scope"})
    assert provider._perform_authorization_code_grant is first
    assert provider._hermes_configured_scope == "other scope"  # value refreshes, wrapper does not stack


def test_pin_survives_a_refusing_metadata_object():
    """A metadata model that rejects the write must degrade to SDK selection, not break auth."""
    from tools.mcp_oauth_manager import HermesMCPOAuthProvider

    provider = HermesMCPOAuthProvider(
        server_url="https://mcp.example/mcp", server_name="s",
        client_metadata=MagicMock(), storage=MagicMock(),
        redirect_handler=MagicMock(), callback_handler=MagicMock())
    provider._hermes_bind_configured_scope({"scope": CONFIGURED})

    class _Frozen:
        scope = SELECTED

        def __setattr__(self, name, value):
            raise TypeError("frozen")

    provider.context.client_metadata = _Frozen()
    provider._pin_configured_scope()  # must not raise
    assert provider.context.client_metadata.scope == SELECTED
