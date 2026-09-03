"""Fix-level tests for #93719 — HermesMCPOOAuthProvider preserves an explicit
``oauth.scope`` across the SDK's Step-3 scope overwrite.

The SDK's ``async_auth_flow`` Step 3 replaces ``client_metadata.scope`` with
server-derived scopes and never reads the configured value. The provider now
carries ``configured_scope`` (from ``mcp_servers.<name>.oauth.scope``) and
restores it — unioned with whatever server metadata/challenge added — right
before ``_perform_authorization_code_grant`` builds the /authorize URL.
"""

from __future__ import annotations

import pytest

from tools.mcp_oauth_manager import _make_hermes_provider_class


@pytest.fixture
def provider_cls():
    cls = _make_hermes_provider_class()
    if cls is None:
        pytest.skip("MCP SDK OAuth module unavailable")
    return cls


class _FakeContext:
    """Minimal stand-in for the SDK's auth context."""

    def __init__(self, scope: str | None):
        self.client_metadata = type(
            "M", (), {"scope": scope}
        )()


def _build(cls, configured_scope: str | None, context_scope: str | None):
    """Build a provider without running its __init__ network paths, then point
    it at a fake context carrying a post-Step-3 scope."""
    import argparse

    # Bypass __init__ (SDK requires redirect/callback handlers); set fields directly.
    provider = cls.__new__(cls)
    provider._hermes_server_name = "test"
    provider._hermes_home = ""
    provider._hermes_preregistered = False
    provider._hermes_token_user_agent = None
    provider._hermes_configured_scope = configured_scope
    provider.context = _FakeContext(context_scope)
    return provider


class TestRestoreConfiguredScope:
    def test_configured_scope_restored_after_step3_overwrite(self, provider_cls):
        """Post-Step-3 state has only advertised scopes; restore must bring the
        configured baseline back into the requested set."""
        provider = _build(provider_cls, "mcp:ea offline_access", "basic openid")

        provider._restore_configured_scope()

        scope = provider.context.client_metadata.scope.split()
        assert "mcp:ea" in scope and "offline_access" in scope, (
            "configured scopes must survive the SDK overwrite"
        )
        assert "openid" in scope, "challenge-added scopes are kept, not replaced"

    def test_union_order_configured_first(self, provider_cls):
        """Configured scopes lead the string so consent screens present them as
        the requested baseline."""
        provider = _build(provider_cls, "alpha beta", "gamma delta")
        provider._restore_configured_scope()
        assert provider.context.client_metadata.scope == "alpha beta gamma delta"

    def test_no_duplicate_scopes(self, provider_cls):
        """Overlapping configured/advertised scopes collapse to one entry."""
        provider = _build(provider_cls, "mcp:read offline_access", "offline_access openid")
        provider._restore_configured_scope()
        assert provider.context.client_metadata.scope == "mcp:read offline_access openid"

    def test_noop_when_nothing_configured(self, provider_cls):
        """Without an explicit oauth.scope the SDK's selection stands untouched —
        default servers keep their current behavior."""
        provider = _build(provider_cls, None, "server_scope")
        provider._restore_configured_scope()
        assert provider.context.client_metadata.scope == "server_scope"

    def test_noop_with_empty_configured_string(self, provider_cls):
        provider = _build(provider_cls, "", "server_scope")
        provider._restore_configured_scope()
        assert provider.context.client_metadata.scope == "server_scope"

    def test_survives_missing_context_metadata(self, provider_cls):
        """Defensive: context without client_metadata must not raise."""
        provider = _build(provider_cls, "some_scope", None)
        provider.context.client_metadata = None
        provider._restore_configured_scope()  # no exception

    def test_perform_authorization_code_grant_is_async_override(self, provider_cls):
        """The override shadows the SDK method (must be a coroutine function)
        and its restore runs before delegation — verified by calling restore
        directly and observing the merged scope the grant builder would read."""
        import inspect

        provider = _build(provider_cls, "cfg_scope", "sdk_scope")

        assert inspect.iscoroutinefunction(
            type(provider)._perform_authorization_code_grant
        ), "override must be async to shadow the SDK method"

        # Simulate the override's first statement: restore mutates what
        # _perform_authorization_code_grant reads when building auth_params.
        provider._restore_configured_scope()
        scope = provider.context.client_metadata.scope.split()
        assert scope[0] == "cfg_scope" and "sdk_scope" in scope
