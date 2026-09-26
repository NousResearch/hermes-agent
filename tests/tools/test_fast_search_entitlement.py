"""Managed Fast Search eligibility for registered Portal accounts without credits.

Regression for #122394: the generic Tool Gateway credit gate blocked web_search on the
managed Perplexity route for a registered account with no usable credits, although Fast
Search is free across all Portal tiers. Fast Search gets its own eligibility; every other
managed capability keeps the generic credit gate.
"""

import json

from hermes_cli.nous_account import NousPortalAccountInfo, NousToolAccessInfo


def _zero_credit_account() -> NousPortalAccountInfo:
    """Canned /api/oauth/account snapshot: registered, logged in, no usable credits."""
    return NousPortalAccountInfo(
        logged_in=True, source="account_api", fresh=True,
        paid_service_access=False,
        tool_access=NousToolAccessInfo(enabled=False, coverage={}),
    )


def _managed_env(monkeypatch, tmp_path):
    from hermes_cli.config import atomic_config_write

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {"web": {"backend": "nous", "keyless_rescue": False}})
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "test-nous-token")


def test_zero_credit_account_serves_managed_fast_search(monkeypatch, tmp_path):
    """web_search succeeds through the managed backend for a registered zero-credit account."""
    from hermes_cli import nous_account
    from tests.tools.conftest import register_all_web_providers
    from tools import web_tools

    import plugins.web.perplexity.provider as perplexity

    _managed_env(monkeypatch, tmp_path)
    monkeypatch.setattr(nous_account, "get_nous_portal_account_info", lambda **kw: _zero_credit_account())
    register_all_web_providers()

    # The canned account must read as ineligible to the generic Tool Gateway gate.
    from tools.tool_backend_helpers import managed_nous_tools_enabled
    assert managed_nous_tools_enabled() is False

    # Canned provider layer: record how the call was routed, answer with a canned response.
    calls = []

    def fake_request(endpoint, payload, gateway=None):
        calls.append((endpoint, payload, gateway))
        return {"results": [{"title": "Local fixture", "url": "https://example.test", "snippet": "fixture"}]}

    monkeypatch.setattr(perplexity, "_perplexity_request", fake_request)

    result = json.loads(web_tools.web_search_tool("local fixture", limit=3))

    assert result["success"] is True, result
    assert result["data"]["web"][0]["url"] == "https://example.test"
    # Served by the managed backend: gateway resolved, Fast Search payload.
    endpoint, payload, gateway = calls[0]
    assert endpoint == "search"
    assert payload["search_type"] == "fast"
    assert gateway is not None
    assert gateway.nous_user_token == "test-nous-token"
    assert perplexity.PerplexityWebSearchProvider().is_available() is True


def test_zero_credit_account_stays_blocked_for_other_managed_capabilities(monkeypatch, tmp_path):
    """The generic credit gate still gates every other capability for the same account."""
    from hermes_cli import nous_account
    from tools.managed_tool_gateway import resolve_managed_tool_gateway
    from tools.tool_backend_helpers import managed_nous_tools_enabled

    _managed_env(monkeypatch, tmp_path)
    monkeypatch.setattr(nous_account, "get_nous_portal_account_info", lambda **kw: _zero_credit_account())

    assert managed_nous_tools_enabled() is False
    # Managed extract (Firecrawl) and paid tools keep the generic entitlement block.
    assert resolve_managed_tool_gateway("firecrawl") is None
    assert resolve_managed_tool_gateway("modal") is None
    assert resolve_managed_tool_gateway("fal-queue") is None
